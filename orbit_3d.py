"""
3D Orbital Trajectory Predictor & 2-Stage XGBoost Surrogate
===========================================================
Simulates a two-stage launch vehicle in 3D Earth-Centered Inertial (ECI)
coordinates (X, Y, Z) using scipy.integrate.solve_ivp, coupled with a
two-stage XGBoost surrogate model:
  - Stage 1: XGBClassifier predicting LEO insertion success
  - Stage 2: XGBRegressors (trained strictly on successful LEO orbits)
             predicting final altitude, final speed, inclination, and Max-Q.

Author: TUM Data Science Portfolio Project
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor


# ---------------------------------------------------------------------------
# Physical Constants (SI Units)
# ---------------------------------------------------------------------------
MU_EARTH: float = 3.986004418e14      # Earth gravitational parameter (m^3 / s^2)
R_EARTH: float = 6_371_000.0          # Mean Earth radius (m)
OMEGA_EARTH: float = 7.2921159e-5     # Sidereal rotation rate around +Z (rad / s)
G0: float = 9.80665                   # Standard gravity (m / s^2)

# Exponential Atmosphere & Vehicle Aerodynamics
RHO_0: float = 1.225                  # Sea-level air density (kg / m^3)
H_SCALE: float = 8_500.0              # Atmospheric scale height (m)
ATMOS_CUTOFF_M: float = 150_000.0     # Negligible drag above 150 km
CD_VEHICLE: float = 0.30              # Drag coefficient
REF_AREA_M2: float = 10.5             # Cross-sectional area (~3.66 m fairing)

# Fixed Two-Stage Vehicle Structural & Propulsion Constants
STAGE1_ISP_S: float = 295.0
STAGE2_ISP_S: float = 345.0
STAGE1_DRY_MASS_KG: float = 26_000.0
STAGE2_DRY_MASS_KG: float = 5_500.0
INTERSTAGE_COAST_S: float = 4.0
PITCH_START_ALT_M: float = 1_500.0    # Vertical tower-clearance climb

# LEO Success Thresholds
LEO_MIN_ALT_KM: float = 180.0
LEO_MIN_SPEED_MS: float = 7_750.0

FEATURE_NAMES: List[str] = [
    "launch_lat_deg",
    "launch_az_deg",
    "stage1_thrust_kn",
    "stage1_burn_s",
    "stage2_thrust_kn",
    "stage2_burn_s",
    "payload_mass_kg",
    "pitch_end_alt_km",
]

TARGET_NAMES: List[str] = [
    "final_alt_km",
    "final_speed_ms",
    "inclination_deg",
    "max_q_kpa",
]

PARAMETER_BOUNDS: Dict[str, Tuple[float, float]] = {
    "launch_lat_deg": (-35.0, 55.0),
    "launch_az_deg": (35.0, 145.0),
    "stage1_thrust_kn": (6_500.0, 9_000.0),
    "stage1_burn_s": (130.0, 175.0),
    "stage2_thrust_kn": (750.0, 1_150.0),
    "stage2_burn_s": (270.0, 380.0),
    "payload_mass_kg": (5_000.0, 19_000.0),
    "pitch_end_alt_km": (95.0, 175.0),
}


@dataclass(frozen=True)
class LaunchConfig:
    """Input configuration for a 3D two-stage orbital launch."""

    launch_lat_deg: float = 28.5
    launch_az_deg: float = 90.0
    stage1_thrust_kn: float = 7_600.0
    stage1_burn_s: float = 155.0
    stage2_thrust_kn: float = 880.0
    stage2_burn_s: float = 315.0
    payload_mass_kg: float = 13_500.0
    pitch_end_alt_km: float = 140.0

    def to_feature_vector(self) -> np.ndarray:
        """Convert configuration to ordered NumPy feature vector for ML."""
        return np.array(
            [
                self.launch_lat_deg,
                self.launch_az_deg,
                self.stage1_thrust_kn,
                self.stage1_burn_s,
                self.stage2_thrust_kn,
                self.stage2_burn_s,
                self.payload_mass_kg,
                self.pitch_end_alt_km,
            ],
            dtype=np.float64,
        )


@dataclass
class SimulationResult:
    """Full 3D ECI trajectory and orbital summary metrics."""

    leo_success: bool
    crashed: bool
    final_alt_km: float
    final_speed_ms: float
    radial_vel_ms: float
    inclination_deg: float
    max_q_kpa: float
    eccentricity: float
    perigee_km: float
    apogee_km: float
    solve_time_ms: float
    # Time-series arrays for 3D visualization and telemetry charts
    t_s: np.ndarray
    x_km: np.ndarray
    y_km: np.ndarray
    z_km: np.ndarray
    alt_km: np.ndarray
    speed_ms: np.ndarray
    q_kpa: np.ndarray
    stage_id: np.ndarray  # 1 = Stage 1, 0 = Coast, 2 = Stage 2


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return np.zeros_like(vec)
    return vec / norm


def _atmospheric_density(alt_m: float) -> float:
    if alt_m < 0.0:
        return RHO_0
    if alt_m >= ATMOS_CUTOFF_M:
        return 0.0
    return RHO_0 * math.exp(-alt_m / H_SCALE)


def _pitch_angle_rad(alt_m: float, pitch_end_alt_m: float, vr_ms: float, stage: int) -> float:
    """
    Altitude-scheduled gravity-turn pitch angle (radians from local vertical).

    - theta = 0: pure vertical climb (tower clearance below 1.5 km)
    - 0 < theta < pi/2: smooth power-law pitchover toward local horizontal
    - stage 2 above pitch_end_alt_m: gentle proportional vertical-velocity
      damping so the rocket levels out cleanly at insertion.
    """
    if alt_m <= PITCH_START_ALT_M:
        return 0.0

    span = max(10_000.0, pitch_end_alt_m - PITCH_START_ALT_M)
    if alt_m < pitch_end_alt_m:
        frac = (alt_m - PITCH_START_ALT_M) / span
        return (math.pi / 2.0) * (frac ** 0.72)

    theta = math.pi / 2.0
    if stage == 2:
        # If rising (vr > 0), pitch slightly below horizon (theta > pi/2) to arrest climb.
        # If sinking (vr < 0), pitch slightly above horizon (theta < pi/2) to arrest sink.
        correction = max(-0.22, min(0.22, vr_ms / 450.0))
        theta += correction
    return theta


def _launch_frame_eci(lat_deg: float, az_deg: float, lon_deg: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute initial ECI position r0, Earth-rotation velocity v0,
    azimuth unit vector u_az0, and target orbital plane normal h_hat.
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    az = math.radians(az_deg)

    r0 = R_EARTH * np.array(
        [
            math.cos(lat) * math.cos(lon),
            math.cos(lat) * math.sin(lon),
            math.sin(lat),
        ],
        dtype=np.float64,
    )
    omega_vec = np.array([0.0, 0.0, OMEGA_EARTH], dtype=np.float64)
    v0 = np.cross(omega_vec, r0)

    u_r0 = _unit(r0)
    east0 = np.array([-math.sin(lon), math.cos(lon), 0.0], dtype=np.float64)
    north0 = np.array(
        [
            -math.sin(lat) * math.cos(lon),
            -math.sin(lat) * math.sin(lon),
            math.cos(lat),
        ],
        dtype=np.float64,
    )
    u_az0 = math.cos(az) * north0 + math.sin(az) * east0
    h_hat = _unit(np.cross(u_r0, u_az0))
    return r0, v0, u_az0, h_hat


def simulate_launch_3d(cfg: LaunchConfig, max_step_s: float = 2.0) -> SimulationResult:
    """
    Integrate the 7-state 3D ECI equations of motion [x, y, z, vx, vy, vz, m]
    across Stage 1 burn, interstage coast, and Stage 2 burn using solve_ivp.
    """
    t_wall_start = time.perf_counter()

    r0, v0, u_az0, h_hat = _launch_frame_eci(cfg.launch_lat_deg, cfg.launch_az_deg)
    omega_vec = np.array([0.0, 0.0, OMEGA_EARTH], dtype=np.float64)

    thrust1_n = cfg.stage1_thrust_kn * 1_000.0
    thrust2_n = cfg.stage2_thrust_kn * 1_000.0
    mdot1 = thrust1_n / (STAGE1_ISP_S * G0)
    mdot2 = thrust2_n / (STAGE2_ISP_S * G0)

    prop1_kg = mdot1 * cfg.stage1_burn_s
    prop2_kg = mdot2 * cfg.stage2_burn_s
    m0 = STAGE1_DRY_MASS_KG + prop1_kg + STAGE2_DRY_MASS_KG + prop2_kg + cfg.payload_mass_kg
    pitch_end_m = cfg.pitch_end_alt_km * 1_000.0

    def make_rhs(stage: int) -> Callable[[float, np.ndarray], np.ndarray]:
        if stage == 1:
            thrust_n, mdot = thrust1_n, mdot1
        elif stage == 2:
            thrust_n, mdot = thrust2_n, mdot2
        else:
            thrust_n, mdot = 0.0, 0.0

        def rhs(_t: float, y: np.ndarray) -> np.ndarray:
            r = y[0:3]
            v = y[3:6]
            m = max(1.0, float(y[6]))

            rmag = float(np.linalg.norm(r))
            alt_m = rmag - R_EARTH
            u_r = r / rmag

            u_t = np.cross(h_hat, u_r)
            ut_norm = float(np.linalg.norm(u_t))
            u_t = u_t / ut_norm if ut_norm > 1e-12 else u_az0

            vr_ms = float(np.dot(v, u_r))
            theta = _pitch_angle_rad(alt_m, pitch_end_m, vr_ms, stage)
            u_thrust = math.cos(theta) * u_r + math.sin(theta) * u_t

            # Central two-body gravity
            a_grav = (-MU_EARTH / (rmag ** 3)) * r

            # Atmospheric drag relative to co-rotating air mass
            rho = _atmospheric_density(alt_m)
            if rho > 0.0:
                v_atm = np.cross(omega_vec, r)
                v_rel = v - v_atm
                v_rel_mag = float(np.linalg.norm(v_rel))
                a_drag = (-0.5 * rho * CD_VEHICLE * REF_AREA_M2 * v_rel_mag / m) * v_rel
            else:
                a_drag = np.zeros(3, dtype=np.float64)

            a_thrust = (thrust_n / m) * u_thrust
            a_total = a_grav + a_drag + a_thrust

            return np.array(
                [v[0], v[1], v[2], a_total[0], a_total[1], a_total[2], -mdot],
                dtype=np.float64,
            )

        return rhs

    def crash_event(_t: float, y: np.ndarray) -> float:
        return float(np.linalg.norm(y[0:3]) - R_EARTH + 50.0)

    crash_event.terminal = True
    crash_event.direction = -1

    y_init = np.concatenate([r0, v0, np.array([m0], dtype=np.float64)])

    t_s1_end = cfg.stage1_burn_s
    t_coast_end = t_s1_end + INTERSTAGE_COAST_S
    t_s2_end = t_coast_end + cfg.stage2_burn_s

    segments = [
        (0.0, t_s1_end, 1, False),
        (t_s1_end, t_coast_end, 0, True),
        (t_coast_end, t_s2_end, 2, False),
    ]

    all_t: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_stage: List[np.ndarray] = []
    crashed = False
    y_curr = y_init

    for t_start, t_end, stg, drop_stage1 in segments:
        if drop_stage1:
            y_curr = y_curr.copy()
            y_curr[6] = max(1.0, y_curr[6] - STAGE1_DRY_MASS_KG)

        sol = solve_ivp(
            make_rhs(stg),
            (t_start, t_end),
            y_curr,
            method="RK45",
            max_step=max_step_s,
            rtol=1e-7,
            atol=1e-9,
            events=[crash_event],
        )

        # Avoid duplicate boundary timestamp between contiguous segments
        slice_idx = 0 if not all_t else 1
        all_t.append(sol.t[slice_idx:])
        all_y.append(sol.y[:, slice_idx:])
        all_stage.append(np.full(sol.t.shape[0] - slice_idx, stg, dtype=np.int32))

        y_curr = sol.y[:, -1]
        if len(sol.t_events[0]) > 0:
            crashed = True
            break

    t_arr = np.concatenate(all_t)
    y_arr = np.concatenate(all_y, axis=1)
    stage_arr = np.concatenate(all_stage)

    pos_m = y_arr[0:3, :]
    vel_ms = y_arr[3:6, :]
    r_mags = np.linalg.norm(pos_m, axis=0)
    alt_m_arr = r_mags - R_EARTH
    speed_arr = np.linalg.norm(vel_ms, axis=0)

    # Dynamic pressure q = 0.5 * rho * |v_rel|^2 along trajectory
    q_pa_arr = np.zeros_like(t_arr)
    for i in range(t_arr.shape[0]):
        rho = _atmospheric_density(float(alt_m_arr[i]))
        if rho > 0.0:
            v_atm = np.cross(omega_vec, pos_m[:, i])
            v_rel_mag = float(np.linalg.norm(vel_ms[:, i] - v_atm))
            q_pa_arr[i] = 0.5 * rho * (v_rel_mag ** 2)

    r_final = pos_m[:, -1]
    v_final = vel_ms[:, -1]
    rmag_final = float(r_mags[-1])
    final_alt_km = float(alt_m_arr[-1] / 1_000.0)
    final_speed_ms = float(speed_arr[-1])
    radial_vel_ms = float(np.dot(v_final, r_final / rmag_final))
    max_q_kpa = float(np.max(q_pa_arr) / 1_000.0)

    # Specific angular momentum & Keplerian orbital elements at burnout
    h_vec = np.cross(r_final, v_final)
    h_mag = float(np.linalg.norm(h_vec))
    cos_inc = max(-1.0, min(1.0, float(h_vec[2]) / max(1e-9, h_mag)))
    inclination_deg = math.degrees(math.acos(cos_inc))

    specific_energy = 0.5 * (final_speed_ms ** 2) - (MU_EARTH / rmag_final)
    e_vec = (np.cross(v_final, h_vec) / MU_EARTH) - (r_final / rmag_final)
    eccentricity = float(np.linalg.norm(e_vec))

    if specific_energy < 0.0:
        semi_major_m = -MU_EARTH / (2.0 * specific_energy)
        perigee_km = (semi_major_m * (1.0 - eccentricity) - R_EARTH) / 1_000.0
        apogee_km = (semi_major_m * (1.0 + eccentricity) - R_EARTH) / 1_000.0
    else:
        perigee_km = final_alt_km
        apogee_km = float("inf")

    leo_success = (
        (not crashed)
        and (final_alt_km >= LEO_MIN_ALT_KM)
        and (final_speed_ms >= LEO_MIN_SPEED_MS)
    )

    solve_time_ms = (time.perf_counter() - t_wall_start) * 1_000.0

    return SimulationResult(
        leo_success=leo_success,
        crashed=crashed,
        final_alt_km=final_alt_km,
        final_speed_ms=final_speed_ms,
        radial_vel_ms=radial_vel_ms,
        inclination_deg=inclination_deg,
        max_q_kpa=max_q_kpa,
        eccentricity=eccentricity,
        perigee_km=perigee_km,
        apogee_km=apogee_km,
        solve_time_ms=solve_time_ms,
        t_s=t_arr,
        x_km=pos_m[0, :] / 1_000.0,
        y_km=pos_m[1, :] / 1_000.0,
        z_km=pos_m[2, :] / 1_000.0,
        alt_km=alt_m_arr / 1_000.0,
        speed_ms=speed_arr,
        q_kpa=q_pa_arr / 1_000.0,
        stage_id=stage_arr,
    )


# ---------------------------------------------------------------------------
# Dataset Generation & 2-Stage XGBoost Surrogate
# ---------------------------------------------------------------------------
def generate_training_dataset(
    n_samples: int = 800,
    seed: int = 42,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate labeled 3D launch trajectories across the parameter envelope.

    Returns:
        X: shape (n_samples, 8) input launch features
        y_cls: shape (n_samples,) binary LEO success indicator (0 or 1)
        y_reg: shape (n_samples, 4) [final_alt_km, final_speed_ms, inclination_deg, max_q_kpa]
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, len(FEATURE_NAMES)), dtype=np.float64)
    y_cls = np.zeros(n_samples, dtype=np.int32)
    y_reg = np.zeros((n_samples, len(TARGET_NAMES)), dtype=np.float64)

    for idx, name in enumerate(FEATURE_NAMES):
        lo, hi = PARAMETER_BOUNDS[name]
        # Stratified jittered grid per dimension reduces sample clumping
        bins = (np.arange(n_samples) + rng.random(n_samples)) / n_samples
        rng.shuffle(bins)
        X[:, idx] = lo + bins * (hi - lo)

    for i in range(n_samples):
        cfg = LaunchConfig(*[float(v) for v in X[i]])
        res = simulate_launch_3d(cfg, max_step_s=3.0)
        y_cls[i] = 1 if res.leo_success else 0
        y_reg[i, 0] = res.final_alt_km
        y_reg[i, 1] = res.final_speed_ms
        y_reg[i, 2] = res.inclination_deg
        y_reg[i, 3] = res.max_q_kpa

        if progress_callback is not None and ((i + 1) % 25 == 0 or i + 1 == n_samples):
            progress_callback(i + 1, n_samples)

    return X, y_cls, y_reg


class TwoStageOrbitalSurrogate:
    """
    Two-stage AI surrogate model for instant orbital trajectory prediction:
      Stage 1: XGBClassifier predicts P(LEO success)
      Stage 2: Dedicated XGBRegressors trained ONLY on successful LEO orbits
               predict final_alt_km, final_speed_ms, inclination_deg, and max_q_kpa.
    """

    def __init__(self) -> None:
        self.classifier = XGBClassifier(
            n_estimators=220,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=42,
        )
        self.regressors: Dict[str, XGBRegressor] = {
            target: XGBRegressor(
                n_estimators=260,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="reg:squarederror",
                random_state=42,
            )
            for target in TARGET_NAMES
        }
        self.metrics: Dict[str, Any] = {}
        self.is_fitted: bool = False

    def fit(
        self,
        X: np.ndarray,
        y_cls: np.ndarray,
        y_reg: np.ndarray,
        test_size: float = 0.2,
    ) -> Dict[str, Any]:
        """Train classifier on all launches and regressors strictly on successful LEO orbits."""
        X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
            X,
            y_cls,
            y_reg,
            test_size=test_size,
            random_state=42,
            stratify=y_cls,
        )

        # Stage 1: Binary LEO Success Classifier
        self.classifier.fit(X_train, y_cls_train)
        cls_probs = self.classifier.predict_proba(X_test)[:, 1]
        cls_preds = (cls_probs >= 0.5).astype(np.int32)

        metrics: Dict[str, Any] = {
            "n_total": int(len(X)),
            "n_leo_success": int(np.sum(y_cls)),
            "leo_success_rate": float(np.mean(y_cls)),
            "classifier": {
                "accuracy": float(accuracy_score(y_cls_test, cls_preds)),
                "f1": float(f1_score(y_cls_test, cls_preds)),
                "roc_auc": float(roc_auc_score(y_cls_test, cls_probs)),
            },
            "regressors": {},
            "feature_importances": {
                name: float(imp)
                for name, imp in zip(FEATURE_NAMES, self.classifier.feature_importances_)
            },
        }

        # Stage 2: Regressors trained ONLY on successful LEO trajectories
        train_ok = y_cls_train == 1
        test_ok = y_cls_test == 1
        X_reg_train = X_train[train_ok]
        X_reg_test = X_test[test_ok]

        for col_idx, target in enumerate(TARGET_NAMES):
            y_tr = y_reg_train[train_ok, col_idx]
            y_te = y_reg_test[test_ok, col_idx]

            reg = self.regressors[target]
            reg.fit(X_reg_train, y_tr)
            preds = reg.predict(X_reg_test)

            metrics["regressors"][target] = {
                "mae": float(mean_absolute_error(y_te, preds)),
                "r2": float(r2_score(y_te, preds)),
                "n_train_success": int(len(y_tr)),
                "n_test_success": int(len(y_te)),
            }

        self.metrics = metrics
        self.is_fitted = True
        return metrics

    def predict(self, cfg: LaunchConfig) -> Dict[str, Any]:
        """Run instant 2-stage surrogate inference for a single LaunchConfig."""
        if not self.is_fitted:
            raise RuntimeError("TwoStageOrbitalSurrogate must be fitted or loaded before predicting.")

        t0 = time.perf_counter()
        x_row = cfg.to_feature_vector().reshape(1, -1)
        prob_leo = float(self.classifier.predict_proba(x_row)[0, 1])
        pred_success = prob_leo >= 0.5

        reg_preds = {
            target: float(self.regressors[target].predict(x_row)[0])
            for target in TARGET_NAMES
        }
        infer_time_ms = (time.perf_counter() - t0) * 1_000.0

        return {
            "leo_success_pred": pred_success,
            "leo_probability": prob_leo,
            "final_alt_km": reg_preds["final_alt_km"],
            "final_speed_ms": reg_preds["final_speed_ms"],
            "inclination_deg": reg_preds["inclination_deg"],
            "max_q_kpa": reg_preds["max_q_kpa"],
            "infer_time_ms": infer_time_ms,
        }

    def save(self, model_dir: str | Path = "artifacts/surrogate_3d") -> Path:
        """Persist trained models and validation metrics to disk."""
        out_dir = Path(model_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.classifier, out_dir / "classifier.joblib")
        for target, reg in self.regressors.items():
            joblib.dump(reg, out_dir / f"reg_{target}.joblib")
        with (out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
            json.dump(self.metrics, fh, indent=2)
        return out_dir

    @classmethod
    def load(cls, model_dir: str | Path = "artifacts/surrogate_3d") -> "TwoStageOrbitalSurrogate":
        """Load pre-trained surrogate models and metrics from disk."""
        in_dir = Path(model_dir)
        instance = cls()
        instance.classifier = joblib.load(in_dir / "classifier.joblib")
        for target in TARGET_NAMES:
            instance.regressors[target] = joblib.load(in_dir / f"reg_{target}.joblib")
        with (in_dir / "metrics.json").open("r", encoding="utf-8") as fh:
            instance.metrics = json.load(fh)
        instance.is_fitted = True
        return instance


def load_or_train_surrogate(
    model_dir: str | Path = "artifacts/surrogate_3d",
    n_samples: int = 800,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> TwoStageOrbitalSurrogate:
    """Return cached surrogate if present on disk; otherwise generate data and train."""
    target_dir = Path(model_dir)
    required = [target_dir / "classifier.joblib", target_dir / "metrics.json"] + [
        target_dir / f"reg_{t}.joblib" for t in TARGET_NAMES
    ]
    if all(p.exists() for p in required):
        return TwoStageOrbitalSurrogate.load(target_dir)

    X, y_cls, y_reg = generate_training_dataset(
        n_samples=n_samples,
        seed=42,
        progress_callback=progress_callback,
    )
    surrogate = TwoStageOrbitalSurrogate()
    surrogate.fit(X, y_cls, y_reg)
    surrogate.save(target_dir)
    return surrogate


if __name__ == "__main__":
    print("Running nominal 3D ECI physics verification...")
    nominal = LaunchConfig()
    sim_res = simulate_launch_3d(nominal)
    print(
        f"Nominal Launch -> LEO={sim_res.leo_success} | "
        f"Alt={sim_res.final_alt_km:.2f} km | Speed={sim_res.final_speed_ms:.1f} m/s | "
        f"Inc={sim_res.inclination_deg:.2f} deg | Max-Q={sim_res.max_q_kpa:.2f} kPa | "
        f"Solve={sim_res.solve_time_ms:.1f} ms"
    )

    print("\nGenerating 800-launch dataset and training 2-stage XGBoost surrogate...")
    model = load_or_train_surrogate(
        n_samples=800,
        progress_callback=lambda done, total: print(f"  Simulated {done}/{total} trajectories"),
    )
    print("Surrogate validation metrics:")
    print(json.dumps(model.metrics, indent=2))

    pred = model.predict(nominal)
    print("\nNominal Comparison (AI Surrogate vs 3D ODE):")
    print(f"  LEO Success : AI={pred['leo_success_pred']} (p={pred['leo_probability']:.3f}) vs ODE={sim_res.leo_success}")
    print(f"  Altitude    : AI={pred['final_alt_km']:.2f} km vs ODE={sim_res.final_alt_km:.2f} km")
    print(f"  Speed       : AI={pred['final_speed_ms']:.1f} m/s vs ODE={sim_res.final_speed_ms:.1f} m/s")
    print(f"  Inclination : AI={pred['inclination_deg']:.2f} deg vs ODE={sim_res.inclination_deg:.2f} deg")
    print(f"  Max-Q       : AI={pred['max_q_kpa']:.2f} kPa vs ODE={sim_res.max_q_kpa:.2f} kPa")
    print(f"  Runtime     : AI={pred['infer_time_ms']:.2f} ms vs ODE={sim_res.solve_time_ms:.1f} ms")
