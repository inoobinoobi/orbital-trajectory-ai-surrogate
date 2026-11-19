"""
3D Orbital Trajectory Predictor & AI Surrogate — Streamlit Dashboard
====================================================================
Interactive mission console comparing instant 2-stage XGBoost surrogate
predictions against the full 3D Earth-Centered Inertial (ECI) ODE solver.

Run locally with:
    streamlit run app_3d.py
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from orbit_3d import (
    FEATURE_NAMES,
    LEO_MIN_ALT_KM,
    LEO_MIN_SPEED_MS,
    MU_EARTH,
    PARAMETER_BOUNDS,
    R_EARTH,
    LaunchConfig,
    SimulationResult,
    TwoStageOrbitalSurrogate,
    generate_training_dataset,
    load_or_train_surrogate,
    simulate_launch_3d,
)

R_EARTH_KM: float = R_EARTH / 1_000.0

PRESETS: Dict[str, Dict[str, float]] = {
    "Cape Canaveral Nominal (28.5° N, East)": {
        "launch_lat_deg": 28.5,
        "launch_az_deg": 90.0,
        "stage1_thrust_kn": 7_600.0,
        "stage1_burn_s": 155.0,
        "stage2_thrust_kn": 880.0,
        "stage2_burn_s": 315.0,
        "payload_mass_kg": 13_500.0,
        "pitch_end_alt_km": 140.0,
    },
    "Munich High-Latitude (48.1° N, East)": {
        "launch_lat_deg": 48.1,
        "launch_az_deg": 90.0,
        "stage1_thrust_kn": 7_800.0,
        "stage1_burn_s": 158.0,
        "stage2_thrust_kn": 900.0,
        "stage2_burn_s": 325.0,
        "payload_mass_kg": 12_500.0,
        "pitch_end_alt_km": 140.0,
    },
    "High-Inclination NE Ascent (28.5° N, Az 45°)": {
        "launch_lat_deg": 28.5,
        "launch_az_deg": 45.0,
        "stage1_thrust_kn": 7_900.0,
        "stage1_burn_s": 158.0,
        "stage2_thrust_kn": 920.0,
        "stage2_burn_s": 330.0,
        "payload_mass_kg": 12_000.0,
        "pitch_end_alt_km": 142.0,
    },
    "Overweight Failure Case (18,500 kg)": {
        "launch_lat_deg": 28.5,
        "launch_az_deg": 90.0,
        "stage1_thrust_kn": 7_400.0,
        "stage1_burn_s": 150.0,
        "stage2_thrust_kn": 850.0,
        "stage2_burn_s": 300.0,
        "payload_mass_kg": 18_500.0,
        "pitch_end_alt_km": 140.0,
    },
}


@st.cache_resource(show_spinner=False)
def get_surrogate(n_samples: int = 800) -> TwoStageOrbitalSurrogate:
    """Load cached 2-stage XGBoost surrogate or train it once."""
    return load_or_train_surrogate(n_samples=n_samples)


def build_earth_surface(n_lat: int = 40, n_lon: int = 80) -> go.Surface:
    """Build a 3D Earth sphere using go.Surface in ECI kilometers."""
    u = np.linspace(0.0, 2.0 * np.pi, n_lon)
    v = np.linspace(0.0, np.pi, n_lat)
    x = R_EARTH_KM * np.outer(np.cos(u), np.sin(v)).T
    y = R_EARTH_KM * np.outer(np.sin(u), np.sin(v)).T
    z = R_EARTH_KM * np.outer(np.ones_like(u), np.cos(v)).T

    # Subtle latitude/longitude shading texture for clear 3D orientation
    lon_grid, lat_grid = np.meshgrid(u, v)
    surface_color = 0.35 + 0.15 * np.cos(2.0 * lat_grid) + 0.08 * np.sin(6.0 * lon_grid) * np.sin(3.0 * lat_grid)

    return go.Surface(
        x=x,
        y=y,
        z=z,
        surfacecolor=surface_color,
        colorscale=[
            [0.0, "#0b1d3a"],
            [0.45, "#12345d"],
            [0.75, "#1c4e80"],
            [1.0, "#2f6fa3"],
        ],
        showscale=False,
        opacity=0.95,
        name="Earth (6,371 km)",
        hoverinfo="skip",
        lighting=dict(ambient=0.65, diffuse=0.75, specular=0.18, roughness=0.6),
    )


def compute_keplerian_ring(sim: SimulationResult, n_pts: int = 180) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagate an unpowered Keplerian ellipse from burnout state for visualization."""
    r_vec = np.array([sim.x_km[-1], sim.y_km[-1], sim.z_km[-1]], dtype=np.float64) * 1_000.0
    # Reconstruct final velocity direction approximately using last two samples
    if len(sim.x_km) < 2:
        return np.array([]), np.array([]), np.array([])

    dt = max(1e-3, float(sim.t_s[-1] - sim.t_s[-2]))
    v_approx = np.array(
        [
            (sim.x_km[-1] - sim.x_km[-2]) * 1_000.0 / dt,
            (sim.y_km[-1] - sim.y_km[-2]) * 1_000.0 / dt,
            (sim.z_km[-1] - sim.z_km[-2]) * 1_000.0 / dt,
        ],
        dtype=np.float64,
    )
    v_mag = float(np.linalg.norm(v_approx))
    if v_mag > 1.0:
        v_vec = v_approx * (sim.final_speed_ms / v_mag)
    else:
        return np.array([]), np.array([]), np.array([])

    r_mag = float(np.linalg.norm(r_vec))
    eps = 0.5 * (sim.final_speed_ms ** 2) - MU_EARTH / r_mag
    if eps >= 0.0 or sim.eccentricity >= 0.95:
        return np.array([]), np.array([]), np.array([])

    h_vec = np.cross(r_vec, v_vec)
    h_hat = h_vec / np.linalg.norm(h_vec)
    e_vec = (np.cross(v_vec, h_vec) / MU_EARTH) - (r_vec / r_mag)
    e = float(np.linalg.norm(e_vec))
    p = float(np.dot(h_vec, h_vec)) / MU_EARTH

    p_hat = e_vec / e if e > 1e-6 else r_vec / r_mag
    q_hat = np.cross(h_hat, p_hat)

    nu = np.linspace(0.0, 2.0 * np.pi, n_pts)
    r_orb = p / (1.0 + e * np.cos(nu))
    pts = (np.outer(np.cos(nu) * r_orb, p_hat) + np.outer(np.sin(nu) * r_orb, q_hat)) / 1_000.0
    return pts[:, 0], pts[:, 1], pts[:, 2]


def build_3d_trajectory_figure(sim: SimulationResult) -> go.Figure:
    """Create the interactive 3D Plotly globe + rocket trajectory figure."""
    fig = go.Figure()
    fig.add_trace(build_earth_surface())

    # Equator reference circle
    ang = np.linspace(0.0, 2.0 * np.pi, 180)
    eq_r = R_EARTH_KM * 1.004
    fig.add_trace(
        go.Scatter3d(
            x=eq_r * np.cos(ang),
            y=eq_r * np.sin(ang),
            z=np.zeros_like(ang),
            mode="lines",
            line=dict(color="rgba(148, 163, 184, 0.35)", width=2, dash="dot"),
            name="Equatorial Plane",
            hoverinfo="skip",
        )
    )

    # ECI Coordinate Axes (X, Y, Z)
    axis_len = R_EARTH_KM * 1.38
    axes_spec = [
        ("ECI +X (Vernal Equinox)", [0, axis_len], [0, 0], [0, 0], "#ef4444"),
        ("ECI +Y", [0, 0], [0, axis_len], [0, 0], "#22c55e"),
        ("ECI +Z (North Pole)", [0, 0], [0, 0], [0, axis_len], "#38bdf8"),
    ]
    for label, ax, ay, az, col in axes_spec:
        fig.add_trace(
            go.Scatter3d(
                x=ax,
                y=ay,
                z=az,
                mode="lines",
                line=dict(color=col, width=3),
                name=label,
            )
        )

    # Stage 1 Ascent Arc
    s1_mask = sim.stage_id == 1
    if np.any(s1_mask):
        fig.add_trace(
            go.Scatter3d(
                x=sim.x_km[s1_mask],
                y=sim.y_km[s1_mask],
                z=sim.z_km[s1_mask],
                mode="lines",
                line=dict(color="#f59e0b", width=7),
                name="Stage 1 Ascent",
                customdata=np.column_stack([sim.t_s[s1_mask], sim.alt_km[s1_mask], sim.speed_ms[s1_mask]]),
                hovertemplate="<b>Stage 1</b><br>t = %{customdata[0]:.1f} s<br>Alt = %{customdata[1]:.1f} km<br>Speed = %{customdata[2]:.0f} m/s<extra></extra>",
            )
        )

    # Stage 2 Orbital Insertion Arc (including coast connection)
    s2_mask = sim.stage_id != 1
    if np.any(s2_mask):
        fig.add_trace(
            go.Scatter3d(
                x=sim.x_km[s2_mask],
                y=sim.y_km[s2_mask],
                z=sim.z_km[s2_mask],
                mode="lines",
                line=dict(color="#06b6d4", width=6),
                name="Stage 2 Burn",
                customdata=np.column_stack([sim.t_s[s2_mask], sim.alt_km[s2_mask], sim.speed_ms[s2_mask]]),
                hovertemplate="<b>Stage 2</b><br>t = %{customdata[0]:.1f} s<br>Alt = %{customdata[1]:.1f} km<br>Speed = %{customdata[2]:.0f} m/s<extra></extra>",
            )
        )

    # Launch Pad & Burnout Markers
    fig.add_trace(
        go.Scatter3d(
            x=[sim.x_km[0]],
            y=[sim.y_km[0]],
            z=[sim.z_km[0]],
            mode="markers",
            marker=dict(size=6, color="#facc15"),
            name="Launch Pad",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[sim.x_km[-1]],
            y=[sim.y_km[-1]],
            z=[sim.z_km[-1]],
            mode="markers",
            marker=dict(size=7, color="#22c55e" if sim.leo_success else "#ef4444"),
            name="SECO / Burnout",
        )
    )

    # Projected Post-Burnout Orbit Ring (if bound orbit)
    ox, oy, oz = compute_keplerian_ring(sim)
    if ox.size > 0 and sim.leo_success:
        fig.add_trace(
            go.Scatter3d(
                x=ox,
                y=oy,
                z=oz,
                mode="lines",
                line=dict(color="rgba(34, 197, 94, 0.55)", width=3, dash="dash"),
                name="Projected Keplerian Orbit",
                hoverinfo="skip",
            )
        )

    bound = R_EARTH_KM * 1.45
    fig.update_layout(
        height=620,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="#07090e",
        plot_bgcolor="#07090e",
        font=dict(color="#e2e8f0"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.01,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(15, 23, 42, 0.75)",
        ),
        scene=dict(
            bgcolor="#07090e",
            aspectmode="cube",
            xaxis=dict(range=[-bound, bound], title="X (km)", gridcolor="#1e293b", zerolinecolor="#334155"),
            yaxis=dict(range=[-bound, bound], title="Y (km)", gridcolor="#1e293b", zerolinecolor="#334155"),
            zaxis=dict(range=[-bound, bound], title="Z (km)", gridcolor="#1e293b", zerolinecolor="#334155"),
            camera=dict(eye=dict(x=1.45, y=1.35, z=0.85)),
        ),
    )
    return fig


def build_telemetry_figure(sim: SimulationResult) -> go.Figure:
    """Create 2-panel telemetry plot: Altitude/Speed and Dynamic Pressure (Max-Q)."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Altitude & Inertial Speed vs. Time", "Dynamic Pressure q(t) & Max-Q"),
        specs=[[{"secondary_y": True}, {"secondary_y": False}]],
    )

    fig.add_trace(
        go.Scatter(x=sim.t_s, y=sim.alt_km, name="Altitude (km)", line=dict(color="#06b6d4", width=2.5)),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=sim.t_s, y=sim.speed_ms, name="Speed (m/s)", line=dict(color="#f59e0b", width=2.5)),
        row=1,
        col=1,
        secondary_y=True,
    )
    fig.add_hline(y=LEO_MIN_ALT_KM, line_dash="dot", line_color="#22c55e", row=1, col=1, secondary_y=False)

    fig.add_trace(
        go.Scatter(
            x=sim.t_s,
            y=sim.q_kpa,
            name="Dynamic Pressure (kPa)",
            fill="tozeroy",
            line=dict(color="#a855f7", width=2.5),
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        height=340,
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0b0f19",
        font=dict(color="#e2e8f0"),
        margin=dict(l=40, r=40, t=45, b=35),
        legend=dict(orientation="h", y=-0.2),
    )
    fig.update_xaxes(title_text="Mission Elapsed Time (s)", gridcolor="#1e293b")
    fig.update_yaxes(title_text="Altitude (km)", gridcolor="#1e293b", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Speed (m/s)", gridcolor="#1e293b", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Dynamic Pressure (kPa)", gridcolor="#1e293b", row=1, col=2)
    return fig


def main() -> None:
    st.set_page_config(
        page_title="3D Orbital Trajectory Predictor & AI Surrogate",
        page_icon="🛰️",
        layout="wide",
    )

    st.title("3D Orbital Trajectory Predictor & 2-Stage XGBoost Surrogate")
    st.caption(
        "Comparing instant 2-stage XGBoost inference (Classifier + LEO Regressors) "
        "against full 3D Earth-Centered Inertial (X, Y, Z) solve_ivp physics."
    )

    # Load or train surrogate on first launch
    with st.sidebar:
        st.header("Launch Mission Controls")
        preset_name = st.selectbox("Mission Preset", list(PRESETS.keys()), index=0)
        preset = PRESETS[preset_name]

        launch_lat_deg = st.slider(
            "Launch Latitude (deg)",
            min_value=float(PARAMETER_BOUNDS["launch_lat_deg"][0]),
            max_value=float(PARAMETER_BOUNDS["launch_lat_deg"][1]),
            value=float(preset["launch_lat_deg"]),
            step=0.5,
        )
        launch_az_deg = st.slider(
            "Launch Azimuth (deg, 90° = East)",
            min_value=float(PARAMETER_BOUNDS["launch_az_deg"][0]),
            max_value=float(PARAMETER_BOUNDS["launch_az_deg"][1]),
            value=float(preset["launch_az_deg"]),
            step=1.0,
        )
        payload_mass_kg = st.slider(
            "Payload Mass (kg)",
            min_value=float(PARAMETER_BOUNDS["payload_mass_kg"][0]),
            max_value=float(PARAMETER_BOUNDS["payload_mass_kg"][1]),
            value=float(preset["payload_mass_kg"]),
            step=250.0,
        )
        stage1_thrust_kn = st.slider(
            "Stage 1 Thrust (kN)",
            min_value=float(PARAMETER_BOUNDS["stage1_thrust_kn"][0]),
            max_value=float(PARAMETER_BOUNDS["stage1_thrust_kn"][1]),
            value=float(preset["stage1_thrust_kn"]),
            step=50.0,
        )
        stage1_burn_s = st.slider(
            "Stage 1 Burn Time (s)",
            min_value=float(PARAMETER_BOUNDS["stage1_burn_s"][0]),
            max_value=float(PARAMETER_BOUNDS["stage1_burn_s"][1]),
            value=float(preset["stage1_burn_s"]),
            step=1.0,
        )
        stage2_thrust_kn = st.slider(
            "Stage 2 Thrust (kN)",
            min_value=float(PARAMETER_BOUNDS["stage2_thrust_kn"][0]),
            max_value=float(PARAMETER_BOUNDS["stage2_thrust_kn"][1]),
            value=float(preset["stage2_thrust_kn"]),
            step=10.0,
        )
        stage2_burn_s = st.slider(
            "Stage 2 Burn Time (s)",
            min_value=float(PARAMETER_BOUNDS["stage2_burn_s"][0]),
            max_value=float(PARAMETER_BOUNDS["stage2_burn_s"][1]),
            value=float(preset["stage2_burn_s"]),
            step=2.0,
        )
        pitch_end_alt_km = st.slider(
            "Pitch Program End Altitude (km)",
            min_value=float(PARAMETER_BOUNDS["pitch_end_alt_km"][0]),
            max_value=float(PARAMETER_BOUNDS["pitch_end_alt_km"][1]),
            value=float(preset["pitch_end_alt_km"]),
            step=1.0,
        )

        st.divider()
        if st.button("Retrain 2-Stage XGBoost Surrogate (800 runs)"):
            progress = st.progress(0.0, text="Simulating 3D trajectories...")
            X, y_cls, y_reg = generate_training_dataset(
                n_samples=800,
                seed=42,
                progress_callback=lambda d, tot: progress.progress(d / tot, text=f"Simulated {d}/{tot} trajectories"),
            )
            fresh = TwoStageOrbitalSurrogate()
            fresh.fit(X, y_cls, y_reg)
            fresh.save()
            st.cache_resource.clear()
            st.success("Surrogate retrained and cached!")

    with st.spinner("Initializing 2-stage XGBoost surrogate (first run simulates 800 3D trajectories)..."):
        surrogate = get_surrogate()

    cfg = LaunchConfig(
        launch_lat_deg=launch_lat_deg,
        launch_az_deg=launch_az_deg,
        stage1_thrust_kn=stage1_thrust_kn,
        stage1_burn_s=stage1_burn_s,
        stage2_thrust_kn=stage2_thrust_kn,
        stage2_burn_s=stage2_burn_s,
        payload_mass_kg=payload_mass_kg,
        pitch_end_alt_km=pitch_end_alt_km,
    )

    ai_pred = surrogate.predict(cfg)
    ode_res = simulate_launch_3d(cfg)

    # Top Comparison Strip: AI Surrogate vs. 3D Physics ODE
    st.subheader("AI Surrogate vs. 3D Physics ODE Comparison")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        ai_label = "LEO PASS" if ai_pred["leo_success_pred"] else "SUBORBITAL"
        ode_label = "LEO PASS" if ode_res.leo_success else "FAIL"
        st.metric(
            "Orbit Outcome",
            f"ODE: {ode_label}",
            delta=f"AI: {ai_label} ({ai_pred['leo_probability']:.1%})",
        )

    with col2:
        alt_err = ai_pred["final_alt_km"] - ode_res.final_alt_km
        st.metric(
            "Burnout Altitude",
            f"{ode_res.final_alt_km:.1f} km",
            delta=f"AI: {ai_pred['final_alt_km']:.1f} km ({alt_err:+.1f})" if ai_pred["leo_success_pred"] else "Reg skipped (Suborbital)",
        )

    with col3:
        spd_err = ai_pred["final_speed_ms"] - ode_res.final_speed_ms
        st.metric(
            "Burnout Speed",
            f"{ode_res.final_speed_ms:,.0f} m/s",
            delta=f"AI: {ai_pred['final_speed_ms']:,.0f} m/s ({spd_err:+.0f})" if ai_pred["leo_success_pred"] else "Reg skipped (Suborbital)",
        )

    with col4:
        inc_err = ai_pred["inclination_deg"] - ode_res.inclination_deg
        st.metric(
            "Inclination",
            f"{ode_res.inclination_deg:.2f}°",
            delta=f"AI: {ai_pred['inclination_deg']:.2f}° ({inc_err:+.2f}°)",
        )

    with col5:
        q_err = ai_pred["max_q_kpa"] - ode_res.max_q_kpa
        speedup = ode_res.solve_time_ms / max(1e-3, ai_pred["infer_time_ms"])
        st.metric(
            "Max-Q / Speedup",
            f"{ode_res.max_q_kpa:.1f} kPa",
            delta=f"AI: {ai_pred['max_q_kpa']:.1f} kPa ({q_err:+.1f}) | {speedup:.0f}x faster",
        )

    # Interactive 3D Globe & Trajectory Arc
    st.plotly_chart(build_3d_trajectory_figure(ode_res), use_container_width=True)

    # Secondary Tabs: Telemetry Curves vs. Surrogate Model Diagnostics
    tab_tele, tab_ml = st.tabs(["Flight Telemetry & Max-Q", "2-Stage XGBoost Surrogate Diagnostics"])

    with tab_tele:
        st.plotly_chart(build_telemetry_figure(ode_res), use_container_width=True)
        st.info(
            f"LEO Criteria: Altitude ≥ {LEO_MIN_ALT_KM:.0f} km and Speed ≥ {LEO_MIN_SPEED_MS:,.0f} m/s | "
            f"Computed Perigee: {ode_res.perigee_km:.1f} km | Eccentricity: {ode_res.eccentricity:.4f} | "
            f"ODE Solve Time: {ode_res.solve_time_ms:.1f} ms vs. Surrogate: {ai_pred['infer_time_ms']:.2f} ms"
        )

    with tab_ml:
        m = surrogate.metrics
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("#### Stage 1: `XGBClassifier` (All Launches)")
            cls_m = m["classifier"]
            st.write(
                f"- **Test Accuracy:** `{cls_m['accuracy']:.3f}`\n"
                f"- **F1 Score:** `{cls_m['f1']:.3f}`\n"
                f"- **ROC-AUC:** `{cls_m['roc_auc']:.3f}`\n"
                f"- **Dataset LEO Success Rate:** `{m['leo_success_rate']:.1%}` "
                f"({m['n_leo_success']} / {m['n_total']} runs)"
            )
            imp = m["feature_importances"]
            imp_fig = go.Figure(
                go.Bar(
                    x=list(imp.values()),
                    y=list(imp.keys()),
                    orientation="h",
                    marker_color="#06b6d4",
                )
            )
            imp_fig.update_layout(
                height=280,
                title="Classifier Feature Importance",
                paper_bgcolor="#0b0f19",
                plot_bgcolor="#0b0f19",
                font=dict(color="#e2e8f0"),
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(imp_fig, use_container_width=True)

        with c_right:
            st.markdown("#### Stage 2: `XGBRegressor` (Trained Strictly on LEO Successes)")
            rows = []
            for target_name, t_metrics in m["regressors"].items():
                rows.append(
                    {
                        "Target": target_name,
                        "MAE": round(t_metrics["mae"], 3),
                        "R² Score": round(t_metrics["r2"], 4),
                        "Train Orbits": t_metrics["n_train_success"],
                        "Test Orbits": t_metrics["n_test_success"],
                    }
                )
            st.dataframe(rows, use_container_width=True, hide_index=True)
            st.caption(
                "Note: When Stage 1 predicts a suborbital failure, Stage 2 orbital regression "
                "targets are flagged because crashed/suborbital trajectories follow a different "
                "physical regime than successful LEO insertions."
            )


if __name__ == "__main__":
    main()
