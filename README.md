# 3D Orbital Trajectory Predictor & 2-Stage AI Surrogate

A dual-track aerospace simulation and machine learning surrogate project built as a **Technical University of Munich (TUM) Data Science Portfolio**.

The project models a two-stage orbital launch vehicle in **3D Earth-Centered Inertial (ECI) `(X, Y, Z)` coordinates** using `scipy.integrate.solve_ivp`, trains a **two-stage XGBoost hurdle surrogate** to predict Low Earth Orbit (LEO) insertion in under a millisecond, and compares AI predictions live against the full 3D physics integrator in both a **Streamlit + Plotly** app and an **interactive browser 3D web console**.

---

## System Architecture

```
                     8 Launch Parameters
  (lat, azimuth, stage 1/2 thrust & burn, payload, pitch-end alt)
                              │
         ┌────────────────────┴────────────────────┐
         ▼                                         ▼
┌─────────────────────────┐             ┌───────────────────────────┐
│ 3D ECI ODE Solver       │             │ 2-Stage XGBoost Surrogate │
│ (scipy RK45 · 3 stages) │             │                           │
│  1. Stage 1 Ascent      │             │ Stage 1: XGBClassifier    │
│  2. 4s Interstage Coast │             │   -> P(LEO Success)       │
│     (-26,000 kg drop)   │             │                           │
│  3. Stage 2 Insertion   │             │ Stage 2: 4x XGBRegressor  │
└────────────┬────────────┘             │   (trained strictly on    │
             │                          │    successful LEO runs)   │
             │                          └─────────────┬─────────────┘
             ▼                                        ▼
┌───────────────────────────────────────────────────────────────────┐
│  Live Comparison & 3D Visualization                               │
│  • Streamlit + Plotly (go.Surface Earth + go.Scatter3d arc)       │
│  • Interactive TypeScript 3D Web Simulator & Telemetry Console    │
└───────────────────────────────────────────────────────────────────┘
```

---

## Key Technical Highlights

### 1. 3D ECI Orbital Physics (`orbit_3d.py`)
- **State Vector:** 7-dimensional ECI state $\mathbf{y} = [x, y, z, v_x, v_y, v_z, m]^T$ integrating point-mass gravity, exponential co-rotating atmospheric drag ($\mathbf{v}_{\text{rel}} = \mathbf{v} - \boldsymbol{\omega}_E \times \mathbf{r}$), and steerable thrust.
- **Altitude-Scheduled Gravity Turn:** Vertical tower-clearance climb below `1.5 km`, smooth power-law pitchover $(\frac{h - h_0}{h_{\text{end}} - h_0})^{0.72}\cdot\frac{\pi}{2}$ up to `pitch_end_alt_km`, and proportional radial-velocity damping during Stage 2 so launches reliably reach LEO (`alt >= 180 km`, `speed >= 7,750 m/s`).
- **Piecewise-Smooth `solve_ivp`:** Splits integration across Stage 1, 4s Coast, and Stage 2 boundaries so the instantaneous `26,000 kg` stage separation does not degrade adaptive `RK45` step control.

### 2. Two-Stage XGBoost Hurdle Surrogate (`orbit_3d.py`)
- **Why Two Stages?** Crashed suborbital launches (`alt ~ 0 km`) and successful LEO orbits (`alt >= 180 km`, `speed >= 7,750 m/s`) occupy distinct physical regimes. Training a single regressor across both mixes regimes and degrades orbital accuracy.
- **Stage 1 (`XGBClassifier`):** Trained on 800 stratified jittered trajectories across the 8D launch parameter space to predict `P(LEO success)`.
- **Stage 2 (`XGBRegressor`):** Four separate regressors (`final_alt_km`, `final_speed_ms`, `inclination_deg`, `max_q_kpa`) trained **strictly on successful LEO orbits** (`y_cls == 1`) with an upfront stratified 80/20 split to prevent data leakage.

### 3. Interactive Visualization Dashboards
- **`app_3d.py`:** Streamlit + Plotly 3D Earth (`go.Surface`) and trajectory arc (`go.Scatter3d`) dashboard comparing instant XGBoost predictions against the 3D ODE solver, including projected Keplerian orbit rings, Max-Q curves, and classifier feature importances.
- **Interactive Web Simulator (`src/`):** Pure TypeScript port of the 3D ECI physics solver and 2-stage gradient-boosted surrogate running live in the browser with unit tests (`node:test`).

---

## Repository Contents

| File / Folder | Purpose |
| :--- | :--- |
| `orbit_3d.py` | 3D ECI `solve_ivp` trajectory simulator + 2-stage XGBoost surrogate |
| `app_3d.py` | Streamlit & Plotly 3D Earth + trajectory comparison dashboard |
| `requirements.txt` | Python dependencies for the ODE solver, XGBoost, and Streamlit app |
| `src/` | Interactive browser 3D simulator, telemetry charts, and TypeScript unit tests |
| `package.json` | Node/Vite frontend dependencies and build/test scripts |

---

## Quickstart

### Option A: Python Physics & Streamlit 3D Dashboard
Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Verify nominal 3D orbit & train/cache the 2-stage XGBoost surrogate
python orbit_3d.py

# 2. Launch the interactive Streamlit + Plotly 3D dashboard
streamlit run app_3d.py
```

### Option B: Interactive Web Simulator (React + TypeScript)
Requires Node.js 20+.

```bash
npm install
npm run dev      # Starts the interactive 3D web console on http://localhost:3000
npm run lint     # TypeScript type-check
npm run build    # Production bundle
npx tsx --test src/physics/orbit3d.test.ts   # Run physics & surrogate unit tests
```
