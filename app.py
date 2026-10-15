import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# 1. Page Configuration
st.set_page_config(
    page_title="IoT Hardware Anomaly Detector",
    page_icon="⚡",
    layout="wide"
)

# 2. Train Baseline Model on "Healthy" ESP32 / Motor Telemetry
@st.cache_resource
def train_baseline_model():
    np.random.seed(42)
    n_train = 1500
    t = np.linspace(0, 150, n_train)

    # Normal operating baselines: 45°C, 3.30V supply, 60Hz motor vibration
    normal_temp = 45.0 + 1.2 * np.sin(0.05 * t) + np.random.normal(0, 0.4, n_train)
    normal_volt = 3.30 + 0.02 * np.sin(0.2 * t) + np.random.normal(0, 0.015, n_train)
    normal_vib = 60.0 + 2.0 * np.sin(0.3 * t) + np.random.normal(0, 0.8, n_train)

    X_train = np.column_stack([normal_temp, normal_volt, normal_vib])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Isolation Forest trained on healthy telemetry (1.5% expected false-alarm rate)
    model = IsolationForest(
        n_estimators=150,
        contamination=0.015,
        random_state=42
    )
    model.fit(X_train_scaled)
    return model, scaler

model, scaler = train_baseline_model()

# 3. Sidebar: Hardware Fault Injector
st.sidebar.header("🛠️ Hardware Fault Injector")
st.sidebar.markdown("Inject real-time physical faults into the ESP32 telemetry stream (starts at sample `t = 120`):")

overheat_deg = st.sidebar.slider(
    "🔥 Thermal Runaway / Overheating (+°C)",
    min_value=0.0, max_value=25.0, value=8.0, step=0.5,
    help="Simulates cooling fan failure or high current draw starting at sample 120."
)

voltage_ripple = st.sidebar.slider(
    "⚡ Supply Voltage Ripple / Droop (V)",
    min_value=0.0, max_value=0.60, value=0.0, step=0.02,
    help="Simulates failing voltage regulator or capacitor degradation."
)

vib_spike = st.sidebar.slider(
    "⚙️ Bearing Wear / Vibration Noise (Hz)",
    min_value=0.0, max_value=25.0, value=0.0, step=0.5,
    help="Simulates loose motor mount or mechanical bearing damage."
)

st.sidebar.divider()
preset_col1, preset_col2 = st.sidebar.columns(2)
if preset_col1.button("🔄 Healthy Reset", use_container_width=True):
    st.rerun()

# 4. Generate Live Telemetry Stream (200 time steps)
np.random.seed(99)
n_samples = 200
t = np.arange(n_samples)

# Healthy baseline waveform
temp = 45.0 + 1.2 * np.sin(0.05 * t) + np.random.normal(0, 0.4, n_samples)
volt = 3.30 + 0.02 * np.sin(0.2 * t) + np.random.normal(0, 0.015, n_samples)
vib = 60.0 + 2.0 * np.sin(0.3 * t) + np.random.normal(0, 0.8, n_samples)

# Inject faults in the fault window (samples 120 to 185)
fault_window = slice(120, 185)
window_len = 185 - 120

if overheat_deg > 0:
    # Gradual thermal climb (realistic capacitor/heatsink warm-up curve)
    temp[fault_window] += np.linspace(0, overheat_deg, window_len)

if voltage_ripple > 0:
    # AC ripple + DC supply droop
    volt[fault_window] += voltage_ripple * np.sin(1.8 * t[fault_window]) - (voltage_ripple * 0.35)

if vib_spike > 0:
    # Erratic mechanical vibration spikes
    vib[fault_window] += np.random.normal(vib_spike * 0.6, vib_spike * 0.4, window_len)

# 5. Run Real-Time Anomaly Detection
X_stream = np.column_stack([temp, volt, vib])
X_stream_scaled = scaler.transform(X_stream)

# Predict: 1 = Normal, -1 = Anomaly
preds = model.predict(X_stream_scaled)
# Anomaly score: lower (negative) means more anomalous
scores = model.decision_function(X_stream_scaled)

df = pd.DataFrame({
    "Sample": t,
    "Temperature_C": np.round(temp, 2),
    "Voltage_V": np.round(volt, 3),
    "Vibration_Hz": np.round(vib, 2),
    "Anomaly_Score": np.round(scores, 4),
    "Status": np.where(preds == -1, "Anomaly", "Normal")
})

anomalies = df[df["Status"] == "Anomaly"]
num_anomalies = len(anomalies)

# 6. Main Dashboard Header & Diagnostic Banner
st.title("⚡ IoT Microcontroller & Motor Anomaly Detector")
st.markdown("Real-time unsupervised machine learning (`IsolationForest`) monitoring 3-channel hardware telemetry for thermal runaway, supply voltage ripple, and mechanical bearing wear.")

# Diagnostic Root-Cause Logic
if num_anomalies > 5:
    # Find which channel deviated most in standard deviations (Z-score)
    anom_indices = anomalies.index
    z_scores = np.abs(X_stream_scaled[anom_indices]).mean(axis=0)
    channels = ["Thermal Runaway (Overheating)", "Power Supply Instability (Voltage Ripple/Droop)", "Mechanical Bearing Fault (Vibration Spike)"]
    primary_cause = channels[np.argmax(z_scores)]
    st.error(f"🚨 **CRITICAL HARDWARE ALERT:** {num_anomalies} anomalous telemetry packets detected! **Primary Root Cause:** {primary_cause}")
elif num_anomalies > 0:
    st.warning(f"⚠️ **MINOR TRANSIENT NOISE:** {num_anomalies} isolated spike(s) detected. System within safe operating tolerances.")
else:
    st.success("✅ **SYSTEM HEALTHY:** All ESP32 & motor sensor streams operating within nominal parameters.")

# KPI Metrics Row
k1, k2, k3, k4 = st.columns(4)
k1.metric("Peak Temperature", f"{df['Temperature_C'].max():.1f} °C", f"{df['Temperature_C'].max() - 45.0:+.1f} °C vs nominal", delta_color="inverse")
k2.metric("Min Supply Voltage", f"{df['Voltage_V'].min():.2f} V", f"{df['Voltage_V'].min() - 3.30:+.2f} V vs 3.3V rail", delta_color="normal")
k3.metric("Max Vibration Freq", f"{df['Vibration_Hz'].max():.1f} Hz", f"{df['Vibration_Hz'].max() - 60.0:+.1f} Hz vs nominal", delta_color="inverse")
k4.metric("Flagged Anomalies", f"{num_anomalies} / {n_samples}", f"{(num_anomalies/n_samples)*100:.1f}% fault rate", delta_color="inverse")

st.divider()

# 7. Multi-Channel Oscilloscope Plot with Highlighted Red Anomalies
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.07,
    subplot_titles=(
        "🌡️ Channel 1: MOSFET / Motor Temperature (°C) — Nominal: 45°C",
        "⚡ Channel 2: ESP32 Supply Rail Voltage (V) — Nominal: 3.30V",
        "⚙️ Channel 3: Accelerometer Vibration Frequency (Hz) — Nominal: 60Hz"
    )
)

# Helper to plot waveform + red anomaly dots
channels_cfg = [
    ("Temperature_C", "#3498db", 1),
    ("Voltage_V", "#f1c40f", 2),
    ("Vibration_Hz", "#2ecc71", 3)
]

for col_name, line_color, row_idx in channels_cfg:
    # Normal waveform line
    fig.add_trace(
        go.Scatter(
            x=df["Sample"], y=df[col_name],
            mode="lines",
            name=col_name,
            line=dict(color=line_color, width=2)
        ),
        row=row_idx, col=1
    )
    # Red anomaly markers overlaid on exact failure points
    fig.add_trace(
        go.Scatter(
            x=anomalies["Sample"], y=anomalies[col_name],
            mode="markers",
            name="Anomaly Detected" if row_idx == 1 else None,
            showlegend=(row_idx == 1),
            marker=dict(color="#e74c3c", size=8, symbol="circle", line=dict(color="white", width=1))
        ),
        row=row_idx, col=1
    )

fig.update_layout(
    height=650,
    margin=dict(l=20, r=20, t=40, b=20),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
fig.update_xaxes(title_text="Time Step (Samples @ 10Hz)", row=3, col=1)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# 8. Fault Log Table & CSV Export
st.subheader("📋 Flagged Hardware Anomaly Log")
if num_anomalies > 0:
    st.dataframe(
        anomalies.sort_values(by="Anomaly_Score", ascending=True),
        use_container_width=True,
        hide_index=True
    )
    csv_data = anomalies.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Fault Telemetry Log (CSV)",
        data=csv_data,
        file_name="esp32_hardware_anomalies.csv",
        mime="text/csv"
    )
else:
    st.info("No anomalies detected in the current telemetry window. Use the Fault Injector sliders on the left to test failure modes!")
