# iot-anomaly-detector
# ⚡ IoT Microcontroller & Motor Anomaly Detector

**[🚀 Click Here to Try the Live Telemetry Simulator](https://iot-anomaly-detector-jfktdmgvgaqkdaskd4pvyz.streamlit.app)**

An interactive hardware telemetry simulator and unsupervised machine learning dashboard that detects and diagnoses physical failure modes in embedded systems (ESP32) and industrial motors in real time. Built with **Python**, **Streamlit**, **Scikit-Learn**, **NumPy**, and **Plotly**.

---

## 🎯 The Engineering Problem
In embedded systems and industrial IoT, hardware failures rarely happen without warning—voltage regulators develop ripple before shorting out, bearings vibrate before seizing, and MOSFETs exhibit thermal drift before burning out. Simple static thresholds often miss subtle multivariate anomalies across multiple sensors.

## 💡 How It Works
1. **Waveform & Telemetry Generation (`NumPy`):** Simulates 3-channel hardware sensor streams sampled at 10Hz—**MOSFET/Motor Temperature (°C)**, **ESP32 3.3V Supply Rail (V)**, and **Accelerometer Vibration Frequency (Hz)**—complete with realistic thermal and Gaussian sensor noise.
2. **Interactive Fault Injector:** Users can inject real-time physical faults via sidebar sliders—including **Thermal Runaway**, **AC Supply Voltage Ripple/Droop**, and **Mechanical Bearing Wear**—to test how the AI responds to single or combined hardware failures.
3. **Unsupervised Anomaly Detection (`IsolationForest`):** A `scikit-learn` Isolation Forest model trained strictly on healthy baseline telemetry monitors all three standardized channels simultaneously, flagging anomalous packets without requiring labeled failure data.
4. **Automated Root-Cause Triage (`Z-Score`):** When a fault cluster is detected, the system computes channel-wise standard deviation shifts to automatically diagnose the primary physical root cause and exports a timestamped fault log to CSV.

## 🛠️ Tech Stack
* **Machine Learning:** Scikit-Learn (`IsolationForest`, `StandardScaler`)
* **Signal Simulation & Math:** NumPy, Pandas
* **Oscilloscope Visualization & UI:** Streamlit, Plotly Subplots
