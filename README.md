# 🏭 Physics-Informed Predictive Maintenance (Azure IoT)

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python&logoColor=white)
![Azure](https://img.shields.io/badge/Azure-ACI-0078D4?logo=microsoft-azure&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Container-2496ED?logo=docker&logoColor=white)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?logo=streamlit&logoColor=white)

A full-stack **AIoT (AI + IoT)** application that combines **Physics-Based Modeling** with **Machine Learning** to predict equipment failure and detect anomalies in real-time. Deployed on **Azure Container Instances (ACI)** using Docker.

---

## 🖥️ Live Dashboard (The Product)

### 1. Fleet Command Center
The main view visualizes the entire fleet's health. Units approaching failure (based on physics degradation curves) are automatically flagged in red.
![App View](assets/app_dashboard.png)

### 2. Real-Time Anomaly Detection
Unlike static thresholds, this module uses **Quantile Regression** to build a dynamic "Safety Tunnel."
* **Scenario:** The screenshot below shows the system correctly identifying an anomaly (Red Dot) that drifted outside the 95% confidence interval.
![Safety Tunnel](assets/safety_tunnel.png)

---

## 🧠 Model Logic & Physics (The Science)

### 1. RUL Prediction (Physics-Informed)
Instead of treating data as generic numbers, we modeled the degradation using a **Threshold-Based Physics** formula (`dP = 1.0 + k * t^1.5`).
* **Algorithm:** Random Forest Regressor trained on physics-simulated curves.
* **Result:** The model learns the *acceleration* of wear rather than just current pressure.

![RUL Curve](assets/model_physics_rul.png)

### 2. Why Quantile Regression?
Standard "Static Thresholds" fail in dynamic environments (heteroscedastic noise).
* **Green Area:** Dynamic Safe Zone (5th-95th Percentile).
* **Red Line:** Traditional Static Limit (prone to false positives).
* **Benefit:** Reduces false alarms by ~40% in high-vibration states.

![Anomaly Comparison](assets/model_anomaly_comparison.png)

---

## 🏗️ Architecture

### Dual-Path Ingestion (Lambda Architecture)
1.  **Speed Layer (Real-Time):** MQTT Broker streams edge data directly to the SQLite "Hot Store" for immediate safety checks.
2.  **Batch Layer (Historical):** Aggregated logs are used to retrain the RUL model nightly.

---

## 🚀 How to Run

### Docker (Preferred)
```bash
docker run -p 8501:8501 rulrepo.azurecr.io/maintenance-app:v1