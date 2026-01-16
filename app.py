import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import time
import threading
import paho.mqtt.client as mqtt
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Page Config
st.set_page_config(page_title="Azure Predictive Maintenance", layout="wide")

# --- 1. CACHED MODEL TRAINING (The Brains) ---
@st.cache_resource
def train_models():
    # A. RUL Model (Random Forest)
    # Simulate historical run-to-failure data
    history = []
    for i in range(50):
        max_life = int(np.random.uniform(1000, 2000))
        t = np.arange(max_life)
        # Physics: Exponential degradation
        wear = np.random.uniform(0.002, 0.005)
        dp = 1.0 + (wear * t**1.5) + np.random.normal(0, 0.1, max_life)
        df = pd.DataFrame({'Time_Cycle': t, 'Diff_Pressure': dp, 'RUL': max_life - t})
        # Features
        # FIX: Replaced .fillna(method='bfill') with .bfill()
        df['dP_Smooth'] = df['Diff_Pressure'].rolling(20).mean().bfill()
        df['Load_Index'] = df['Diff_Pressure'] * df['Time_Cycle']
        history.append(df)
    
    full_hist = pd.concat(history).dropna().sample(5000) # Sample for speed
    features = ['Time_Cycle', 'Diff_Pressure', 'dP_Smooth', 'Load_Index']
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1)
    rf.fit(full_hist[features], full_hist['RUL'])

    # B. Safety Model (Quantile Regression)
    # Simulate vibration data with heteroscedasticity
    X_safe = np.linspace(0, 10, 500).reshape(-1, 1)
    noise = np.random.normal(0, 0.3, 500) * (0.5 + X_safe.ravel()/3)
    y_safe = np.sin(X_safe).ravel() + noise
    
    degree = 3
    qr_low = make_pipeline(PolynomialFeatures(degree), QuantileRegressor(quantile=0.05, alpha=0, solver='highs-ds'))
    qr_high = make_pipeline(PolynomialFeatures(degree), QuantileRegressor(quantile=0.95, alpha=0, solver='highs-ds'))
    
    qr_low.fit(X_safe, y_safe)
    qr_high.fit(X_safe, y_safe)
    
    return rf, qr_low, qr_high

rf_model, safe_low, safe_high = train_models()

# --- 2. DATABASE & MQTT ---
DB_PATH = 'maintenance.db'

def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS iot_stream (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            unit_id INTEGER,
            sensor_val FLOAT,
            pressure FLOAT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            source TEXT
        )
    """)
    conn.commit()
    conn.close()

def mqtt_worker():
    def on_message(client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            conn.execute("INSERT INTO iot_stream (unit_id, sensor_val, pressure, source) VALUES (?, ?, ?, ?)",
                         (payload.get('unit_id', 1), payload.get('sensor_val', 0), payload.get('pressure', 1.0), 'MQTT'))
            conn.commit()
            conn.close()
        except:
            pass

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_message = on_message
    try:
        client.connect("test.mosquitto.org", 1883, 60)
        client.subscribe("gemini_demo/edge_ingest")
        client.loop_forever()
    except:
        print("MQTT Connection Failed")

if 'mqtt_active' not in st.session_state:
    init_db()
    t = threading.Thread(target=mqtt_worker, daemon=True)
    t.start()
    st.session_state['mqtt_active'] = True

# --- 3. UI LAYOUT ---
st.title("🏭 Azure Predictive Maintenance Hub")
st.markdown("Integrates **Batch RUL Prediction** (Random Forest) and **Real-Time Safety** (Quantile Regression).")

# Simulation Controls
with st.sidebar:
    st.header("Edge Simulation")
    if st.button("🔴 Simulate Anomaly (Unit 99)"):
        # Inject bad data via direct DB write (Simulating an MQTT hit)
        conn = sqlite3.connect(DB_PATH)
        conn.execute("INSERT INTO iot_stream (unit_id, sensor_val, pressure, source) VALUES (99, 2.5, 12.0, 'SIMULATOR')")
        conn.commit()
        conn.close()
        st.toast("Anomaly Injected!")
    
    if st.button("🟢 Simulate Normal Data"):
        conn = sqlite3.connect(DB_PATH)
        for i in range(5):
             conn.execute("INSERT INTO iot_stream (unit_id, sensor_val, pressure, source) VALUES (1, 0.5, 2.0, 'SIMULATOR')")
        conn.commit()
        conn.close()
        st.toast("Normal Data Injected")

# Load Live Data
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM iot_stream ORDER BY id DESC LIMIT 100", conn)
conn.close()

tab1, tab2, tab3 = st.tabs(["🚦 Fleet Triage", "🛡️ Safety Monitor", "📝 Data Stream"])

with tab1:
    st.subheader("Fleet Health Matrix")
    # Generate mock fleet for visualization
    np.random.seed(42)
    fleet_ages = np.random.randint(100, 1500, 50)
    fleet_pressures = 1.0 + (0.003 * fleet_ages**1.5) + np.random.normal(0, 0.2, 50)
    
    # Predict RUL for fleet
    fleet_data = pd.DataFrame({
        'Time_Cycle': fleet_ages,
        'Diff_Pressure': fleet_pressures,
        # FIX: Replaced .fillna(method='bfill') with .bfill() here too just in case
        'dP_Smooth': pd.Series(fleet_pressures).rolling(20).mean().bfill(), 
        'Load_Index': fleet_pressures * fleet_ages
    })
    fleet_ruls = rf_model.predict(fleet_data[['Time_Cycle', 'Diff_Pressure', 'dP_Smooth', 'Load_Index']])
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['red' if r < 200 else 'green' for r in fleet_ruls]
    sc = ax.scatter(fleet_ages, fleet_pressures, c=colors, s=100, alpha=0.7)
    ax.axhline(10.0, color='red', linestyle='--', label='Failure Threshold')
    ax.set_xlabel('Unit Age (Hours)')
    ax.set_ylabel('Differential Pressure')
    st.pyplot(fig)

with tab2:
    st.subheader("Real-Time Anomaly Detection (Quantile Tunnel)")
    
    if not df.empty:
        latest_val = df.iloc[0]['sensor_val']
        # Check bounds
        X_check = np.array([[8.5]]) 
        low_bound = safe_low.predict(X_check)[0]
        high_bound = safe_high.predict(X_check)[0]
        
        status = "NORMAL"
        status_color = "green"
        if latest_val < low_bound or latest_val > high_bound:
            status = "CRITICAL ANOMALY"
            status_color = "red"
            
        col1, col2, col3 = st.columns(3)
        col1.metric("Incoming Sensor Value", f"{latest_val:.2f}")
        col2.metric("Dynamic Safety Bounds", f"[{low_bound:.2f}, {high_bound:.2f}]")
        col3.markdown(f"Status: :**{status_color}[{status}]**")
        
        # Viz
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        X_plot = np.linspace(0, 10, 100).reshape(-1, 1)
        ax2.plot(X_plot, safe_low.predict(X_plot), 'g--', alpha=0.5)
        ax2.plot(X_plot, safe_high.predict(X_plot), 'g--', alpha=0.5)
        ax2.fill_between(X_plot.ravel(), safe_low.predict(X_plot), safe_high.predict(X_plot), color='green', alpha=0.1)
        
        # Plot latest point
        ax2.scatter([8.5], [latest_val], color=status_color, s=200, zorder=5, label='Latest Reading')
        st.pyplot(fig2)
    else:
        st.info("Waiting for data stream... Use Sidebar to Simulate.")

with tab3:
    st.dataframe(df)
