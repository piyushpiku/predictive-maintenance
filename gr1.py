import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- Plot 1: The Physics RUL Curve ---
def plot_rul_physics():
    t = np.arange(0, 1500)
    # Physics Formula: dP = 1 + k * t^1.5
    dp = 1.0 + (0.003 * t**1.5)
    
    plt.figure(figsize=(10, 5))
    plt.plot(t, dp, color='#1f77b4', linewidth=3, label='Physics Model (Degradation)')
    plt.axhline(40, color='red', linestyle='--', label='Failure Threshold')
    plt.title("Physics-Informed Degradation Model (Threshold Logic)", fontsize=14)
    plt.xlabel("Time Cycles (Hours)", fontsize=12)
    plt.ylabel("Differential Pressure (psi)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('assets/model_physics_rul.png', dpi=300)
    print("✅ Generated: assets/model_physics_rul.png")

# --- Plot 2: Static vs. Dynamic Anomaly Detection ---
def plot_anomaly_comparison():
    x = np.linspace(0, 10, 100)
    # Heteroscedastic Signal (Noise grows with X)
    np.random.seed(42)
    noise = np.random.normal(0, 0.5, 100) * (0.5 + x/2)
    y = np.sin(x) + noise
    
    # Quantile Bounds (Simplified for Viz)
    upper_bound = np.sin(x) + (1.0 + x/2) # Dynamic
    static_threshold = 2.5 # Static
    
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, 'o', color='gray', alpha=0.5, label='Noisy Sensor Data')
    
    # Dynamic Tunnel
    plt.fill_between(x, np.sin(x) - (1.0+x/2), upper_bound, color='green', alpha=0.1, label='Quantile Safety Tunnel (95%)')
    plt.plot(x, upper_bound, color='green', linestyle='--')
    
    # Static Threshold
    plt.axhline(static_threshold, color='red', linewidth=2, label='Static Threshold (Traditional)')
    
    plt.title("Why Static Thresholds Fail: Quantile Regression vs. Traditional Limits", fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('assets/model_anomaly_comparison.png', dpi=300)
    print("✅ Generated: assets/model_anomaly_comparison.png")

if __name__ == "__main__":
    plot_rul_physics()
    plot_anomaly_comparison()
