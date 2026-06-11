import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Wide, professional layout setup
st.set_page_config(layout="wide")
st.title("🏙️ SonRite Energy Consultant: Metropolitan GHG Emissions Forecaster")
st.write("Simulate 30-day carbon emission trajectories based on building infrastructure footprints and utility consumption rates.")

# Folder path pointing to your frozen files
folder_path = r"D:\5.MBA\1.0 MBA 4-SEM\4.0 Machine Learning\ML Steps\data\appliances+energy+prediction-Resso\\"

# ==========================================
# SIDEBAR INTERFACE: INFRASTRUCTURE METRICS
# ==========================================
st.sidebar.header("🏢 Property Specifications")
building_area = st.sidebar.slider("Gross Floor Area (ft²)", 10000, 1000000, 150000, step=5000)
year_built = st.sidebar.slider("Year of Construction", 1900, 2025, 1975, step=1)
occupancy_rate = st.sidebar.slider("Building Occupancy Percentage (%)", 0, 100, 90, step=5)
num_buildings = st.sidebar.radio("Number of Structural Units on Loop", [1, 2, 3, 4, 5], index=0)

st.sidebar.markdown("---")

# ==========================================
# SIDEBAR INTERFACE: BASELINE UTILITY LOADS
# ==========================================
st.sidebar.header("⚡ Base Utility Consumption Rates")
base_elec = st.sidebar.slider("Baseline Daily Grid Electricity (kWh)", 1000, 100000, 25000, step=1000)
base_gas = st.sidebar.slider("Baseline Daily Natural Gas (kBtu)", 5000, 500000, 120000, step=5000)
base_fuel = st.sidebar.slider("Baseline Daily Fuel Oil #2 (kBtu)", 0, 200000, 30000, step=2000)
base_water = st.sidebar.slider("Baseline Daily Water Consumption (kgal)", 10, 2000, 450, step=10)

# ==========================================
# THE 30-DAY EMISSIONS SIMULATION
# ==========================================
if st.button("Generate 30-Day Carbon Forecast Report"):
    with st.spinner("Processing thermodynamic emissions vectors..."):
        
        # 1. Load the specialized carbon brains
        scaler = joblib.load(folder_path + 'nyc_carbon_scaler.pkl')
        model = joblib.load(folder_path + 'nyc_carbon_rf_model.pkl')
        
        # 2. Build our upcoming 30-day timeline
        days = np.arange(1, 31)
        
        # Simulate realistic daily operational shifts across a month
        # (Emissions fluctuate based on weekend production pullbacks and mid-week peaks)
        weekend_modifier = np.array([0.70 if (d % 7 == 5 or d % 7 == 6) else 1.05 for d in days])
        
        simulated_days_list = []
        for i in range(30):
            # Formulate the feature structure matching our 10 trained columns exactly:
            # ['DOF Gross Floor Area (ft²)', 'Year Built', 'Number of Buildings', 'Occupancy',
            #  'Electricity Use - Grid Purchase (kWh)', 'Natural Gas Use (kBtu)', 'Fuel Oil #2 Use (kBtu)',
            #  'Water Use (All Water Sources) (kgal)', 'Latitude', 'Longitude']
            day_profile = [
                building_area,
                year_built,
                num_buildings,
                occupancy_rate,
                base_elec * weekend_modifier[i],
                base_gas * weekend_modifier[i],
                base_fuel * weekend_modifier[i],
                base_water * weekend_modifier[i],
                40.75,   # Baseline Centralized Latitude Coordinate
                -73.98   # Baseline Centralized Longitude Coordinate
            ]
            simulated_days_list.append(day_profile)
            
        future_df = pd.DataFrame(simulated_days_list)
        
        # 3. Transform and predict
        future_scaled = scaler.transform(future_df)
        predictions = model.predict(future_scaled)
        
        # Ensure physical laws are bounded safely at zero
        predictions = np.clip(predictions, 0, None)
        
        # ==========================================
        # RENDER DASHBOARD VISUALIZATIONS
        # ==========================================
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.fill_between(days, predictions, color="#2c3e50", alpha=0.15)
        ax.plot(days, predictions, color="#e74c3c", marker='o', linewidth=2.5, label="Predicted Carbon Threshold")
        
        ax.set_title("30-Day Direct Carbon Footprint Tracking Projection", fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Operational Calendar Day (Days)", fontsize=10)
        ax.set_ylabel("Direct GHG Emissions (Metric Tons CO2e)", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xticks(days)
        ax.set_ylim(bottom=0)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.pyplot(fig)
            
        with col2:
            st.metric(label="Predicted Peak Day", value=f"{predictions.max():.2f} MT")
            st.metric(label="Predicted Minimum Day", value=f"{predictions.min():.2f} MT")
            st.metric(label="Total Projected Monthly Volume", value=f"{predictions.sum():.2f} MT")
            
        st.success("Industrial compliance forecasting successfully computed via Random Forest Engine!")