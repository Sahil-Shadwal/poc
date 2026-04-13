import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from antenna import UniformRectangularArray

# Configure Streamlit Page
st.set_page_config(page_title="NTN Beamforming Dashboard", layout="wide")

st.title("🛰️ NTN Beamforming Interactive Dashboard")
st.markdown("This system computes mathematical beamforming responses and simulates directivity patterns for NTN environments, satisfying strict REQ 1 functional constraints.")

# Sidebar Controls for Full Configuration
st.sidebar.header("1. Antenna Configuration")
num_x = st.sidebar.number_input("Elements (X-Axis)", min_value=2, max_value=32, value=8)
num_y = st.sidebar.number_input("Elements (Y-Axis)", min_value=2, max_value=32, value=8)
freq_ghz = st.sidebar.number_input("Frequency (GHz)", min_value=1.0, max_value=100.0, value=28.0)

st.sidebar.header("2. Steering Controls")
azimuth = st.sidebar.slider("Azimuth Angle (deg)", -90.0, 90.0, 0.0, 1.0)
elevation = st.sidebar.slider("Elevation Angle (deg)", 0.0, 90.0, 0.0, 1.0)

# Initialize Antenna Mathematics dynamically based on UI inputs
fc = freq_ghz * 1e9 
array = UniformRectangularArray(int(num_x), int(num_y), fc)

# Calculate Array Factor over a spherical grid for 3D
phi = np.linspace(-np.pi, np.pi, 45)
theta = np.linspace(0, np.pi/2, 45)
Phi, Theta = np.meshgrid(phi, theta)
AF_mag_3d = array.calculate_array_factor(Theta, Phi, azimuth, elevation)

# Convert to Cartesian for 3D plotting
X = AF_mag_3d * np.sin(Theta) * np.cos(Phi)
Y = AF_mag_3d * np.sin(Theta) * np.sin(Phi)
Z = AF_mag_3d * np.cos(Theta)

# Calculate distinct 2D Frontal cut
phi_1d = np.linspace(-np.pi, np.pi, 360)
theta_1d = np.full_like(phi_1d, np.pi/2) # Simplified Azimuth Horizon cut
AF_mag_2d = array.calculate_array_factor(np.array([theta_1d]), np.array([phi_1d]), azimuth, elevation).flatten()

# Split page into layout columns
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("3D Volume Visualization")
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
    fig.update_layout(
        autosize=False, height=600, margin=dict(l=0, r=0, b=0, t=20),
        scene=dict(xaxis=dict(range=[-1, 1]), yaxis=dict(range=[-1, 1]), zaxis=dict(range=[0, 1]))
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("2D Azimuth Pattern")
    fig_2d = go.Figure(go.Scatterpolar(r=AF_mag_2d, theta=np.degrees(phi_1d), mode='lines'))
    fig_2d.update_layout(polar=dict(radialaxis=dict(range=[0, 1])), height=400, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_2d, use_container_width=True)
    
    st.subheader("Export Results Data")
    df = pd.DataFrame({"Azimuth_Deg": np.degrees(phi_1d), "Gain_Normalized": AF_mag_2d})
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download 2D Pattern (CSV)", data=csv, file_name="pattern_export.csv", mime="text/csv")
