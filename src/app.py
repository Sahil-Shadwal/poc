import streamlit as st
import numpy as np
import plotly.graph_objects as go
from antenna import UniformRectangularArray

# Configure Streamlit Page
st.set_page_config(page_title="NTN Beamforming Dashboard", layout="wide")

st.title("🛰️ NTN Beamforming Interactive Dashboard")
st.markdown("This dashboard calculates and visualizes complex phased array beamforming weights for a simulation of a Non-Terrestrial Network (NTN) communication link in the Ka-Band (28 GHz).")

# Sidebar for controls
st.sidebar.header("Steering Controls")
st.sidebar.markdown("Drag the sliders to adjust the phase shifts across the 64 antenna elements and dynamically steer the RF beam.")
azimuth = st.sidebar.slider("Azimuth Angle (deg)", -90.0, 90.0, 0.0, 1.0)
elevation = st.sidebar.slider("Elevation Angle (deg)", 0.0, 90.0, 0.0, 1.0)

# Initialize Antenna Mathematical Engine
fc = 28e9 
array = UniformRectangularArray(8, 8, fc)

# Calculate Array Factor over a spherical grid
phi = np.linspace(-np.pi, np.pi, 45)
theta = np.linspace(0, np.pi/2, 45)
Phi, Theta = np.meshgrid(phi, theta)

# Retrieve array factor magnitudes based on desired steering vector
AF_mag = array.calculate_array_factor(Theta, Phi, azimuth, elevation)

# Convert to Cartesian for 3D plotting, scaled by Directivity Magnitude
X = AF_mag * np.sin(Theta) * np.cos(Phi)
Y = AF_mag * np.sin(Theta) * np.sin(Phi)
Z = AF_mag * np.cos(Theta)

# Construct Plotly 3D Surface
fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])

fig.update_layout(
    title=f"3D Beam Directivity Pattern",
    autosize=False,
    width=900,
    height=800,
    margin=dict(l=0, r=0, b=0, t=40),
    scene=dict(
        xaxis=dict(range=[-1, 1], title='X'),
        yaxis=dict(range=[-1, 1], title='Y'),
        zaxis=dict(range=[0, 1], title='Z (Gain)'),
        aspectratio=dict(x=1, y=1, z=0.8)
    )
)

st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Current Antenna Lock:**\n* Azimuth: {azimuth}°\n* Elevation: {elevation}°")
