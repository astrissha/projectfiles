import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import folium
import streamlit.components.v1 as components
from datetime import datetime

# Load trained model
automl = joblib.load("smartguard_model.pkl")

# Required feature columns
feature_columns = [
    'timestamp', 'engine_id', 'rpm', 'oil_pressure', 'oil_temp',
    'fuel_pressure', 'coolant_temp', 'vibration_level',
    'exhaust_gas_temp', 'engine_load', 'ambient_temp',
    'humidity', 'altitude', 'hours_since_maintenance',
    'latitude', 'longitude'
]

# Streamlit config and style
st.set_page_config(page_title="SMARTGUARD Digital Twin", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-color: #001f3f; 
        color: white;
    }
    h1, h2, h3, label, .stMarkdown, .stNumberInput > label {
        color: white !important;
    }
    .stButton > button {
        background-color: #0074D9;
        color: white;
        border-radius: 5px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Logo
st.image("WhatsApp Image 2025-04-13 at 07.16.33.jpeg", width=120)

# Title and description
st.markdown("<h1 style='text-align: center;'>🚛 SMARTGUARD: Military Engine Digital Twin</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>Monitor, Predict & Visualize Military Engine Health Using AI</p>", unsafe_allow_html=True)

# Clock
current_time = datetime.now().strftime("%H:%M:%S")
st.markdown(f"<h4 style='text-align:right; color:gray;'>🕒 {current_time}</h4>", unsafe_allow_html=True)

# File uploader
st.subheader("📄 Upload Engine Sensor Data (Excel Format)")
uploaded_file = st.file_uploader("Upload .xlsx file with all required columns", type=["xlsx"])

# Prediction function
def digital_twin_prediction(df):
    df = df[feature_columns]
    prediction = automl.predict(df)
    return prediction

# Map function
def create_geolocation_map(lat, lon):
    m = folium.Map(location=[lat, lon], zoom_start=12)
    folium.Marker([lat, lon], popup=f"Latitude: {lat}, Longitude: {lon}").add_to(m)
    return m

if uploaded_file is not None:
    df_input = pd.read_excel(uploaded_file)

    if set(feature_columns).issubset(df_input.columns):
        preds = digital_twin_prediction(df_input)
        df_input['prediction'] = preds

        label_map = {0: "normal", 1: "warning", 2: "critical"}
        emoji_map = {"normal": "🟢 NORMAL", "warning": "🟡 WARNING", "critical": "🔴 CRITICAL"}

        for idx, row in df_input.iterrows():
            label = label_map.get(row['prediction'], "unknown")
            emoji = emoji_map.get(label, "❓ UNKNOWN")

            st.markdown(f"<h3>🛠️ Engine ID: {int(row['engine_id'])} - Health Status: {emoji}</h3>", unsafe_allow_html=True)

            # Show 3D Visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(
                x=[row['rpm']],
                y=[row['vibration_level']],
                z=[row['engine_load']],
                mode='markers',
                marker=dict(size=10, color='red'),
                name='Sensor'
            ))
            fig.update_layout(
                scene=dict(
                    xaxis_title='RPM',
                    yaxis_title='Vibration',
                    zaxis_title='Engine Load'
                ),
                margin=dict(l=0, r=0, b=0, t=0),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

            # Show Location Map
            st.subheader("📍 Engine Location on Map")
            map_obj = create_geolocation_map(row['latitude'], row['longitude'])
            components.html(map_obj._repr_html_(), height=400)
            st.markdown("---")

    else:
        st.error("❌ Uploaded file must contain all required columns:\n\n" + ", ".join(feature_columns))
