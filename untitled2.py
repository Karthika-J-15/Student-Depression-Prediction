import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page layout
st.set_page_config(layout="centered")

# Inject custom CSS
st.markdown("""
    <style>
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1616531770075-66026c7603f0?auto=format&fit=crop&w=1950&q=80');
        background-size: cover;
        background-position: center;
        color: white;
    }
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        background: rgba(0, 0, 0, 0.6); /* semi-transparent dark background */
        border-radius: 15px;
    }
    h1, h2, h3 {
        color: #ffcc00;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Session state for navigation
if "started" not in st.session_state:
    st.session_state.started = False

# Welcome screen
if not st.session_state.started:
    st.markdown("<h1>🎓 Welcome to the Depression Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>This app predicts depression likelihood in students based on academic and lifestyle inputs.</p>", unsafe_allow_html=True)
    if st.button("Start"):
        st.session_state.started = True
    st.stop()

# Load trained model & transformer
pca = joblib.load("pca_transformer.pkl")
model = joblib.load("logistic_model.pkl")

# Category mappings
sleep_duration_map = {
    'Less than 5 hours': 0,
    'More than 9 hours': 1,
    'between 5-6 hours': 2,
    'between 7-8 hours': 3
}

dietary_habits_map = {
    'Healthy': 0,
    'Moderate': 1,
    'Unhealthy': 2
}

# App content
st.title("🧠 Depression Prediction")

age = st.slider("Age", 15, 40, 20)
academic_pressure = st.slider("Academic Pressure (0-5)", 0, 5, 2)
study_hours = st.slider("Study Hours per Day", 0, 12, 4)
sleep_duration = st.selectbox("Sleep Duration", list(sleep_duration_map.keys()))
dietary_habit = st.selectbox("Dietary Habit", list(dietary_habits_map.keys()))

# Encode categorical inputs
sleep_encoded = sleep_duration_map[sleep_duration]
diet_encoded = dietary_habits_map[dietary_habit]

# Prepare input
input_df = pd.DataFrame([[age, academic_pressure, study_hours, sleep_encoded, diet_encoded]],
                        columns=["Age", "Academic Pressure", "Study Hours", "Sleep Duration_encoded", "Dietary Habits_encoded"])

# Predict
input_pca = pca.transform(input_df)
prediction = model.predict(input_pca)[0]
probability = model.predict_proba(input_pca)[0][1]

# Display results
st.subheader("📊 Prediction Result:")
st.success("🚨 Depressed" if prediction == 1 else "✅ Not Depressed")
st.write(f"🔢 Probability of Depression: **{probability:.2%}**")


