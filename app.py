import joblib
import pandas as pd
import streamlit as st

# Load trained pipeline
model = joblib.load("delay_model.pkl")

st.title("🚆 Mumbai Local Train Delay Analytics & Prediction")

# User input form
distance = st.number_input("Distance (km)", min_value=1.0)
speed = st.number_input("Speed (kmph)", min_value=10.0)
passengers = st.number_input("Passengers daily", min_value=1000)
station = st.text_input("Station")
line = st.text_input("Line")

if st.button("Predict Delay Time"):
    input_data = pd.DataFrame([[distance, speed, passengers, station, line]],
                              columns=["Distance_km", "Speed_kmph", "Passengers_daily", "Station", "Line"])
    prediction = model.predict(input_data)
    st.success(f"Predicted Time (minutes): {prediction[0]:.2f}")

