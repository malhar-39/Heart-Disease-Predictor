import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('best_model.pkl')

# Define the app
st.title("Heart Disease Prediction")

# Input fields for user data
age = st.number_input("Age", min_value=0, max_value=120, value=50)
gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
chest_pain_type = st.number_input("Chest Pain Type", min_value=0, max_value=3, value=0)
resting_blood_pressure = st.number_input("Resting Blood Pressure", min_value=0, max_value=300, value=120)
serum_cholesterol = st.number_input("Serum Cholesterol", min_value=0, max_value=600, value=200)
fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
resting_ecg_results = st.number_input("Resting ECG Results", min_value=0, max_value=2, value=0)
maximum_heart_rate = st.number_input("Maximum Heart Rate", min_value=0, max_value=220, value=150)
exercise_induced_angina = st.selectbox("Exercise Induced Angina", options=[0, 1])
st_depression = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0)
slope_of_st_segment = st.number_input("Slope of ST Segment", min_value=0, max_value=2, value=0)
number_of_major_vessels = st.number_input("Number of Major Vessels", min_value=0, max_value=4, value=0)
thalassemia_type = st.number_input("Thalassemia Type", min_value=0, max_value=3, value=0)

# Create a numpy array from the input
input_data = np.array([[age, gender, chest_pain_type, resting_blood_pressure, serum_cholesterol, fasting_blood_sugar,
                        resting_ecg_results, maximum_heart_rate, exercise_induced_angina, st_depression,
                        slope_of_st_segment, number_of_major_vessels, thalassemia_type]])

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.write(f"Prediction: The patient is likely to have heart disease. Probability: {probability:.2f}")
    else:
        st.write(f"Prediction: The patient is unlikely to have heart disease. Probability: {probability:.2f}")


# streamlit run app.py
