import streamlit as st
import joblib
import numpy as np


model = joblib.load("copd_model.pkl")


st.title("🩺 COPD Risk Predictor")


age = st.number_input("Enter your age", min_value=1, max_value=120, step=1)
smoking_input = st.radio("Do you smoke?", ["Yes", "No"])
smoking = 1 if smoking_input == "Yes" else 0

breath_input = st.radio("Shortness of breath?", ["Yes", "No"])
breath = 1 if breath_input == "Yes" else 0

cough_input = st.radio("Coughing frequently?", ["Yes", "No"])
cough = 1 if cough_input == "Yes" else 0

lung = st.slider("Lung function score (1–100)", 1, 100)

if st.button("Predict COPD Risk"):
    user_input = np.array([[age, smoking, breath, cough, lung]])
    result = model.predict(user_input)

    if result[0] == 1:
        st.error("⚠️ High risk of COPD. Please consult a doctor.")
    else:
        st.success("✅ Low risk of COPD. Stay healthy!")
