import streamlit as st
import joblib
import pandas as pd
import numpy as np
import glob
import os

# Path to your saved models
model_path = "*.pkl"

# Load all saved models
model_files = glob.glob(model_path)
models = {}
for file in model_files:
    name = os.path.basename(file).replace(".pkl", "")
    models[name] = joblib.load(file)

# App title
st.title("Heart Failure Prediction App 🫀")

# Model selection
model_name = st.selectbox("Select a model", list(models.keys()))
selected_model = models[model_name]

st.write(f"**Selected model:** {model_name}")

# User input fields
st.subheader("Enter Patient Data")
age = st.slider("Age", min_value=40, max_value=95, value=60)
creatinine_phosphokinase = st.slider("Creatinine Phosphokinase (mcg/L)", min_value=40, max_value=8000, value=200)
ejection_fraction = st.slider("Ejection Fraction (%)", min_value=10, max_value=70, value=40)
high_blood_pressure = st.selectbox("High Blood Pressure", ["Yes", "No"])
platelets = st.slider("Platelet Count (kiloplatelets/mL)", min_value=47000, max_value=500000, value=250000, step=1000)
serum_creatinine = st.slider("Serum Creatinine (mg/dL)", min_value=0.5, max_value=10.0, value=1.0, step=0.1)
serum_sodium = st.slider("Serum Sodium (mEq/L)", min_value=115, max_value=150, value=137)
smoking_status = st.selectbox("Smoking Status", ["Yes", "No"])
time = st.slider("Follow-up Days", min_value=4, max_value=300, value=100)

# Predict button
if st.button("Predict"):
    # Prepare input data
    input_data = pd.DataFrame([[
        age,
        creatinine_phosphokinase,
        ejection_fraction,
        1 if high_blood_pressure == "Yes" else 0,
        platelets,
        serum_creatinine,
        serum_sodium,
        1 if smoking_status == "Yes" else 0,
        time
    ]], 
        columns=[
    'Age', 'CreatininePhosphokinase', 'EjectionFraction',
    'HighBloodPressure', 'PlateletCount', 'SerumCreatinine',
    'SerumSodium', 'SmokingStatus', 'FollowupDays'
    ])

    # Prediction
    prediction = selected_model.predict(input_data)[0]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("⚠️ High risk of Heart Failure")
    else:
        st.success("✅ Low risk of Heart Failure")
