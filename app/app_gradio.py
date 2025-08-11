import gradio as gr
import joblib
import pandas as pd
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

if not models:
    raise FileNotFoundError("No model files found! Please place '_model*.pkl' files in the same directory.")

# Prediction function
def predict_heart_failure(model_name, age, creatinine_phosphokinase, ejection_fraction,
                          high_blood_pressure, platelets, serum_creatinine,
                          serum_sodium, smoking_status, time):

    selected_model = models[model_name]

    # Match the feature names used during training
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
    ]], columns=[
        'Age', 'CreatininePhosphokinase', 'EjectionFraction',
        'HighBloodPressure', 'PlateletCount', 'SerumCreatinine',
        'SerumSodium', 'SmokingStatus', 'FollowupDays'
    ])

    prediction = selected_model.predict(input_data)[0]

    return "⚠️ High risk of Heart Failure" if prediction == 1 else "✅ Low risk of Heart Failure"

# Build Gradio interface
model_names = list(models.keys())

iface = gr.Interface(
    fn=predict_heart_failure,
    inputs=[
        gr.Dropdown(model_names, label="Select Model"),
        gr.Slider(40, 95, value=60, step=1, label="Age"),
        gr.Slider(40, 8000, value=200, step=1, label="Creatinine Phosphokinase (mcg/L)"),
        gr.Slider(10, 70, value=40, step=1, label="Ejection Fraction (%)"),
        gr.Radio(["Yes", "No"], label="High Blood Pressure"),
        gr.Slider(47000, 500000, value=250000, step=1000, label="Platelet Count (kiloplatelets/mL)"),
        gr.Slider(0.5, 10.0, value=1.0, step=0.1, label="Serum Creatinine (mg/dL)"),
        gr.Slider(115, 150, value=137, step=1, label="Serum Sodium (mEq/L)"),
        gr.Radio(["Yes", "No"], label="Smoking Status"),
        gr.Slider(4, 300, value=100, step=1, label="Follow-up Days"),
    ],
    outputs=gr.Textbox(label="Prediction Result"),
    title="🫀 Heart Failure Prediction App",
    description="Select a model, enter patient data, and predict the risk of heart failure."
)

if __name__ == "__main__":
    iface.launch()
