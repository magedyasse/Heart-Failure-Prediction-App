import gradio as gr
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
import matplotlib.pyplot as plt
import glob
import os
import requests

# Direct download links for your models
MODEL_URLS = {
    "DecisionTreeClassifier_model": "https://drive.google.com/uc?export=download&id=14C-UMmPKILhbmoQStE1JuLWEp3QAyP8v",
    "LogisticRegression_model": "https://drive.google.com/uc?export=download&id=1dSkRFICO_s7ermb-T9ehnX2FUDlf7eJA",
    "RandomForestClassifier_model": "https://drive.google.com/uc?export=download&id=1BbqyG-rGEll29jNyKDi173pkW9scJK3N",
    "SVC_model": "https://drive.google.com/uc?export=download&id=120e8J1t19yXkGSs9CtEaFTlfGqvms4yB"
}

# Step 1: Check for local models first
model_files = glob.glob("models/*.pkl")

# Step 2: If models not found locally, download them
if len(model_files) < len(MODEL_URLS):
    os.makedirs("models", exist_ok=True)
    for model_name, url in MODEL_URLS.items():
        file_path = f"models/{model_name}.pkl"
        if not os.path.exists(file_path):
            print(f"Downloading {model_name}...")
            r = requests.get(url)
            r.raise_for_status()
            with open(file_path, "wb") as f:
                f.write(r.content)
    model_files = glob.glob("models/*.pkl")

# Step 3: Load all models
models = {}
for file_path in model_files:
    model_name = os.path.basename(file_path).replace(".pkl", "")
    print(f"Loading model: {model_name}")
    with open(file_path, "rb") as model_file:
        models[model_name] = joblib.load(model_file)

if not models:
    raise FileNotFoundError("No model files found locally or via download.")

print("✅ All models loaded successfully!")

# Prediction function with charts
def predict_with_chart(model_name, age, creatinine_phosphokinase, ejection_fraction,
                       high_blood_pressure, platelets, serum_creatinine,
                       serum_sodium, smoking_status, time):
    
    model = models[model_name]

    # Prepare input
    input_data = pd.DataFrame([[age, creatinine_phosphokinase, ejection_fraction,
                                1 if high_blood_pressure == "Yes" else 0,
                                platelets, serum_creatinine, serum_sodium,
                                1 if smoking_status == "Yes" else 0, time]],
                              columns=['Age', 'CreatininePhosphokinase', 'EjectionFraction',
                                       'HighBloodPressure', 'PlateletCount', 'SerumCreatinine',
                                       'SerumSodium', 'SmokingStatus', 'FollowupDays'])

    # Prediction
    pred = model.predict(input_data)[0]
    result_text = "⚠️ High risk of Heart Failure" if pred == 1 else "✅ Low risk of Heart Failure"

    # Probability chart (only if available)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_data)[0]
        fig, ax = plt.subplots()
        ax.bar(["Low Risk", "High Risk"], probs, color=["green", "red"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Probability")
        prob_fig = fig
    else:
        # Create placeholder chart
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Probability chart not available", ha='center', va='center')
        ax.axis("off")
        prob_fig = fig

   
 

    # Close plots after use to save memory
    plt.close(fig)
    

    return result_text, prob_fig

# Interface
model_names = list(models.keys())

iface = gr.Interface(
    fn=predict_with_chart,
    inputs=[
        gr.Dropdown(model_names, label="Select Model"),
        gr.Slider(40, 95, value=60, step=1, label="Age"),
        gr.Slider(40, 8000, value=200, step=1, label="Creatinine Phosphokinase (mcg/L)"),
        gr.Slider(10, 70, value=40, step=1, label="Ejection Fraction (%)"),
        gr.Radio(["Yes", "No"], label="High Blood Pressure", value="No"),
        gr.Slider(47000, 500000, value=250000, step=1000, label="Platelet Count (kiloplatelets/mL)"),
        gr.Slider(0.5, 10.0, value=1.0, step=0.1, label="Serum Creatinine (mg/dL)"),
        gr.Slider(115, 150, value=137, step=1, label="Serum Sodium (mEq/L)"),
        gr.Radio(["Yes", "No"], label="Smoking Status", value="No"),
        gr.Slider(4, 300, value=100, step=1, label="Follow-up Days"),
    ],
    outputs=[
        gr.Textbox(label="Prediction Result"),
        gr.Plot(label="Probability Chart")
      
    ],
    title="🫀 Heart Failure Prediction App",
    description="Select a model, enter patient data, and see both prediction and charts."
)

if __name__ == "__main__":
    iface.launch(share=False)
