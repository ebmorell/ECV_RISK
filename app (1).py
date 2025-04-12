import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import gdown

# --- ID del archivo en Google Drive (sustituye con el tuyo) ---
MODEL_ID = "1YE0_SDXNQbI3G0WVxiXVdEpNnvuguope"  # ← Sustituye este valor con el ID real
MODEL_FILENAME = "modelo_rsf_ecv.pkl"

# --- Descarga automática desde Drive si no existe ---
if not os.path.exists(MODEL_FILENAME):
    with st.spinner("Descargando modelo desde Google Drive..."):
        url = f"https://drive.google.com/uc?id={MODEL_ID}"
        gdown.download(url, MODEL_FILENAME, quiet=False)

# --- Cargar modelo ---
@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILENAME)

rsf = load_model()

# --- Interfaz ---
st.title("Predicción de Riesgo de Evento Cardiovascular (ECV) en Personas con VIH")
st.markdown("Introduce los datos clínicos del paciente para estimar su riesgo de ECV a 5 años.")

# --- Formulario de entrada ---
def input_to_df():
    data = {
        "Sex": st.selectbox("Sexo", ["Hombre", "Mujer"]),
        "Transmission_mode": st.selectbox("Modo de transmisión", ["Homo/Bisexual", "Heterosexual", "Otros"]),
        "Origin": st.selectbox("Origen", ["España", "No España"]),
        "Education_Level": st.selectbox("Nivel educativo", ["Primaria", "Secundaria", "Universidad"]),
        "AIDS": st.selectbox("¿Diagnóstico de sida?", ["No", "Sí"]),
        "Age": st.number_input("Edad", 18, 90, 45),
        "Viral_Load": st.selectbox("Carga viral suprimida", ["Sí", "No"]),
        "ART": st.selectbox("Clase TAR", ["2NRTI+NNRTI", "2NRTI+IP", "2NRTI+II", "Otro"]),
        "Hepatitis_C": st.selectbox("Hepatitis C", ["No", "Sí"]),
        "Anticore_HBV": st.selectbox("Anticore VHB positivo", ["No", "Sí"]),
        "CD4_Nadir": st.number_input("CD4 nadir", 0, 2000, 350),
        "CD8_Nadir": st.number_input("CD8 nadir", 0, 2000, 800),
        "CD4_CD8_Ratio": st.number_input("Ratio CD4/CD8", 0.0, 3.0, 0.9),
        "HBP": st.selectbox("Hipertensión arterial", ["No", "Sí"]),
        "Smoking": st.selectbox("Tabaquismo", ["Nunca", "Exfumador", "Actual"]),
        "Cholesterol": st.number_input("Colesterol total", 100, 400, 200),
        "HDL": st.number_input("HDL", 20, 150, 50),
        "Triglycerides": st.number_input("Triglicéridos", 50, 600, 150),
        "Non_HDL_Cholesterol": st.number_input("Colesterol no-HDL", 80, 350, 150),
        "Triglyceride_HDL_Ratio": st.number_input("Relación TG/HDL", 0.1, 10.0, 3.0),
        "Diabetes": st.selectbox("Diabetes", ["No", "Sí"]),
    }
    return pd.DataFrame([data])

input_df = input_to_df()

# --- Codificación manual (como en entrenamiento) ---
encoding_dict = {
    "Sex": {"Hombre": 0, "Mujer": 1},
    "Transmission_mode": {"Homo/Bisexual": 0, "Heterosexual": 1, "Otros": 2},
    "Origin": {"España": 0, "No España": 1},
    "Education_Level": {"Primaria": 0, "Secundaria": 1, "Universidad": 2},
    "AIDS": {"No": 0, "Sí": 1},
    "Viral_Load": {"Sí": 1, "No": 0},
    "ART": {"2NRTI+NNRTI": 0, "2NRTI+IP": 1, "2NRTI+II": 2, "Otro": 3},
    "Hepatitis_C": {"No": 0, "Sí": 1},
    "Anticore_HBV": {"No": 0, "Sí": 1},
    "HBP": {"No": 0, "Sí": 1},
    "Smoking": {"Nunca": 0, "Exfumador": 1, "Actual": 2},
    "Diabetes": {"No": 0, "Sí": 1}
}

for col in encoding_dict:
    input_df[col] = input_df[col].map(encoding_dict[col])

# --- Predicción ---
if st.button("Predecir riesgo de ECV"):
    pred_surv = rsf.predict_survival_function(input_df, return_array=True)
    time_points = rsf.event_times_
    t5_index = np.searchsorted(time_points, 5)
    risk_5yr = 1 - pred_surv[0][t5_index]

    # Clasificación de riesgo con colores
    if risk_5yr < 0.10:
        categoria = "Bajo"
        color = "green"
    elif risk_5yr < 0.20:
        categoria = "Moderado"
        color = "orange"
    else:
        categoria = "Alto"
        color = "red"

    st.markdown(
        f"<h3>Riesgo estimado a 5 años: <span style='color:{color}'>{risk_5yr:.1%} ({categoria})</span></h3>",
        unsafe_allow_html=True
    )

    # Curva de supervivencia
    plt.figure(figsize=(8, 4))
    plt.step(time_points, pred_surv[0], where="post", label="Curva de supervivencia")
    plt.axvline(5, color="red", linestyle="--", label="5 años")
    plt.xlabel("Tiempo (años)")
    plt.ylabel("Probabilidad de no tener ECV")
    plt.title("Curva de supervivencia estimada")
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)
