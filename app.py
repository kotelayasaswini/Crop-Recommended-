import streamlit as st
import numpy as np
import pickle

# ----------------------------------
# Page config
# ----------------------------------
st.set_page_config(
    page_title="Crop Recommendation ",
    layout="centered"
)

st.title("Crop Recommendation")
st.write("Enter the properties to predict crop")

# ----------------------------------
# Load model and scaler
# ----------------------------------
@st.cache_resource
def load_artifacts():
    with open("RFmodel.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, scaler

model, scaler = load_artifacts()

# ----------------------------------
# Feature inputs
# ----------------------------------
feature_inputs = {
    'Nitrogen': st.number_input('Nitrogen', min_value=0, value=140),
    'P': st.number_input('p', min_value=0, value=140),
    'K': st.number_input('K', min_value=0, value=140),
    'temperature': st.number_input('temperature', min_value=0, value=40),
    'humidity': st.number_input('humidity', min_value=10, value=100),
    'ph': st.number_input('ph', min_value=3, value=10),
    'rainfall': st.number_input('rainfall', min_value=0, value=300),
}

# Maintain correct feature order
feature_names = list(feature_inputs.keys())
input_values = [feature_inputs[f] for f in feature_names]

# ----------------------------------
# Prediction
# ----------------------------------
if st.button("Predict Crop Name"):
    input_array = np.array(input_values).reshape(1, -1)

    # Scale input
    scaled_input = scaler.transform(input_array)

    # Predict
    prediction = model.predict(scaled_input)

    st.success(f"Predicted Crop Name: **{(prediction[0])}**")