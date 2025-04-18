import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load trained model and encoders
model = joblib.load("car_price_model.pkl")
le_fuel = joblib.load("le_fuel.pkl")
le_seller = joblib.load("le_seller.pkl")
le_trans = joblib.load("le_trans.pkl")
le_owner = joblib.load("le_owner.pkl")

# Page Config
st.set_page_config(page_title="Car Price Predictor", layout="wide")

# Custom CSS
st.markdown("""
    <style>
        body {
            background-color: #f4f4f4;
        }
        .main {
            background-color: #fff;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.05);
        }
        .stButton button {
            background-color: #6c5ce7;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 10px 24px;
        }
        .stButton button:hover {
            background-color: #5a4bbd;
        }
    </style>
""", unsafe_allow_html=True)

# Main Layout
st.markdown("<div class='main'>", unsafe_allow_html=True)
st.title("🚗 Car Price Prediction")
st.markdown("#### Get an estimated resale price based on your car details")

# Input section
st.sidebar.header("🛠️ Input Car Details")

year = st.sidebar.number_input("Manufacturing Year", min_value=1990, max_value=2025, step=1)
km_driven = st.sidebar.number_input("Kilometers Driven", min_value=0, max_value=1000000, step=1000)

fuel = st.sidebar.selectbox("Fuel Type", le_fuel.classes_)
seller_type = st.sidebar.selectbox("Seller Type", le_seller.classes_)
transmission = st.sidebar.selectbox("Transmission", le_trans.classes_)
owner = st.sidebar.selectbox("Ownership", le_owner.classes_)

# Encode for prediction
input_data = np.array([[year,
                        km_driven,
                        le_fuel.transform([fuel])[0],
                        le_seller.transform([seller_type])[0],
                        le_trans.transform([transmission])[0],
                        le_owner.transform([owner])[0]]])

# Predict button
if st.sidebar.button("🔍 Predict Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Selling Price: ₹{int(prediction):,}")

    # Regression Plot Section
    st.markdown("### 📊 Regression Price Comparison")

    # Dummy data for visualization (replace with real test set if available)
    sample_data = pd.DataFrame({
        "Predicted Price": model.predict(np.random.randint(1995, 2022, size=(100, 1)) * np.ones((100, 6))),
        "Actual Price": np.random.randint(1, 10, 100) * 100000
    })

    # Add the current prediction
    new_data = pd.DataFrame({
        "Predicted Price": [prediction],
        "Actual Price": [prediction]  # assuming we don't have actual here
    })

    sample_data = pd.concat([sample_data, new_data], ignore_index=True)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=sample_data[:-1], x="Actual Price", y="Predicted Price", alpha=0.5, label="Samples")
    sns.scatterplot(data=new_data, x="Actual Price", y="Predicted Price", color="red", s=100, label="Your Car")
    plt.plot([0, sample_data["Actual Price"].max()], [0, sample_data["Actual Price"].max()], color='green', linestyle='--')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted Car Prices")
    plt.legend()
    st.pyplot(fig)

st.markdown("</div>", unsafe_allow_html=True)
