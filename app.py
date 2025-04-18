import streamlit as st
import joblib
import numpy as np

# Load trained model and encoders
model = joblib.load("car_price_model.pkl")
le_fuel = joblib.load("le_fuel.pkl")
le_seller = joblib.load("le_seller.pkl")
le_trans = joblib.load("le_trans.pkl")
le_owner = joblib.load("le_owner.pkl")

# Set Streamlit page configuration
st.set_page_config(page_title="Car Price Predictor", layout="centered")

# Custom CSS for better UI
st.markdown("""
    <style>
        body {
            background-color: #f9f7f1;
            font-family: 'Segoe UI', sans-serif;
        }
        .main {
            background-color: #ffffff;
            padding: 30px 50px;
            border-radius: 15px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.1);
        }
        h1, h4 {
            color: #3E3E3E;
        }
        .stButton button {
            background-color: #6c5ce7;
            color: white;
            border-radius: 10px;
            padding: 10px 24px;
        }
        .stButton button:hover {
            background-color: #5a4bbd;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main'>", unsafe_allow_html=True)
st.title("🚗 Car Price Prediction App")
st.markdown("##### Enter car details below and get the estimated resale value!")

# Sidebar Inputs
year = st.number_input("Manufacturing Year", min_value=1990, max_value=2025, step=1)
km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=1000000, step=1000)

fuel = st.selectbox("Fuel Type", le_fuel.classes_)
seller_type = st.selectbox("Seller Type", le_seller.classes_)
transmission = st.selectbox("Transmission", le_trans.classes_)
owner = st.selectbox("Ownership", le_owner.classes_)

# Encode input
input_data = np.array([[
    year,
    km_driven,
    le_fuel.transform([fuel])[0],
    le_seller.transform([seller_type])[0],
    le_trans.transform([transmission])[0],
    le_owner.transform([owner])[0]
]])

# Predict button
if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Selling Price: ₹{int(prediction):,}")

st.markdown("</div>", unsafe_allow_html=True)
