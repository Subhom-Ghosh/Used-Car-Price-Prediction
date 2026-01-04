import streamlit as st
import numpy as np
import xgboost as xgb
import joblib
import pickle

# --- 1. LOAD ASSETS ---
model = joblib.load("car_price_model.pkl")

# Load the mappings file you just created
with open('car_mappings.pkl', 'rb') as f:
    mappings = pickle.load(f)

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")
st.title("🚗 Used Car Price Prediction")
st.sidebar.header("Enter Car Details")

# --- 2. INPUTS (Using Dropdowns for Text) ---

# BRAND: Show names, but we will look up the number later
brand_name = st.sidebar.selectbox("Brand", mappings['brand']['classes'])

# MODEL: Show names
model_name = st.sidebar.selectbox("Model", mappings['model']['classes'])

# MODEL YEAR (Number is fine)
model_year = st.sidebar.number_input("Model Year", min_value=1990, max_value=2026, value=2018)

# MILEAGE
milage = st.sidebar.number_input("Mileage (miles)", min_value=0, value=40000)

# FUEL TYPE
fuel_name = st.sidebar.selectbox("Fuel Type", mappings['fuel_type']['classes'])

# HORSEPOWER
horsepower = st.sidebar.number_input("Horsepower", min_value=50, value=150)

# TRANSMISSION
trans_name = st.sidebar.selectbox("Transmission", mappings['transmission']['classes'])

# ACCIDENT & TITLE (Manual mapping)
accident_input = st.sidebar.selectbox("Accident History", ["None", "At least one"])
clean_title_input = st.sidebar.selectbox("Clean Title", ["Yes", "No"])

if st.button("Predict Price 💰"):
    # 1. Calculate Dependent Features
    # The error says the model expects 'model_year' AND 'car-age' AND 'miles_per_year'
    car_age = 2025 - model_year
    miles_per_year = milage / (car_age + 1)

    # 2. Encode Categoricals
    # Use the mappings we loaded
    brand_encoded = mappings['brand']['encoder'].transform([brand_name])[0]
    model_encoded = mappings['model']['encoder'].transform([model_name])[0]
    fuel_encoded = mappings['fuel_type']['encoder'].transform([fuel_name])[0]
    trans_encoded = mappings['transmission']['encoder'].transform([trans_name])[0]

    # Map Binary
    accident_val = 1 if accident_input == "At least one" else 0
    clean_title_val = 1 if clean_title_input == "Yes" else 0

    # 3. Create Input Array
    # CRITICAL: This order matches your Error Message's "Expected" list exactly:
    # ['model_year', 'milage', 'fuel_type', 'transmission', 'accident', 'clean_title', 
    #  'car-age', 'horsepower', 'miles_per_year', 'brand_encoded', 'model_encoded']
    
    input_data = np.array([
        model_year,       # 1
        milage,           # 2
        fuel_encoded,     # 3
        trans_encoded,    # 4
        accident_val,     # 5
        clean_title_val,  # 6
        car_age,          # 7
        horsepower,       # 8
        miles_per_year,   # 9
        brand_encoded,    # 10
        model_encoded     # 11
    ]).reshape(1, -1)

    # 4. Create DMatrix with Correct Feature Names
    # These names must match the "Expected" list from your error
    feature_names = [
        'model_year', 'milage', 'fuel_type', 'transmission', 'accident', 
        'clean_title', 'car-age', 'horsepower', 'miles_per_year', 
        'brand_encoded', 'model_encoded'
    ]
    
    dmatrix_data = xgb.DMatrix(input_data, feature_names=feature_names)

    # 5. Predict
    try:
        log_price = model.predict(dmatrix_data)[0]
        price = np.expm1(log_price)
        st.success(f"💵 Estimated Car Price: **${int(price):,}**")
    except Exception as e:
        st.error(f"Prediction Error: {e}")