import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🏠 House Price Predictor")
st.markdown("""
This application predicts house prices based on various features. Enter the details
of the house below to get an estimated price.
""")

# Load the model
@st.cache_resource
def load_model():
    return joblib.load('./models/house_price_model.joblib')

model = load_model()

# Create input sections
st.header("House Features")

# Create columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Basic Information")
    lot_area = st.number_input("Lot Area (sq ft)", min_value=1000, max_value=200000, value=10000)
    year_built = st.number_input("Year Built", min_value=1800, max_value=2024, value=2000)
    overall_qual = st.slider("Overall Quality", 1, 10, 5, help="1 is poor, 10 is excellent")
    
with col2:
    st.subheader("Living Area")
    first_flr = st.number_input("First Floor Area (sq ft)", min_value=0, max_value=5000, value=1000)
    second_flr = st.number_input("Second Floor Area (sq ft)", min_value=0, max_value=5000, value=0)
    basement = st.number_input("Total Basement Area (sq ft)", min_value=0, max_value=5000, value=0)
    
with col3:
    st.subheader("Additional Features")
    garage_area = st.number_input("Garage Area (sq ft)", min_value=0, max_value=2000, value=0)
    full_bath = st.number_input("Full Bathrooms", min_value=0, max_value=5, value=2)
    bedrooms = st.number_input("Bedrooms Above Ground", min_value=0, max_value=10, value=3)

# Calculate derived features
total_sf = first_flr + second_flr + basement
total_bathrooms = full_bath  # Simplified for the demo
house_age = 2024 - year_built

# Create a dictionary of features in the correct order
def prepare_features():
    features = {
        'LotArea': lot_area,
        'YearBuilt': year_built,
        'TotalBsmtSF': basement,
        '1stFlrSF': first_flr,
        '2ndFlrSF': second_flr,
        'BedroomAbvGr': bedrooms,
        'FullBath': full_bath,
        'GarageArea': garage_area,
        'OverallQual': overall_qual,
        'TotalSF': total_sf,
        'TotalBathrooms': total_bathrooms,
        'HouseAge': house_age
    }
    return pd.DataFrame([features])

# Add a predict button
if st.button("Predict Price"):
    # Prepare the features
    X = prepare_features()
    
    # Make prediction (model expects log-transformed target)
    log_price = model.predict(X)[0]
    
    # Transform back to actual price
    predicted_price = np.exp(log_price)
    
    # Display the prediction
    st.header("Predicted House Price")
    st.markdown(f"""
    <div style='padding: 1rem; background-color: #f0f2f6; border-radius: 10px; text-align: center;'>
        <h2 style='color: #1f77b4;'>${predicted_price:,.2f}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Display feature importance if using tree-based model
    if hasattr(model, 'feature_importances_'):
        st.subheader("Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        st.bar_chart(importance_df.set_index('Feature'))

# Add footer
st.markdown("""
---
Created with ❤️ using Streamlit
""")