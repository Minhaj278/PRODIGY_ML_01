import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("house_price_model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit app title
st.title("House Price Prediction App")

# Input fields for user data
gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", min_value=0, value=1500)
bedrooms = st.number_input("Number of Bedrooms", min_value=0, value=3)
full_baths = st.number_input("Number of Full Bathrooms", min_value=0, value=2)

# Prediction button
if st.button("Predict Price"):
    # Prepare input data for prediction
    input_data = np.array([[gr_liv_area, bedrooms, full_baths]])
    
    # Make prediction
    predicted_price = model.predict(input_data)
    
    # Display the result
    st.success(f"The predicted house price is: ${predicted_price[0]:,.2f}")
