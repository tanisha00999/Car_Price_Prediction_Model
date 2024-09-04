import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load the pre-trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Function to predict car price
def predict_price(year, present_price, kms_driven, fuel_type, seller_type, transmission, owner):
    # Prepare the input data for prediction
    input_data = pd.DataFrame([[year, present_price, kms_driven, fuel_type, seller_type, transmission, owner]], 
                              columns=['Year', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner'])
    # Predict using the loaded model
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit app layout
st.title('Car Price Prediction')

# User inputs
year = st.number_input('Year:', min_value=1900, max_value=2100, step=1)
present_price = st.number_input('Present Price (in lakhs):', format="%.2f")
kms_driven = st.number_input('KMs Driven:')
fuel_type = st.selectbox('Fuel Type:', [0, 1, 2], format_func=lambda x: ['Petrol', 'Diesel', 'CNG'][x])
seller_type = st.selectbox('Seller Type:', [0, 1], format_func=lambda x: ['Dealer', 'Individual'][x])
transmission = st.selectbox('Transmission:', [0, 1], format_func=lambda x: ['Manual', 'Automatic'][x])
owner = st.number_input('Number of Previous Owners:')

# Predict button
if st.button('Predict'):
    price = predict_price(year, present_price, kms_driven, fuel_type, seller_type, transmission, owner)
    st.write(f'Predicted Price: ₹{price:.2f} lakhs')

