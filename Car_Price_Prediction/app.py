from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load your trained Linear Regression model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from the form
        year = int(request.form['year'])
        present_price = float(request.form['present_price'])
        kms_driven = float(request.form['kms_driven'])
        fuel_type = int(request.form['fuel_type'])
        seller_type = int(request.form['seller_type'])
        transmission = int(request.form['transmission'])
        owner = int(request.form['owner'])

        # Arrange inputs in the order expected by the model
        features = np.array([year, present_price, kms_driven, fuel_type, seller_type, transmission, owner]).reshape(1, -1)

        # Make prediction using Linear Regression
        linear_pred = model.predict(features)

        # Render the result
        return render_template('index.html', linear_pred=f"Rs{linear_pred[0]:.2f} lakhs")

if __name__ == '__main__':
    app.run(debug=True)
