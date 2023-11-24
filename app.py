from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from unique_car_names import car_names
app = Flask(__name__)

# Load the trained model
MODEL = joblib.load('car_price_prediction_model.pkl')



# Label Encoding for car names
label_encoder = LabelEncoder()

# # Fit the LabelEncoder on the car names
# label_encoder.fit(car_names)
# @app.route('/')
# def home():
#     # Pass the list of car names to the template
#     return render_template('index.html', car_names=car_names)

@app.route('/')
def home():
    # Pass the list of car names to the template
    return render_template('index.html', car_names=car_names)

@app.route('/predict', methods=['POST'])
def predict():
    # Get input from the form
    input_data = [request.form['Car_Name'], int(request.form['Year']), float(request.form['Present_Price']),
                  float(request.form['Driven_kms']), request.form['Fuel_Type'], request.form['Selling_type'],
                  request.form['Transmission'], int(request.form['Owner'])]

    current_year = datetime.now().year
    # Add the difference between the current year and the year of the car 
    input_data[1] = current_year - input_data[1]

    # Convert categorical features to numerical using Label Encoding
    input_data[0] = label_encoder.fit_transform([input_data[0]])[0]
    input_data[4] = label_encoder.fit_transform([input_data[4]])[0]
    input_data[5] = label_encoder.fit_transform([input_data[5]])[0]
    input_data[6] = label_encoder.fit_transform([input_data[6]])[0]

    # Make a prediction
    prediction = MODEL.predict([input_data])[0]

    return render_template('index.html', car_names=car_names, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
