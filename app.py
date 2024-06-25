from flask import Flask, request, render_template
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Load the model
model = pickle.load(open('RFR.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the form data
        form_data = request.form

        # Extract features from form data
        age = float(form_data['age'])
        diabetes = float(form_data['diabetes'])
        blood_pressure_problems = float(form_data['blood pressure'])
        any_transplants = float(form_data['transplants'])
        any_chronic_diseases = float(form_data['disease'])
        height = float(form_data['height'])
        weight = float(form_data['weight'])
        known_allergies = float(form_data['allergies'])
        history_of_cancer_in_family = float(form_data['cancer'])
        number_of_major_surgeries = float(form_data['surgeries'])

        # Calculate BMI
        height_m = height / 100.0
        bmi = weight / (height_m ** 2)

        # Prepare the input data for prediction
        input_features = np.array([[age, diabetes, blood_pressure_problems, any_transplants, any_chronic_diseases, weight, known_allergies, history_of_cancer_in_family, number_of_major_surgeries, height_m, bmi]])

        # Predict the premium price
        prediction = model.predict(input_features)[0]

        formatted_prediction = f"{prediction:.2f}"

        return formatted_prediction

    except Exception as e:
        return str(e), 400

if __name__ == '__main__':
    app.run(debug=True)
