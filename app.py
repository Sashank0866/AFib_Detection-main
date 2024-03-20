from flask_cors import CORS
from flask import Flask, jsonify
from joblib import load
import mysql.connector
import os
import json

app = Flask(__name__)

CORS(app)
# Load the trained RandomForest model
model = load('RandomForest_best_model.pkl')

# Set up MySQL connection parameters
db_config = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME")
}

@app.route('/predict/patient/<patient_id>', methods=['GET'])
def predict_by_patient_id(patient_id):
    try:
        # Connect to the MySQL database
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()

        # Fetch atrial fibrillation data from the atrial_fibrillation table based on patient_id
        cursor.execute(f"SELECT Age, Gender, Channel_1, Channel_2, Channel_3, Channel_4, Channel_5, Channel_6, Channel_7, Channel_8, Channel_9, Channel_10, Channel_11, Channel_12 FROM atrial_fibrillation WHERE patient_id = {patient_id}")
        patient_data = cursor.fetchone()

        if not patient_data:
            return jsonify({'error': 'Patient data not found'}), 404

        age, gender, *channels = patient_data
        # Ensure that 'male' is 1 if Gender is 'Male', else 0, adjusting for your model's requirements
        sex = 1 if gender == 'Male' else 0

        # Initialize features with age and sex in correct order
        features = [age, sex]

        # Extract features from each channel's JSON data
        for channel_json in channels:
            channel_data = json.loads(channel_json)
            features.extend([channel_data['SDNN'][0], channel_data['RMSSD'][0], channel_data['PNN50'][0], channel_data['Mean_RR'][0]])

        # Prediction
        prediction = model.predict([features])[0]
        # Assuming your model's output maps directly to these labels
        prediction_label = {
            0: 'Atrial Fibrillation',
            1: 'Sinus Rhythm',
            2: 'Various Arrhythmias'
        }.get(prediction, 'Unknown')

        return jsonify({
            'patient_id': patient_id,
            'prediction': prediction_label,
            'features': {
                'age': age,
                'sex': gender
            }
        })
    
    except mysql.connector.Error as err:
        return jsonify({'error': f'Database error: {err}'}), 500
    
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()



if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
