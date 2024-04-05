from flask_cors import CORS
from flask import Flask, jsonify
from joblib import load
from dotenv import load_dotenv
import mysql.connector
import os
import json
import logging


app = Flask(__name__)


# Load the trained RandomForest model
model = load('RandomForest_best_model.pkl')

logger = logging.getLogger(__name__)


###Check the Databse Configuration

load_dotenv()

# Set up MySQL connection parameters
db_config = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME")
}

@app.route('/<patient_id>', methods=['GET'])
def home_page(patient_id):
    return f"Hello World {patient_id}"
    
@app.route('/test_db_connection')
def test_db_connection():
    cursor = None
    connection = None

    try:
        # Connect to the PostgreSQL database
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()

        # Perform a simple query to check the database connection
        cursor.execute("SELECT count(*) FROM atrial_fibrillation;")
        count = cursor.fetchone()[0]

        return jsonify({'message': f'Database connection successful. Found {count} records in the table'}), 200

    except mysql.connector.Error as e:
        return jsonify({'error': f'Database error: {e}'}), 500

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


@app.route('/predict/patient/<patient_id>', methods=['GET'])
def predict_by_patient_id(patient_id):
    cursor = None

    try:
        logger.debug('Start of prediction process')
        # Connect to the MySQL database
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()

        # Fetch atrial fibrillation data from the atrial_fibrillation table based on patient_id
        cursor.execute(f"SELECT Age, Gender, Channel_1, Channel_2, Channel_3, Channel_4, Channel_5, Channel_6, Channel_7, Channel_8, Channel_9, Channel_10, Channel_11, Channel_12 FROM atrial_fibrillation WHERE patient_id = {patient_id}")
        patient_data = cursor.fetchone()
        logging.debug('Cursor ',patient_data)

        if not patient_data:
            return jsonify({'error': 'Patient data not found'}), 404

        age, gender, *channels = patient_data
        # Ensure that 'male' is 1 if Gender is 'Male', else 0, adjusting for your model's requirements
        sex = 1 if gender == 'Male' else 0

        # Initialize features with age and sex in correct order
        features = [age, sex]

        # Extract features from each channel's JSON data
        for channel_json in channels:
            # New approach, check type first:
            if isinstance(channel_json, str):
                channel_data = json.loads(channel_json)  # Decode JSON string to dict
            else:
                channel_data = channel_json  # Use directly if already a dict
            features.extend([channel_data['SDNN'][0], channel_data['RMSSD'][0], channel_data['PNN50'][0], channel_data['Mean_RR'][0]])

        # Prediction
        logger.debug('Stating pred')
        prediction = model.predict([features])[0]
        # Assuming your model's output maps directly to these labels
        logger.debug('Finish pred')
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
        logger.debug('Found an error')
        return jsonify({'error': f'Database error: {err}'}), 500
    
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

CORS(app, origins=[
    'http://localhost:3000'
])

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)