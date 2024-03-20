from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS  # Import CORS


app = Flask(__name__)

CORS(app)

# Load the pre-trained model
model_path = 'RandomForest_best_model.pkl'  #
model = joblib.load(model_path)

prediction_labels = {
    0: 'Atrial Fibrillation',
    1: 'Sinus Rhythm',
    2: 'Various Arrhythmias'
}

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.get_json()  #Get data as a JSON format
            #Defining Feature Order Explicitly
            features_order = ['Age', 'Sex','SDNN_Channel_1','RMSSD_Channel_1', 'PNN50_Channel_1', 'Mean_RR_Channel_1',
       'SDNN_Channel_2', 'RMSSD_Channel_2', 'PNN50_Channel_2',
       'Mean_RR_Channel_2', 'SDNN_Channel_3', 'RMSSD_Channel_3',
       'PNN50_Channel_3', 'Mean_RR_Channel_3', 'SDNN_Channel_4',
       'RMSSD_Channel_4', 'PNN50_Channel_4', 'Mean_RR_Channel_4',
       'SDNN_Channel_5', 'RMSSD_Channel_5', 'PNN50_Channel_5',
       'Mean_RR_Channel_5', 'SDNN_Channel_6', 'RMSSD_Channel_6',
       'PNN50_Channel_6', 'Mean_RR_Channel_6', 'SDNN_Channel_7',
       'RMSSD_Channel_7', 'PNN50_Channel_7', 'Mean_RR_Channel_7',
       'SDNN_Channel_8', 'RMSSD_Channel_8', 'PNN50_Channel_8',
       'Mean_RR_Channel_8', 'SDNN_Channel_9', 'RMSSD_Channel_9',
       'PNN50_Channel_9', 'Mean_RR_Channel_9', 'SDNN_Channel_10',
       'RMSSD_Channel_10', 'PNN50_Channel_10', 'Mean_RR_Channel_10',
       'SDNN_Channel_11', 'RMSSD_Channel_11', 'PNN50_Channel_11',
       'Mean_RR_Channel_11', 'SDNN_Channel_12', 'RMSSD_Channel_12',
       'PNN50_Channel_12', 'Mean_RR_Channel_12']  
            features = [data[feature] for feature in features_order]
            prediction = model.predict([features])[0]  
            prediction_label = prediction_labels.get(prediction, 'Unknown')

            return jsonify({'Prediction': prediction_label})
        except Exception as e:
            return jsonify({'Error': str(e), 'message': 'Error processing request'})

if __name__ == '__main__':
    app.run(debug=False) 
