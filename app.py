
from flask import Flask, request, jsonify
import joblib
import time

app = Flask(__name__)

# Load Trained Model
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return "ML REST API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()

    data = request.get_json()

    features = data['features']

    prediction = model.predict([features])

    response_time = (time.time() - start_time) * 1000

    result = {
        "prediction": int(prediction[0]),
        "response_time_ms": round(response_time, 2)
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
