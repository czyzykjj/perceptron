from flask import Flask, request, jsonify
import numpy as np
from sklearn.linear_model import Perceptron

app = Flask(__name__)

# Example Perceptron model
model = Perceptron()
# Train with some example data
X = np.array([[0, 0], [1, 1]])
y = np.array([0, 1]])
model.fit(X, y)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict([data['input']])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
