from flask import Flask, jsonify, request
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import joblib

app = Flask(__name__)

# Sample in-memory data
data = pd.read_csv('data/real_estate_data.csv')
X = data.drop(columns=['price'])
y = data['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = GradientBoostingRegressor()
model.fit(X_train, y_train)
joblib.dump(model, "models/property_valuation_model.pkl")

@app.route('/predict', methods=['POST'])
def predict_property_value():
    model = joblib.load("models/property_valuation_model.pkl")
    features = request.json['features']
    prediction = model.predict([features])
    return jsonify({'predicted_price': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
