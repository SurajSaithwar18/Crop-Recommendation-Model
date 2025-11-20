from flask import Flask, render_template, request
import pickle
import pandas as pd

# Load your best model
model = pickle.load(open("best_model.pkl", "rb"))

# Numerical columns used during training
num_cols = ['N','P','K','temperature','humidity','ph','rainfall']

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Fetch input values
    N = float(request.form['N'])
    P = float(request.form['P'])
    K = float(request.form['K'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])

    # Create input DataFrame
    input_df = pd.DataFrame([[
        N, P, K, temperature, humidity, ph, rainfall
    ]], columns=num_cols)

    # Predict using your best model
    crop = model.predict(input_df)[0]

    return render_template("index.html", crop=crop)

if __name__ == '__main__':
    # change PORT here (e.g., 8000)
    app.run(debug=True, host='0.0.0.0', port=8000)
