from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model, label encoder, and column transformer
model = pickle.load(open('trained_model.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))  # Save your label encoder separately
ct = pickle.load(open('column_transformer.pkl', 'rb'))  # Save your column transformer separately

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = {
        'model': request.form['model'],
        'vehicle_age': int(request.form['vehicle_age']),
        'km_driven': int(request.form['km_driven']),
        'seller_type': request.form['seller_type'],
        'fuel_type': request.form['fuel_type'],
        'transmission_type': request.form['transmission_type'],
        'mileage': float(request.form['mileage']),
        'engine': float(request.form['engine']),
        'max_power': float(request.form['max_power']),
        'seats': int(request.form['seats'])
    }

    df = pd.DataFrame([data])
    df['model'] = le.transform(df['model'])  # encode model
    X_input = pd.DataFrame(ct.transform(df), columns=ct.get_feature_names_out())
    prediction = model.predict(X_input)[0]

    return render_template('index.html', prediction_text=f"Estimated Price: ₹{int(prediction):,}")

if __name__ == '__main__':
    app.run(debug=True)
