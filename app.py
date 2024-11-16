from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import json
import urllib.request

app = Flask(__name__)

# Load models for each module
fertilizer_model = pickle.load(open('classifier.pkl', 'rb'))
fertilizer_info = pickle.load(open('fertilizer.pkl', 'rb'))
crop_model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

weather_api_key = 'b2c0725fd6de9dc60d136379ce423603'  # OpenWeatherMap API Key

# Home Page Route
@app.route('/')
def home():
    return render_template('index.html')

# Fertilizer Recommendation Route
@app.route('/fertilizer')
def fertilizer_page():
    return render_template('Model1.html')

@app.route('/fertilizer/predict', methods=['POST'])
def fertilizer_predict():
    try:
        temp = request.form.get('temp')
        humi = request.form.get('humid')
        mois = request.form.get('mois')
        soil = request.form.get('soil')
        crop = request.form.get('crop')
        nitro = request.form.get('nitro')
        pota = request.form.get('pota')
        phosp = request.form.get('phos')

        # Ensure all input fields are present and numeric
        if None in (temp, humi, mois, soil, crop, nitro, pota, phosp) or not all(val.isdigit() for val in (temp, humi, mois, soil, crop, nitro, pota, phosp)):
            return render_template('Model1.html', x='Invalid input. Please provide numeric values for all fields.')

        # Convert values to integers
        input_data = [int(temp), int(humi), int(mois), int(soil), int(crop), int(nitro), int(pota), int(phosp)]

        # Predict the fertilizer class
        prediction_idx = fertilizer_model.predict([input_data])[0]

        # Retrieve the label using classes_
        result_label = fertilizer_info.classes_[prediction_idx] if hasattr(fertilizer_info, 'classes_') else 'Unknown'

        return render_template('Model1.html', x=result_label)

    except Exception as e:
        return render_template('Model1.html', x=f"Error in prediction: {str(e)}")

# Weather Forecast Route
@app.route('/weather', methods=['GET', 'POST'])
def weather_page():
    data = {}
    forecast_data = {}
    lat = request.args.get('lat')
    lon = request.args.get('lon')

    if lat and lon:
        try:
            # Current Weather API call
            weather_url = f'http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={weather_api_key}'
            weather_source = urllib.request.urlopen(weather_url).read()
            weather_info = json.loads(weather_source)

            data = {
                "country_code": weather_info['sys']['country'],
                "coordinate": f"{lat} {lon}",
                "temp": f"{round(float(weather_info['main']['temp']) - 273.15, 2)}°C",
                "pressure": f"{weather_info['main']['pressure']} hPa",
                "humidity": f"{weather_info['main']['humidity']}%",
            }

            # 5-day Forecast API call
            forecast_url = f'http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={weather_api_key}'
            forecast_source = urllib.request.urlopen(forecast_url).read()
            forecast_info = json.loads(forecast_source)

            forecast_data = []
            for forecast in forecast_info['list']:
                forecast_data.append({
                    "datetime": forecast['dt_txt'],
                    "temp": round(float(forecast['main']['temp']) - 273.15, 2),
                    "pressure": forecast['main']['pressure'],
                    "humidity": forecast['main']['humidity'],
                })

        except Exception as e:
            data = {"error": f"Could not retrieve data: {str(e)}"}

    return render_template('weather.html', data=data, forecast_data=forecast_data)

# Crop Recommendation Route
@app.route('/crop')
def crop_page():
    return render_template("crop.html")

@app.route('/crop/predict', methods=['POST'])
def crop_predict():
    try:
        # Collect data from form
        N, P, K, temp, humidity, ph, rainfall = [request.form.get(field) for field in ('Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall')]

        # Convert and validate the inputs
        feature_list = [int(N), int(P), int(K), float(temp), float(humidity), float(ph), float(rainfall)]
        single_pred = np.array(feature_list).reshape(1, -1)

        # Scaling and prediction
        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)
        prediction = crop_model.predict(final_features)

        # Define crop mapping
        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya",
            7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes",
            12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil", 16: "Blackgram",
            17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas", 20: "Kidneybeans",
            21: "Chickpea", 22: "Coffee"
        }

        result = f"{crop_dict.get(prediction[0], 'Unknown crop')} is the best crop to be cultivated."

        # Return the result to the template
        return render_template('crop.html', result=result)

    except Exception as e:
        return render_template('crop.html', result=f"Error: {str(e)}")

import os

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))  # Get port from environment or default to 5000
    app.run(host='0.0.0.0', port=port)

