from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Create Flask app
app = Flask(__name__)

# Load model and label encoder
model = joblib.load('decision_tree_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data JSON dari request
        data = request.get_json()
        warna_daun_encode = data['warna_daun']
        warna_bercak_encode = data['warna_bercak']
        bentuk_bercak_encode = data['bentuk_bercak']

        # Buat DataFrame dari data input
        input_df = pd.DataFrame({
            'warna_daun_encode': [warna_daun_encode],
            'warna_bercak_encode': [warna_bercak_encode],
            'bentuk_bercak_encode': [bentuk_bercak_encode]
        })

        # Prediksi menggunakan model
        prediction = model.predict(input_df)
        
        # Ubah hasil prediksi menjadi string
        result = str(prediction[0])

        return jsonify({'prediction': result}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)