# flask code(app.py)
import pickle
import numpy as np
from flask import Flask, request, render_template

model = pickle.load(open('model.pkl' , 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    output = prediction[0]
    if output == 1:
        return render_template('index.html', prediction_text='likely Diabetic. visit https://shorturl.at/csDP6 for more info')
    else:
        return render_template('index.html', prediction_text='Not Diabetic. visit https://shorturl.at/fuyHN for more info')

if __name__ == '__main__':
    app.run(debug=True)