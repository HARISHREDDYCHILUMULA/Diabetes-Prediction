import pickle
import flask 
from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

model = pickle.load(open('modelpkl.pkl', 'rb'))
scaler = pickle.load(open('scalerpkl.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    app.logger.debug(f"Received data: {data}")
    new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
    output = model.predict(new_data)
    app.logger.debug(f"Output: {output[0]}")
    return jsonify({'prediction': int(output[0])})

@app.route('/predict',methods=['POST'])
def predict():
    data = []
    for value in request.form.values():
        if value:
            data.append(float(value))
    final_input=scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=model.predict(final_input)[0]
    strp="Person is suffering from Diabetes"
    strn="Person is not suffering from Diabetes"
    if output==1:
        str=strp
    else :
        str=strn
    return render_template("index.html",prediction_text=str)


if __name__ == '__main__':
    app.run(debug=True)
