from flask import Flask, render_template, Response, request,jsonify
import flasgger
from flasgger import Swagger
from ast import literal_eval
from utils import *
import joblib
import json
from pathlib import Path
import os


app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Path to the folder where the model is stored
current_dir = Path().absolute()
naive_model_path = os.path.join(current_dir, 'models\\naive_model.joblib')
decision_model_path = os.path.join(current_dir, 'models\\decision_model.joblib')

# A function that convert the binary into a yes or no
def binary_to_yes_no(binary):
    if binary == 1:
        return 'Yes'
    else:
        return 'No'

#  Load the model
naive_model = joblib.load(naive_model_path)
decision_model = joblib.load(decision_model_path)

@app.route("/")
def home():
    
    return render_template('index.html')

@app.route("/test",methods = ['POST'])
def test():
    # get the request arguments from the form with the key 'a'
    # a = request.form['a']
    prediction = ""
    convertJson = json.loads(request.form['json'])
    data = convertJson['data']
    classifier = convertJson['classifier']

    #  if the classifier is naive then predict using the naive model
    if classifier == 'naive':
        prediction = naive_model.predict([data])[0]
    else:
        prediction = decision_model.predict([data])[0]

    convert = binary_to_yes_no(prediction)
    
    return jsonify({'result': convert})

if __name__ == '__main__':
    app.run(debug=True)
