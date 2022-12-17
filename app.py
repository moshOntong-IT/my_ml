from flask import Flask, render_template, Response, request, jsonify
from tensorflow import keras
import flasgger
from flasgger import Swagger
from ast import literal_eval
from utils import *
import joblib
import json
from pathlib import Path
import os
from options import *


app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.debug = True

# Path to the folder where the model is stored
current_dir = os.getcwd()
naive_model_path = os.path.join(
    current_dir, 'models', 'naive_model.joblib')
decision_model_path = os.path.join(
    current_dir, 'models', 'decision_model.joblib')
ann_model_path = os.path.join(
    current_dir, 'models', 'ann.h5')


# A function that convert the binary into a yes or no


def binary_to_yes_no(binary):

    if binary == 1:
        return 'Yes'
    else:
        return 'No'


#  Load the model
# print(decision_model_path)
naive_model = joblib.load(naive_model_path)
decision_model = joblib.load(decision_model_path)
ann_model = keras.models.load_model(ann_model_path)

  #  Create a dictionary of options
options = {
        'education': educationOptions,
        'marital': maritalOptions,
        'occupation': occupationOptions,
        'gender': genderOptions
    }


@app.route("/")
def home():



    return render_template('index.html', options=options)


@ app.route("/test", methods=['POST'])
def test():
    # get the request arguments from the form with the key 'a'
    # a = request.form['a']
    prediction = ""
    convertJson = json.loads(request.form['json'])
    data = convertJson['data']
    categoryData = []
    for i, value in enumerate(data):
        if i == 0:
            category = "education"
        elif i == 1:
            category = "marital"
        elif i == 2:
            category = "occupation"
        elif i == 3:
            category = "gender"
        # print(options[category])
        
        for option in options[category]:
            if option['value'] == str(value):
                categoryData.append(option['label'])

    # categoryData = [options[i]['value'] for i, option in enumerate(data)]
    # print(categoryData)
    classifier = convertJson['classifier']

    #  if the classifier is naive then predict using the naive model
    if classifier == 'naive':
        prediction = naive_model.predict([data])[0]
        prediction = binary_to_yes_no(prediction)
    elif classifier == 'tree':
        prediction = decision_model.predict([data])[0]
        prediction = binary_to_yes_no(prediction)
    elif classifier == 'ann':
        prediction = ann_model.predict([data])[0][0]
        #  if the prediction is greater than 0.5 then then answer is 1 else 0 after that convert it to yes or no
        print(prediction)
        prediction = 1 if prediction > 0.5 else 0
        prediction = binary_to_yes_no(prediction)
        prediction = str(prediction)
        # prediction = round(prediction * 100, 2)
      

    return jsonify({'result': prediction, 'categoryData': categoryData})


if __name__ == '__main__':
    app.run()
