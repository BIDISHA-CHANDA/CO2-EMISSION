import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('co2_final.pkl', 'rb'))

@app.route('/')
def home():
  return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
  input_features = [int(x) for x in request.form.values()]
  features_value = [np.array(input_features)]

  features_name = ['Engine_Size(L)', 'Cylinders',
        'Fuel_Consumption_City_(L/100 km)',
       'Fuel_Consumption_Hwy_(L/100 km)', 'Fuel_Consumption_Comb_(L/100 km)',
       'Fuel_Consumption_Comb_(mpg)']

  df = pd.DataFrame(features_value, columns=features_name)
  prediction=model.predict_proba(df)
  output='{0:.{1}f}'.format(prediction[0][1], 2)

    if output>str(0.5):
        return render_template('index.html',pred='Your Forest is in Danger.\nProbability of fire occuring is {}'.format(output),bhai="kuch karna hain iska ab?")
    else:
        return render_template('index.html',pred='Your Forest is safe.\n Probability of fire occuring is {}'.format(output),bhai="Your Forest is Safe for now")



if __name__ == "__main__":
  app.run()
