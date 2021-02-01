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
  output = model.predict(df)

  if output == 4:
      res_val = "Breast cancer"
  else:
      res_val = "no Breast cancer"


  return render_template('index.html', prediction_text='Patient has {}'.format(res_val))

if __name__ == "__main__":
  app.run()
