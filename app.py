



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

  features_name = ['Engine Size(L)', 'Cylinders',
        'Fuel Consumption_City_(L/100 km)',
       'Fuel Consumption Hwy (L/100 km)', 'Fuel Consumption Comb (L/100 km)',
       'Fuel Consumption Comb (mpg)']

  df = pd.DataFrame(features_value, columns=features_name)
  output = model.predict(df)

  if output >= 300:
      res_val = "HIGH CO2 EMISSION"
  else:
      res_val = "LOW CO2 EMISSION"


  return render_template('index.html', prediction_text='THE MODEL HAS {}'.format(res_val))
  

if __name__ == "__main__":
  app.run()


    
