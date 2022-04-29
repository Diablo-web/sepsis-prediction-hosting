import numpy as np
import pandas as pd
import os
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('xgboost.pkl', 'rb'))
labtestmodel = pickle.load(open('xgboostSepsis.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/rapidtest')
def rapid():
    return render_template('rapid.html')

@app.route('/adultstest')
def adultstest():
    return render_template('adultstest.html')

@app.route('/childtest')
def childtest():
    return render_template('childtest.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')

@app.route('/hiw')
def hiw():
    return render_template('hiw.html')
    
@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/about')
def about():
    return render_template('aboutsepsis.html')

@app.route('/advancedtest')
def advanced():
    return render_template('advanced.html')

@app.route('/labtest')
def labtest():
    return render_template('labtest.html')

@app.route('/map')
def map():
    return render_template('map.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = [ "HR","O2Sat","Temp","SBP","MAP","DBP","Resp","EtCO2"]
    
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
        
    if output == 1:
        res_val = " may be diagnosed with sepsis and we advice you to visit nearest medical facilities."
    else:
        res_val = "is safe "
        

    return render_template('predict.html', prediction_text='The patient {}'.format(res_val))


@app.route('/predictlab',methods=['POST'])
def predictlab():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]

    features_name = [ "HR","O2Sat","Temp","SBP","MAP","DBP","Resp","EtCO2"]

    df = pd.DataFrame(features_value, columns=features_name)
   
    output = model.predict(df)
        
    if output == 1:
        res_val = " has sepsis "
    else:
        res_val = "is safe "
        

    return render_template('labtest.html', prediction_text='The patient {}'.format(res_val))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)