from flask import Flask
from flask import request
import pickle
import pandas as pd
import json

app = Flask(__name__)

loaded_model = open('/Users/alex/Downloads/trained_model.pkl', 'rb')
server_model = pickle.load(loaded_model)


@app.route('/')
def home():
    return '<h1>Your Server</h1>'


@app.route('/pred')
def get_prediction():
    X = pd.DataFrame(columns=['CRIM', 'ZN', 'CHAS', 'RM', 'PTRATIO', 'B', 'LSTAT'])
    
    for col_name in (X.columns):
      X.loc[0, col_name] = request.args.get(col_name)
      X[col_name].fillna(X[col_name].mean(), inplace=True)
    
    # predict
    prediction = server_model.predict(X)

    output = f'prediction: {prediction[0]}'

    return f'prediction: {prediction[0]}'


@app.route("/mul_pred", methods=["POST"])
def get_multiple_predictions():
    
    # read json
    param_dict = request.get_json()
    
    # convert json to dict
    param_dict = json.loads(param_dict)
    X = pd.DataFrame(columns=['CRIM', 'ZN', 'CHAS', 'RM', 'PTRATIO', 'B', 'LSTAT'])

    for sample in range(len(param_dict["CRIM"])):
        for col in X.columns:
            # populate the df
            try:
                # if we got value via json
                X.loc[sample,col]=param_dict[col][sample]
            except IndexError:
                # if we did not get value via json, fill with mean value
                X.loc[sample,col]=0
    # predict
    prediction = server_model.predict(X)

    # convert to json
    pred_json = json.dumps(prediction.tolist())
    
    return pred_json

if __name__ == '__main__':
    app.run()    