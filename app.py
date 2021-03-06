import pickle
import os

from flask import Flask, request, jsonify

app = Flask(__name__)

try:
    modelpath = os.path.join(os.getcwd(),'Model/PythonSalaryPredictionModel')
    with open(modelpath, 'rb') as f:
        model = pickle.load(f)
    print(modelpath,'This was success')
except Exception as e:
    print(str(e))

@app.route('/hello')
def hello():
    return 'hi'


@app.route('/SalaryPredict', methods=['GET', 'POST'])
def SalaryPredict():
    exp_input = request.json['salary']
    salary = model.predict([[exp_input]])
    return jsonify({'results': salary[0]})


if __name__ == "__main__":
    app.run()
