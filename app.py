from flask import  Flask,request,jsonify
import numpy as np
import  pickle

model= pickle.load(open('cv.pkl','rb'))
app=Flask(__name__)

@app.route('/')
def home():
    return "CardioCare-ML"

@app.route('/predict',methods=['POST'])
def predict():
    age = request.form.get('age')
    gender=request.form.get('gender')
    temperature=request.form.get('temperature')
    heart_rate=request.form.get('heart_rate')
    input_query=np.array([[age,gender,temperature,heart_rate]])
    result=model.predict(input_query)[0]
    return jsonify({'Result':str(result)})


if __name__=='__main__':
    app.run(debug=True)