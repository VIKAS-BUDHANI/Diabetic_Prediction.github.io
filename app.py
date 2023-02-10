from flask import Flask, request, render_template, flash
import pickle
import numpy as np




app = Flask(__name__,template_folder='templates')

app.config['SECRET_KEY'] = 'supersecret'

scaler = pickle.load(open('scaler.pkl','rb'))
model = pickle.load(open('rf_model.pkl','rb'))


@app.route('/', methods=['GET', 'POST'])


def home():
    prediction = -1 
    if request.method == 'POST':
        preg = int(request.form.get('preg'))
        gluc = int(request.form.get('gluc'))
        bp = int(request.form.get('bp'))
        skin = int(request.form.get('skin'))
        insulin = float(request.form.get('insulin'))
        bmi = float(request.form.get('bmi'))
        func = float(request.form.get('func'))
        age = int(request.form.get('age'))

        #np.array()
        input_features =[[preg, gluc, bp, skin, insulin, bmi, func, age]]
        #print(input_features)
        prediction = model.predict(scaler.transform(input_features))
        #print(prediction)

    return render_template("https://github.com/VIKAS-BUDHANI/Diabetic_Prediction.github.io/blob/7a3af347dee583f681f158db72aaddb8fa6e1ed2/index.html", prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
    
