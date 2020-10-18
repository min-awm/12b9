from keras.models import load_model
import cv2
import numpy as np
import base64
from flask import Flask, request,render_template
from flask_ngrok import run_with_ngrok


model = load_model('chinh.h5')

app = Flask(__name__)
run_with_ngrok(app)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uri = request.form['data']
        encoded_data = uri.split(',')[1]
        nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        resized_image = np.array([cv2.resize(img, (64, 64))])
        y = model.predict(resized_image)
        print(y)
        if(y >= 0.7):
            return "1"
        else:
            return "0"
        

    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run()
