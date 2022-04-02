from flask import Flask, request, jsonify
import pandas as pd
from flask import jsonify
import os
from model_pred import load_model,preprocess_vedio
import numpy as np
import time

app=Flask(__name__)

actions = ['0', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9', 'Ain',
'Alf', 'Alslam-Aliukom', 'Arabic-Language', 'Ba', 'Brother',
        'Cold', 'Come-in', 'Daah', 'Dal', 'Deaf', 'Death', 'Doctor',
        'English-Language', 'Evening', 'Fa', 'Family', 'Father', 'Feel',
        'File', 'Gim', 'Gin', 'Ha', 'Haa', 'Hello', 'Hospital', 'Hot',
'How-are-you', 'Job', 'Kaf', 'Kha', 'King', 'Lam', 'Manager',
'Medication', 'Meeting', 'Mem', 'Morning', 'Mosque', 'Mother',
'Name', 'Non', 'Pain', 'Pharmacy', 'Prayer', 'Qaf', 'Ra', 'Reason',
'Sad', 'Saud', 'Shin', 'Sign-Language', 'Sin', 'Sister', 'Sorry',
'Surgery', 'Ta', 'Taah', 'Tha', 'Thad', 'Thal', 'Thank', 'Tired',
'University', 'Vacation', 'Waw', 'Where', 'Ya', 'Zai']

best = load_model('final.h5')

Upload = 'static/upload'

if not os.path.exists(Upload):
    os.makedirs(Upload)
app.config['uploadFolder'] = Upload

@app.route("/pred", methods=[ 'POST'])

def upload_file():

    if request.method == 'POST':
        print(request.files['file'])
        f = request.files['file']
        filename = f.filename
        start_time = time.time()
        f.save(os.path.join(app.config['uploadFolder'], f.filename))
        print(f.filename)
        ready_vedio = preprocess_vedio("static/upload/"+filename)

        print(ready_vedio.shape)
        yhat = best.predict(ready_vedio)
        ytrue = np.argmax(yhat, axis=1).tolist()

        label_num_str_map = {num:label for num, label in enumerate(actions)}
        label_num_str_map[ytrue[0]]
        print(label_num_str_map[ytrue[0]])

        print("--- %s seconds ---" % (time.time() - start_time))
        os.remove("static/upload/"+filename)
        d = {
        "prediction":label_num_str_map[ytrue[0]],
        "number":ytrue[0]
        }
        return jsonify(d)



if __name__ == "__main__":
    app.run()
