from flask import Flask, render_template, request
from pymongo import MongoClient
import pandas as pd
from joblib import load

app = Flask(__name__)

HOST = 'cluster0.dmerp.mongodb.net'
USER = 'hsj8518'
PASSWORD = 'whale1234'
DATABASE_NAME = 'myFirstDatabase'
COLLECTION_NAME = 'daeyeoso'
MONGO_URI = f"mongodb+srv://{USER}:{PASSWORD}@{HOST}/{DATABASE_NAME}?retryWrites=true&w=majority"
client = MongoClient(MONGO_URI)
database = client[DATABASE_NAME]
collection = database[COLLECTION_NAME]

@app.route('/')
def index():
    return render_template('index.html')
        

@app.route('/predict/', methods=['POST'])
def predict():
    add = request.form['대여소번호']
    sigan = request.form['시간대']

    cur = list(collection.find({"대여소번호":int(add)}))
    pipe = load('model.joblib')
    
    pred_x = pd.DataFrame({'대여소번호':[add], '시간대':[sigan], '자치구':[cur[0]['자치구']], '위도':[cur[0]['위도']], '경도':[cur[0]['경도']], '운영방식':[cur[0]['운영방식']], '총운영대수':[cur[0]['총운영대수']]})
    pred_x = pred_x.set_index('대여소번호')
    pred_y = pipe.predict(pred_x)[0]
    return render_template('predict.html', pred = int(pred_y))
        
if __name__ == "__main__":
    app.run(debug=True)