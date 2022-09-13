from fastapi import FastAPI

app = FastAPI()





@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/train")
async def train(json_path: str):
    import face_recognition
    import numpy as np
    import urllib.request
    import os
    import pathlib
    from PIL import Image
    import requests
    from io import BytesIO
    import pandas as pd
    import pyrebase
    firebaseConfig = {
        'apiKey': "AIzaSyDrdZvpXaHRUujP-jWBrd02643CqzAZ1Y0",
        'authDomain': "ezbill-1.firebaseapp.com",
        'databaseURL': "https://ezbill-1-default-rtdb.firebaseio.com",
        'projectId': "ezbill-1",
        'storageBucket': "ezbill-1.appspot.com",
        'messagingSenderId': "1018480204152",
        'appId': "1:1018480204152:web:e33488e0e6f36bf9f2d08d",
        'measurementId': "G-JMQ6T5NT9E"
        }
    firebase = pyrebase.initialize_app(firebaseConfig)
    db= firebase.database()
    auth = firebase.auth()
    storage = firebase.storage()
    import urllib.request
    import json
    jsonURL = storage.child(json_path).get_url(None)
    with urllib.request.urlopen(jsonURL) as url:
        data = json.load(url)
    # read image file from url
    def imgFromUrl(url):
        response = requests.get(url)
        return BytesIO(response.content)

    known_face_encodings_1 = []
    known_face_encodings_2 = []
    known_face_names = []

    def new_face(name,url_1,url_2):
        known_face_names.append(name)
        img_1 = face_recognition.load_image_file(imgFromUrl(url_1))
        img_2 = face_recognition.load_image_file(imgFromUrl(url_2))
        known_face_encodings_1.append(face_recognition.face_encodings(img_1)[0])
        known_face_encodings_2.append(face_recognition.face_encodings(img_2)[0])

    for tenant in data['tenants']:new_face(tenant['user_ID'],storage.child(tenant['face1']).get_url(None), storage.child(tenant['face2']).get_url(None))
    import pickle

    with open("face_id.pickle", "wb") as face_id:
        pickle.dump(known_face_names, face_id)
    with open("face_encodings_1.pickle", "wb") as face1:
        pickle.dump(known_face_encodings_1, face1)
    with open("face_encodings_2.pickle", "wb") as face2:
        pickle.dump(known_face_encodings_2, face2)   
    
    storage.child("data/face_id.pickle").put("face_id.pickle")
    storage.child("data/face_encodings_1.pickle").put("face_encodings_1.pickle")
    storage.child("data/face_encodings_2.pickle").put("face_encodings_2.pickle")
    
    return {"message:": "Training Completed"}

