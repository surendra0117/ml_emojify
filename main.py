from flask import Flask,render_template,redirect
from flask import request,request,url_for,session, jsonify, send_file
import cv2
import base64

from PIL import Image, ImageTk
import os, sys, io
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator


app = Flask(__name__)

@app.route('/')
def main():
    return render_template('home.html')

@app.route('/api', methods=["GET","POST"])
def api():
    emotion_model = Sequential()

    emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
    emotion_model.add(Dropout(0.25))

    emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
    emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
    emotion_model.add(Dropout(0.25))

    emotion_model.add(Flatten())
    emotion_model.add(Dense(1024, activation='relu'))
    emotion_model.add(Dropout(0.5))
    emotion_model.add(Dense(7, activation='softmax'))
    emotion_model.load_weights('model1.h5')

    cv2.ocl.setUseOpenCL(False)

    emotion_dict = {0: "   Angry   ", 1: "Disgusted", 2: "  Fearful  ", 3: "   Happy   ", 4: "  Neutral  ", 5: "    Sad    ", 6: "Surprised"}


    emoji_dist={0:"static/images/angry.png",1:"static/images/disgusted.png",2:"static/images/fearful.png",3:"static/images/happy.png",4:"static/images/neutral.png",5:"static/images/sad.png",6:"static/images/surprised.png"}

    show_text = [0]
    print("Hello")
    if request.method == "GET":
        return "no picture"
    elif request.method == "POST":
        image_data = request.form.get("content").split(",")[1]
        with open("clientimage.png","wb") as f:
            f.write(base64.b64decode(image_data))
        f.close()
        frame1 = cv2.imread("clientimage.png")
        # print(frame1)
        frame1 = cv2.resize(frame1,(600,500))

        bounding_box = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame1, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            prediction = emotion_model.predict(cropped_img)
            # print(prediction)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame1, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            show_text[0]=maxindex
        print(emoji_dist[show_text[0]])
        return emoji_dist[show_text[0]]




@app.route('/contact')
def contact():
    return render_template('contact.html')


if __name__=='__main__':
    app.run(debug=True)
    