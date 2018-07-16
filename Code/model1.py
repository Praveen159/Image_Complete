# -*- coding: utf-8 -*-
from flask import Flask,render_template,request
from keras.preprocessing.image import  img_to_array, load_img
from keras.models import load_model

app=Flask(__name__)
img_width, img_height = 256,256
@app.route('/assessment')
def input():
    return render_template("image_input.html")

@app.route('/imageoutput',methods=['POST'])
def image_class():
    if request.method=='POST':
        model1=load_model('C:/Users/hi/POC/Fine_Tune/ft_model.h5')
        img = load_img('save.jpg', target_size=(img_width, img_height)) 
        x = img_to_array(img) 
        x = x.reshape((1,) + x.shape)/255 
        pred = model1.predict(x)
        if pred[0][0]<0.5:
            a="CAR WITH DENT"
        else:
            a="CAR WITH NO DENT"
            
        return render_template("image_output.html",output=a)
    
if __name__=='__main__':
    app.run(port=5000)