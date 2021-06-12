from flask import Flask,render_template,request,url_for
import os
import pickle

import numpy as np
import pandas as pd
import scipy
import sklearn

import skimage
import skimage.color
import skimage.transform
import skimage.io
import skimage.feature


app = Flask(__name__)

# -------------------CONSTANTS ----------------

BASEPATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASEPATH,'static/uploads/')


MODEL_PATH = os.path.join(BASEPATH,'static/models/')


# load models
model_sgd_path = os.path.join(MODEL_PATH,'image_classification_sgd.pickle')
scaler_path = os.path.join(MODEL_PATH,'dsa_scaler.pickle')

model_sgd = pickle.load(open(model_sgd_path,'rb'))
scaler = pickle.load(open(scaler_path,'rb'))


@app.errorhandler(404)
def error404(err):
   msg = 'page not found'
   return render_template('error.html', message=msg, error=err)

@app.errorhandler(405)
def error405(err):
   msg='method not found'
   return render_template('error.html',message=msg, error=err)

@app.errorhandler(500)
def error500(err):
   msg='internal error due to incorrect logic'
   return render_template('error.html',message=msg,error=err)

@app.route('/about')
def about():
   return render_template('about.html')


@app.route('/',methods=['POST','GET'])
def index():
   if request.method=='POST':

      upload_file = request.files['my_image']
      filename = upload_file.filename
      print(filename)

      ## get the extension of filename , allow only (jpg,jpeg,png)

      ext = filename.split('.')[-1]
      print('extension of filename is',ext)
      if ext.lower() in ['jpg','png','jpeg']:
         path_to_save = os.path.join(UPLOAD_PATH,filename)
         upload_file.save(path_to_save)
         print('filed uploaded')

         height,width = get_height(path_to_save,200)

         # send the file to pipeline model

         results = pipe_fun(path_to_save,scaler,model_sgd)
         print(results)

      else:
         print('use only the extension with .jpg,.png,.jpeg')
         return render_template('upload.html',fileupload=False, extension=True)


      return render_template('upload.html',fileupload=True,data=results,image_name=filename,height=height, width=width,extension=False)

   
   else:
      return render_template('upload.html',fileupload=False, extension=False)

   
def get_height(path,given_width):
   image = skimage.io.imread(path)
   h,w,_ = image.shape
   # given_width = 300
   aspect = h/w
   height = given_width*aspect
   return height,given_width



def pipe_fun(path,scaler_transformed,model_sgd):
   # read the image
   image = skimage.io.imread(path)
   # making transformation
   image_resized = skimage.transform.resize(image,(80,80))
   # rescaling
   rescaled_image = 255*image_resized
   image_transformed = rescaled_image.astype(np.uint8) # converting to 8 bit integer
   # graify
   gray = skimage.color.rgb2gray(image_transformed) # can use custom function as well
   # hog feature extraction
   hog_feature_vector = skimage.feature.hog(gray,
                                 orientations=10, pixels_per_cell=(8,8),cells_per_block=(3,3))
   
   #scaling
   scaled = scaler_transformed.transform(hog_feature_vector.reshape(1,-1))
   y_pred = model_sgd.predict(scaled)
   # confidence score for each class
   decision_value = model_sgd.decision_function(scaled)
   decision_value=decision_value.flatten()
   labels = model_sgd.classes_
   # probabilty 
   z = scipy.stats.zscore(decision_value)
   prob = scipy.special.softmax(z)
   top_5_prob_index = prob.argsort()[::-1][:5]
   top_5_prob_values = prob[top_5_prob_index]
   top_labels = labels[top_5_prob_index]
   #making dictionary
   top_dict = dict()
   for key,value in zip(top_labels,top_5_prob_values):
      top_dict[key]=np.round(value,2)
   
   return top_dict



if __name__=="__main__":
    app.run(debug=True)