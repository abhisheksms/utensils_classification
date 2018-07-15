from keras.models import load_model,model_from_json
from flask import Flask
from flask import request
from flask import jsonify

import base64
import io
from PIL import Image
import keras
from keras import backend as K 
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator,img_to_array


import numpy as np
import tensorflow as tf

#INSTRUCTIONS
#set FLASK_APP=utensil_app.py
#flask run --host=0.0.0.0


app = Flask(__name__)

def get_model():

	global model,graph
	model=load_model('utensil_model.h5')
	print("Model loaded")
	graph = tf.get_default_graph()


def preprocess_image(image,target_size):

	if image.mode != "RGB":
		image=image.convert("RGB")

	image=image.resize(target_size)
	image=img_to_array(image)
	image=np.expand_dims(image,axis=0)

	return image


print("Loading keras model")
get_model()

@app.route('/predict',methods=["POST"])
def predict():
	message=request.get_json(force=True)
	encoded=message['image']
	decoded=base64.b64decode(encoded)


	with graph.as_default():
		
		image=Image.open(io.BytesIO(decoded))
		processed_image=preprocess_image(image,target_size=(224,224))
		prediction=model.predict(processed_image).tolist()

			
		response={
		'prediction':{
		'bottle':prediction[0][0],
		'fork':prediction[0][1],
		'plate':prediction[0][2],
		'spoon':prediction[0][3]
		}
		}


		return jsonify(response)
		

