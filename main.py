from flask import *
import json, time
import tensorflow as tf
import os
import cv2
import urllib.request
import numpy as np
from tensorflow.keras.models import load_model

new_model = load_model(os.path.join('models', 'firefloodmodel.h5'))
user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'

headers={'User-Agent':user_agent,} 

app = Flask(__name__)

# METHOD #1: OpenCV, NumPy, and urllib
def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	# resp = urllib.request.urlopen(url)
    request = urllib.request.Request(url,None,headers)
    response = urllib.request.urlopen(request)
    image = np.asarray(bytearray(response.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	# return the image
    return image

@app.route('/', methods =['GET'])
def home_page():
    img_query = str(request.args.get('image'))
    img = url_to_image(img_query)
    resize = tf.image.resize(img, (256,256))
    yhatnew = new_model.predict(np.expand_dims(resize/255, 0))
    if yhatnew > 0.5:
        data_set ={'Type': 'Flood'}
        
    else:
        data_set ={'Type': 'Fire'}
    json_dump = json.dumps(data_set)
    return json_dump

if __name__ == __name__:
    app.run(port=7777)

