from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import tensorflow_hub as hub

import numpy as np
import streamlit as st
import tensorflow as tf

@st.cache(allow_output_mutation=True)
def get_model():
        #model = load_model('./Model/model.h5',custom_objects={'KerasLayer':hub.KerasLayer}
        #)
        model = tf.saved_model.load('Model/lol')
        print('Model Loaded')
        return model 

        
def predict(image):
        model = get_model()
        '''image = load_img(image, target_size=(224, 224))
        image = img_to_array(image)
        st.success(f'The flower in the picture is most likely  {image} ')
        image = image/255.0
        image = np.reshape(image,[1,224,224,3])'''

        #classes = loaded_model.predict(image)
        ans = model(image, training=False)

        flowers_dict_names = {0: 'daisy', 1: 'dandelion'
                , 2: 'roses', 3: 'sunflowers'
                , 4: 'tulips'}

        return flowers_dict_names[np.argmax(ans)]


