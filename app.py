import streamlit as st
from PIL import Image
import classify
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np


html_temp = """
    <div style="background-color:#e63946 ;padding:10px">
    <h2 style="color:white;text-align:center;">Deep Learning Web App</h2>
    </div><br><br>
    """




st.markdown(html_temp, unsafe_allow_html=True)

st.title("Flower Classification")

st.title(" ")


st.subheader('Upload image of Sunflower,Daisy,Dandelion,Roses  ,Sunflowers  or  Tulips')

st.title(" ")
st.title(" ")



uploaded_file = st.file_uploader(".")
if uploaded_file is not None:

        image = Image.open(uploaded_file)

        lol = load_img(uploaded_file, target_size=(224, 224))
        lol = img_to_array(lol)
        
        lol=lol/225.0
        lol=np.array([lol])
        


        st.image(image, use_column_width=True)

        st.write("")

        if st.button('predict'):


                final_prediction = classify.predict(lol)


                st.success(f'The flower in the picture is most likely  {final_prediction} ')
