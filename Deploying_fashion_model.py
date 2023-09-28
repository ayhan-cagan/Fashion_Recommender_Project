import streamlit as st 
import os
from PIL import Image
import pandas as pd
import numpy as np 
import tensorflow
import os 
from tqdm import tqdm 
from numpy.linalg import norm
import pickle
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors


feature_list = np.array(pickle.load(open("embeddings.pkl","rb"))).reshape((44441, 2048))

filenames_list = np.array(pickle.load(open("filenames.pkl","rb"))) 


model =ResNet50(weights="imagenet",include_top=False,input_shape=(224,224,3))
model.trainable = False 

model = tensorflow.keras.Sequential([model])
model.add(GlobalMaxPooling2D())


def features_extract(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    image_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(image_array,axis=0)
    preprocess_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocess_img)
    normalized_result = result / norm(result) 
    
    return normalized_result

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6,algorithm="brute",metric="euclidean")

    neighbors.fit(feature_list)  

    distances,indices = neighbors.kneighbors(features) 

    return indices

    
st.title('Fashion Recommender System')


def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0
    
    
uploaded_file = st.file_uploader("Choose an İmage") # dosya yükleme ekranı 

if uploaded_file is not None: 
    if save_uploaded_file(uploaded_file):
        # bir dosya yüklendiğinde bunu ekranda görürüz.
        display_image = Image.open(uploaded_file)
        st.image(display_image) 
        # özellik çıkartma
        features = features_extract(os.path.join("uploads",uploaded_file.name),model)
        st.text(features)  
        #recommendention
        indices = recommend(features,feature_list)
        # show 
        col1,col2,col3,col4,col5 = st.columns(5) 
        # en benzer yakın komşuları yanyana ekranda görmemizi sağlar. 
        with col1:
            st.image(filenames_list[indices[0][0]])
        with col2:
            st.image(filenames_list[indices[0][1]])
        with col3:
            st.image(filenames_list[indices[0][2]])
        with col4:
            st.image(filenames_list[indices[0][3]])
        with col5:
            st.image(filenames_list[indices[0][4]])
            
    
    else:
        st.header("Some error occured in file upload")

