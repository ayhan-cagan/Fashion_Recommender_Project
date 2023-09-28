import pickle
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import cv2

feature_list = np.array(pickle.load(open("embeddings.pkl","rb"))) # fashion_model.py üzerinde kaydettiğimiz dosyaları yüklüyoruz. 

#print(feature_list.shape) #(44441,1,2048)

reshaped_feature_list = feature_list.reshape((44441, 2048)) # özellik dizisini yeniden şekillendiriyoruz. 

filenames_list = np.array(pickle.load(open("filenames.pkl","rb"))) 


model =ResNet50(weights="imagenet",include_top=False,input_shape=(224,224,3))
model.trainable = False 

model = tensorflow.keras.Sequential([model])
model.add(GlobalMaxPooling2D())  


img = image.load_img("sample/1541.jpg",target_size=(224,224)) # test 
image_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(image_array,axis=0)
preprocess_img = preprocess_input(expanded_img_array)
result = model.predict(preprocess_img)
normalized_result = result / norm(result) 


neighbors = NearestNeighbors(n_neighbors=6,algorithm="brute",metric="euclidean") # KNN algoritması ile her bir görüntü için en yakın 6 komşu bulunacak. 

neighbors.fit(reshaped_feature_list)  

distances,indices = neighbors.kneighbors(normalized_result) # görüntün en yakın komşuları arasındaki mesafeler ile onların indekslerini içeren iki dize elde ederiz. 

print(indices)


for file in indices[0][1:6]: # kendi görüntüsünü içermemek üzere en yakın 6 komşuyu alırız 
    temp_img = cv2.imread(filenames_list[file]) # görüntüyü filenames_list içinden indexi file olan görüntünün dosya yolundan okur ve değişkene atar. 
    cv2.imshow('output',cv2.resize(temp_img,(350,350))) # görüntüyü ekranda görmemizi sağlar. 
    cv2.waitKey(0)



