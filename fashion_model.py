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


model =ResNet50(weights="imagenet",include_top=False,input_shape=(224,224,3)) #önceden eğitilmiş ResNet50 modeli kullanıyoruz.
model.trainable = False  # ama modelin ağırlıklarını değiştirmemek için yeniden eğitilebilirliğini kapatıyoruz. 

model = tensorflow.keras.Sequential([model])
model.add(GlobalMaxPooling2D()) # CNN modelinde özellik haritasinda özellik katmanlarından en yüksek özellikleri seçiyoruz. 
print(model.summary())


def extract_features(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    image_array = image.img_to_array(img) # görüntüyü matematiksel olarak daha iyi işlemek için numpy dizisine dönüştüyoruz. 
    expanded_img_array = np.expand_dims(image_array,axis=0) # boyut ekliyoruz modele uygun hale getirmek için [1 2 3 4 5] _____ [[1 2 3 4 5]]
    preprocess_img = preprocess_input(expanded_img_array)  # bu fonksiyon, görüntüyü modelin eğitildiği şekilde ön işler.
    result = model.predict(preprocess_img) # model tahminlerini alıyoruz.
    normalized_result = result / norm(result)  # özellik vektörünü bir birim uzunluğuna sahip vektör haline getiriyoruz 
    
    return normalized_result # görüntün özellik çıkarma işlemlerini yaptık ve Resnet50 modeli için uygun hale getirdik. 

# print(os.listdir("images"))


file_names = []

for file in os.listdir("images"):
    file_names.append(os.path.join("images",file)) # dosya adlarını alıp dosyanın tam yolu ile birleştiriyoruz. 'images/9733.jpg', 'images/14147.jpg',

# print(file_names[0:6])
# print("images count: ", len(file_names))

feature_list = []

for file in tqdm(file_names):
    feature_list.append(extract_features(file,model)) #  her dosya üzerinde döngüye girip görüntüyü özelliklerden çıkarıp yeni listeye yazdırıyoruz. 
                                                      #  tdqm ilerleyisi cubuk olarak gösteriyor. 

pickle.dump(feature_list,open("embeddings.pkl","wb"))
pickle.dump(file_names,open("filenames.pkl","wb"))

