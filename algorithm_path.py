import cv2
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import matplotlib.pyplot as plt


from sklearn.datasets import fetch_openml  # veri kümesi için kullanılır
from skimage.filters import threshold_otsu


from sklearn.tree import DecisionTreeClassifier # karar ağacı sınıflandırma modeli
from sklearn.ensemble import RandomForestClassifier # rastgele orman sınıflandırma modeli

from sklearn.metrics import accuracy_score # başarım ölçütü
from sklearn.metrics import f1_score # başarım ölçütü

from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay # confusion matrix hesaplama ve gösterimi

import numpy as np


location = "data/Training/"

training_data = []

gender = ["male" , "female"]


for x in gender:
    num_gender = gender.index(x)
    for i in os.listdir(location+str(x)):
        data = cv2.imread(location + x + "/" + str(i) )
        data_1 = cv2.cvtColor(data , cv2.COLOR_BGR2GRAY)
        data_2 = cv2.resize(data_1 , (50,50))
        training_data.append([data_2 , num_gender])
    
main_data = pd.DataFrame(training_data)

main_data['Gender'] = main_data[1]
del main_data[1]

main_data['Images'] = main_data[0]

del main_data[0]


X_clean = []
for image in main_data["Images"] :
    threshold_value = threshold_otsu(image)
    binary_image = image > threshold_value
    X_clean.append(binary_image.flatten())
    
    
x_train , x_test , y_train , y_test = train_test_split(X_clean , main_data["Gender"] , test_size=0.2 , random_state=42)


model =  RandomForestClassifier()

# model eğitimi
model.fit(x_train , y_train)

# test veri kümesi üzerinden tahmin yapılması

y_pred = model.predict(x_test)


accuracy = accuracy_score(y_test , y_pred)
print("accuracy: " , accuracy)
image_array = np.array(image) 
def algorithm(img):
    threshold_value = threshold_otsu(img)
    binary_image = img > threshold_value
    image_array = binary_image.flatten()
    image_array = image_array.reshape(1,-1)
    prediction= model.predict(image_array)
    

    
