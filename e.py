import cv2
import algoritma
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu


def video_capture(bfr):
    capture = cv2.VideoCapture(bfr)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    
    
    while True:
        kontrol , frame = capture.read()
        frame1 = cv2.cvtColor(frame , cv2.COLOR_RGB2GRAY)
        frame2 = cv2.resize(frame1 , (50,50))
        
        
        threshold_value = threshold_otsu(frame2)
        binary_image = frame2 > threshold_value
        image_array = binary_image.flatten()
        image_array = image_array.reshape(1,-1)
        prediction= algoritma.model.predict(image_array)
        
        
        
        if prediction == 1:
            cv2.putText(frame,"Kadin", (10,700), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),10)
        if prediction== 0:
            cv2.putText(frame,"Erkek", (10,700), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0),10)
        
        
        cv2.imshow("frame" , frame)
        
        
        
        frame1 = cv2.cvtColor(frame , cv2.COLOR_RGB2GRAY)
        frame2 = cv2.resize(frame1 , (50,50))
    
        

        if cv2.waitKey(1) & 0xFF == ord("q"):# q tuşu ile çıkış işlemi gerçekleştirir
            break
    cv2.destroyAllWindows()

if __name__=='__main__':
    video_capture(0)
    