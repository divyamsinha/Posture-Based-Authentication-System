import face_recognition as fr
import os
import pickle
import cv2
import matplotlib.pyplot as plt


class create_Encoding:
    
    def __init__(self, path):
        self.dbPath = path
    ## Detecting the face co-ordinates , cropping the face 
    def Get_Face_Image(self,image):   
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
        face = face_cascade.detectMultiScale(image,1.067,5)
        if len(face)>0:
            face = sorted(face,key = lambda f:f[2]*f[3] , reverse = True)[0]
            offset = 35
            x,y,h,w = face
            image = image[y-offset:y+h+offset,x-offset:x+w+offset] 
        return image
    ## Encoding the provided faces
    def encode_image(self):
        faces = os.listdir(self.dbPath)
        known_face_endings = []
        # Encoding the faces onr by one
        for face in faces:
            if face[0] != '.' :
                image = fr.load_image_file(self.dbPath + face)
                face_image = self.Get_Face_Image(image)
                encoding = fr.face_encodings(image)[0]
                known_face_endings.append(encoding)
        ## Storing it in a file
        file = open('FaceEncodingUser.pickle' ,'wb')
        pickle.dump(known_face_endings,file)
        file.close()
        

def Encode_Images(path):
    ce = create_Encoding(path)
    ce.encode_image()

