import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import mediapipe as mp
import face_recognition as fr
import pickle
import os

## Body Posture Matching
class Body_Posture_Detection:
    def __init__(self, image1, image2, side):
        self.__selfie = image1 
        self.__gesture = image2
        self.side = side
    def find_distance(self,x_plot_uploaded,y_plot_uploaded,x_plot_selfie,y_plot_selfie):


        # Finding Box Co-ordinate
        
        xmin_ges, xmax_ges = min(self.x_plot_gesture), max(self.x_plot_gesture)
        ymin_ges, ymax_ges = min(self.y_plot_gesture), max(self.y_plot_gesture)
        boxW_ges, boxH_ges = xmax_ges - xmin_ges, ymax_ges - ymin_ges
        bbox_ges = xmin_ges, ymin_ges, boxH_ges, boxW_ges
        
        xmin_self, xmax_self = min(self.x_plot_selfie), max(self.x_plot_selfie)
        ymin_self, ymax_self = min(self.y_plot_selfie), max(self.y_plot_selfie)
        boxW_self, boxH_self = xmax_self - xmin_self, ymax_self - ymin_self
        bbox_self = xmin_self, ymin_self, boxH_self, boxW_self 
       
    
        #Scaling 
        
        x_factor = boxW_ges/boxW_self
        y_factor = boxH_ges/boxH_self
        #print(x_factor,y_factor)
        for i in range(0,len(x_plot_uploaded)):
            x_plot_uploaded[i] = x_plot_uploaded[i] - xmin_ges
            y_plot_uploaded[i] = y_plot_uploaded[i] - ymin_ges
        
        for i in range(0,len(x_plot_selfie)):
            x_plot_selfie[i] = x_plot_selfie[i] - xmin_self
            y_plot_selfie[i] = y_plot_selfie[i] - ymin_self
        
        for i in range(0,len(x_plot_selfie)):
            x_plot_selfie[i] = x_plot_selfie[i]*x_factor
            y_plot_selfie[i] = y_plot_selfie[i]*y_factor
            
        totaldist = 0

        # Zero Padding the list
        
        if len(x_plot_uploaded) > len(x_plot_selfie) : 
            for i in range(0,len(x_plot_uploaded) - len(x_plot_selfie)):
                x_plot_selfie.append(0)
                y_plot_selfie.append(0)
        else:
            for i in range(0,(len(x_plot_selfie) - len(x_plot_uploaded))):
                x_plot_uploaded.append(0)
                y_plot_uploaded.append(0)
                
        ## Calculating Distance

        for i in range(0,min(len(x_plot_uploaded), len(x_plot_selfie))):
            dist = math.pow(abs(x_plot_selfie[i] - x_plot_uploaded[i]),2)
            dist = dist + math.pow(abs(y_plot_selfie[i] - y_plot_uploaded[i]),2)
            totaldist = totaldist + math.sqrt(dist)
        return totaldist  
        
    def pose_landmarks(self,photo) :        
        mp_drawing = mp.solutions.drawing_utils
        mp_holistic = mp.solutions.holistic

        ## storing body posture co-ordinates
        
        x_plot = []
        y_plot = []
        unwanted_pts = [15,16,17,18,19,20,21,22,23,24]

        ## unwanted points if right/left side detected
        
        if self.side == "Left" : 
            unwanted_pts.append(14)
            unwanted_pts.append(16)
        else:
            unwanted_pts.append(13)
            unwanted_pts.append(15)
            
        count = 0
        ## Calculating co-ordinates
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            image = cv2.cvtColor(photo,cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            if results.pose_landmarks :
                for lm in results.pose_landmarks.landmark:

                    ih, iw, ic = image.shape
                    x,y = int(lm.x * iw) , int(lm.y * ih)
                    if lm.visibility > 0.8 and count not in unwanted_pts:
                        x_plot.append(x)
                        y_plot.append(y)
                    count = count + 1
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            else :
                print("Body Posture Not Found")
                return -1,-1
        return x_plot[11:],y_plot[11:]


    def Correct_Posture(self):
        self.x_plot_gesture, self.y_plot_gesture = self.pose_landmarks(self.__gesture)
        self.x_plot_selfie, self.y_plot_selfie = self.pose_landmarks(self.__selfie)
        if self.x_plot_selfie == -1:
            return False
       
        distance = self.find_distance(self.x_plot_gesture, self.y_plot_gesture,self.x_plot_selfie, self.y_plot_selfie)
        ## verifying wheather the distance is less than threshold 
        if distance < 150 :
            print("Body Posture Verified")
            return True
        else :
            print("Body Incorrect Posture")
            return False 
    
## Hand Posture Matching

class Hand_Posture_Detection:
    def __init__(self, image1, image2):
        self.__selfie = image1 
        self.__gesture = image2
   
    def find_distance(self,x_plot_uploaded,y_plot_uploaded,x_plot_selfie,y_plot_selfie):

        ## Finding Box Co-ordinate

        xmin_ges, xmax_ges = min(self.x_plot_gesture), max(self.x_plot_gesture)
        ymin_ges, ymax_ges = min(self.y_plot_gesture), max(self.y_plot_gesture)
        boxW_ges, boxH_ges = xmax_ges - xmin_ges, ymax_ges - ymin_ges
        bbox_ges = xmin_ges, ymin_ges, boxH_ges, boxW_ges
        
        xmin_self, xmax_self = min(self.x_plot_selfie), max(self.x_plot_selfie)
        ymin_self, ymax_self = min(self.y_plot_selfie), max(self.y_plot_selfie)
        boxW_self, boxH_self = xmax_self - xmin_self, ymax_self - ymin_self
        bbox_self = xmin_self, ymin_self, boxH_self, boxW_self 
       
    
        ##Scaling 
        
        x_factor = boxW_ges/boxW_self
        y_factor = boxH_ges/boxH_self
        
        for i in range(0,len(x_plot_uploaded)):
            x_plot_uploaded[i] = x_plot_uploaded[i] - xmin_ges
            y_plot_uploaded[i] = y_plot_uploaded[i] - ymin_ges
        
        for i in range(0,len(x_plot_selfie)):
            x_plot_selfie[i] = x_plot_selfie[i] - xmin_self
            y_plot_selfie[i] = y_plot_selfie[i] - ymin_self
        
        x_factor_ges = 140/boxW_ges
        y_factor_ges = 140/boxH_ges
        
        x_factor_self = 140/boxW_self
        y_factor_self = 140/boxH_self
        
        for i in range(0,len(x_plot_uploaded)):
            x_plot_uploaded[i] = x_plot_uploaded[i]*x_factor_ges
            y_plot_uploaded[i] = y_plot_uploaded[i]*y_factor_ges
        
        for i in range(0,len(x_plot_selfie)):
            x_plot_selfie[i] = x_plot_selfie[i]*x_factor_self
            y_plot_selfie[i] = y_plot_selfie[i]*y_factor_self
        
        totaldist = 0
        
        ## Zero Padding the list
        
        if len(x_plot_uploaded) > len(x_plot_selfie) : 
            for i in range(0,len(x_plot_uploaded) - len(x_plot_selfie)):
                x_plot_selfie.append(0)
                y_plot_selfie.append(0)
        else:
            for i in range(0,(len(x_plot_selfie) - len(x_plot_uploaded))):
                x_plot_uploaded.append(0)
                y_plot_uploaded.append(0)
                
        ## Calculating Distance

        for i in range(0,min(len(x_plot_uploaded), len(x_plot_selfie))):
            dist = math.pow(abs(x_plot_selfie[i] - x_plot_uploaded[i]),2)
            dist = dist + math.pow(abs(y_plot_selfie[i] - y_plot_uploaded[i]),2)
            totaldist = totaldist + math.sqrt(dist)
        return totaldist    
        
    def hand_landmarks(self,photo) :        
        mp_drawing = mp.solutions.drawing_utils
        mp_holistic = mp.solutions.holistic

        ## storing hand posture co-ordinates
        
        x_plot = []
        y_plot = []
        self.side = None
        with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic: 
            photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
            results = holistic.process(photo)
            # If left hand detected else if right hand detected
            if results.right_hand_landmarks is not None :
                self.side = "Right"
                mp_drawing.draw_landmarks(photo, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                for lm in results.right_hand_landmarks.landmark:
                    ih, iw, ic = photo.shape
                    x,y = int(lm.x * iw) , int(lm.y * ih)
                    x_plot.append(x)
                    y_plot.append(y)
            elif results.left_hand_landmarks is not None :
                self.side = "Left"for lm in results.left_hand_landmarks.landmark:
                    ih, iw, ic = photo.shape
                    x,y = int(lm.x * iw) , int(lm.y * ih)
                    #print(poselms)
                    x_plot.append(x)
                    y_plot.append(y)
            else :
                print("Hand Posture Not Found")
                return -1,-1,self.side
            
           
        return x_plot,y_plot,self.side


    def Correct_Posture(self): 
        self.__selfie = cv2.resize(self.__selfie,(self.__gesture.shape[1],self.__gesture.shape[0]))
        self.x_plot_gesture, self.y_plot_gesture , self.side_ges = self.hand_landmarks(self.__gesture)
        self.x_plot_selfie, self.y_plot_selfie, self.side_self = self.hand_landmarks(self.__selfie)
        if self.x_plot_selfie == -1:
            return False, None
        
        if self.side_ges != self.side_self or self.side_ges == None or self.side_self == None:
            return False, None 
        
        distance = self.find_distance(self.x_plot_gesture, self.y_plot_gesture,self.x_plot_selfie, self.y_plot_selfie)
        
        ## verifying wheather the distance is less than threshold
        
        if distance < 550 :
            print("Hand Posture Verified")
            return True,self.side_ges
        else :
            print("Incorrect Hand Posture")
            return False,self.side_ges 
    

## Facial Recognition

class face_authentication:
    def __init__(self):
        self.__Selfie = None
        self.__face_encodings = None
    ## detecting face , and returning co-ordinates
        
    def face_extraction(self, Image):
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
        
        face = face_cascade.detectMultiScale(Image,1.067,5)
        if len(face) == 0:
            print("Face Not Found !!")
            return -1,-1,-1,-1
        face = sorted(face,key = lambda f:f[2]*f[3] , reverse = True)[0]
        
        x,y,h,w = face
        return x,y,h,w
    
    
    def identify_face(self, Image):
        self.__Selfie = Image
        x,y,h,w = self.face_extraction(Image)
        if x == -1:
            return False
        self.__Selfie = self.__Selfie[y-50:y+h+50, x-50:x+w+50]
      #  plt.imshow(self.__Selfie)
        print(self.__Selfie.shape)
        known_face_encodings = pickle.load(open('FaceEncodingUser.pickle','rb'))
        face_location = fr.face_locations(self.__Selfie, model = "hog")
        face_encodings = fr.face_encodings(self.__Selfie, face_location)
        if len(face_encodings) == 0:
            print("Please bring up the Light or Retake the photo")
            return False
        ## Matcing the current face encoding with the stored encoding
        
        matches = fr.compare_faces(known_face_encodings,face_encodings[0],0.4)
        if len(face_encodings) == 0:
            print("Face Not Detected")
            return False
        if True in matches:
            print("Face Verified")
            return True
        else:
            print("Face Not Verified")
            return False


def Verify(Gesture_Image , Selfie_Image):

    ## list to numpy array
    
    Gesture = np.asarray(Gesture_Image)
    Selfie = np.asarray(Selfie_Image)
    FacePhoto = Selfie_Image
    GesturePhoto = Gesture_Image

    ## hand posture matching
    
    HandPostureDetection = Hand_Posture_Detection(FacePhoto,GesturePhoto)
    Hands,Side = HandPostureDetection.Correct_Posture()

    ## face matching 
    
    FaceAuthentication = face_authentication()
    Face = FaceAuthentication.identify_face(Selfie)

    ## body posture matching
    
    BodyPostureDetection = Body_Posture_Detection(Selfie_Image, Gesture_Image,Side)
    Body = BodyPostureDetection.Correct_Posture()
    
    
    if Hands is True and Face is True and Body is True :
        print("Access Granted !!!!!")
        return True
    else :
        print("Access Denied !!!!")
        return False


Verify(Gesture_Image,Selfie_Image)

