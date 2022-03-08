# Posture Detection Authentication System

The project is mainly focused on facial recognition and posture detection 
An image will be displayed and the user has to mimic the same posture.
Program will detect the posture of body and hand.
Now program will create encoding of the user face and try match it with the stored face encoding  of user.


## Installation 

pip install opencv-python==4.5.5.62 <br />
pip install face-recognition==1.3.0(first install visual studio in the system) <br />
pip install matplotlib==3.1.1 <br />
pip install mediapipe==0.8.9.1 <br />
pip install numpy==1.16.5 <br />
pip install opencv-python==4.5.5.62 <br />


## Face Encoding.py


def Encode_Images(path):
<t />   ce = create_Encoding(path)<br />
<t />   ce.encode_image()<br />


Pass the file location where the photo of the users are stored with a '/' at the end

## Posture Based Authentication System.py

Verify(Gesture_Image,Selfie_Image)

Pass the provided Gesture and the selfie of the user mimicing the user

