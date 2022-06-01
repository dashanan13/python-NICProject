# import necessary packages
import cv2
import os, sys
import numpy as np
import datetime, time
import mediapipe as mp

# website
from flask import Flask, render_template, Response, request

from threading import Thread

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# initialize mediapipe 
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# store background images in a list
image_path = os.getcwd()  + '\images'
images = os.listdir(image_path)

global capture,rec_frame, grey, switch, neg, face, rec, out 
capture=0
grey=0
neg=0
face=0
switch=1
rec=0

image_index= 0
bg_image = cv2.imread(image_path+'/'+images[image_index])

def cartonify(img):

    myheight , mywidth, channel = img.shape

    #converting an image to grayscale
    grayScaleImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    smoothGrayScale = cv2.medianBlur(grayScaleImage, 5)
    
    #retrieving the edges for cartoon effect
    #by using thresholding technique
    getEdge = cv2.adaptiveThreshold(smoothGrayScale, 255, 
    cv2.ADAPTIVE_THRESH_MEAN_C, 
    cv2.THRESH_BINARY, 9, 9)
    
    #applying bilateral filter to remove noise 
    #and keep edge sharp as required
    colorImage = cv2.bilateralFilter(img, 9, 300, 300)
    
    #masking edged image with our "BEAUTIFY" image
    cartoonImage = cv2.bitwise_and(colorImage, colorImage, mask=getEdge)
    ReSized = cv2.resize(cartoonImage, (mywidth, myheight))

    return ReSized

def img2sketch(img, k_size):
        
    myheight , mywidth, channel = img.shape

    # Convert to Grey Image
    grey_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert Image
    invert_img=cv2.bitwise_not(grey_img)
    #invert_img=255-grey_img

    # Blur image
    blur_img=cv2.GaussianBlur(invert_img, (k_size,k_size),0)

    # Invert Blurred Image
    invblur_img=cv2.bitwise_not(blur_img)
    #invblur_img=255-blur_img

    # Sketch Image
    sketch_img=cv2.divide(grey_img,invblur_img, scale=256.0)

    # Make the grey scale image have three channels
    grey_3_channel = cv2.cvtColor(sketch_img, cv2.COLOR_GRAY2BGR)
    ReSized = cv2.resize(grey_3_channel, (mywidth, myheight))

    return ReSized

def changebackground(frame, bg_image):

    # flip the frame to horizontal direction
    frame = cv2.flip(frame, 1)
    height , width, channel = frame.shape

    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # get the result 
    results = selfie_segmentation.process(RGB)

    # extract segmented mask
    mask = results.segmentation_mask

    # it returns true or false where the condition applies in the mask
    condition = np.stack(
      (results.segmentation_mask,) * 3, axis=-1) > 0.6

    # resize the background image to the same size of the original frame
    bg_image = cv2.resize(bg_image, (width, height))

    # combine frame and background image using the condition
    output_image = np.where(condition, frame, bg_image)

    return output_image

def holisticeatimate(image):
    
    with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
    
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        # Draw landmark annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles
            .get_default_pose_landmarks_style())
        # Flip the image horizontally for a selfie-view display.
        
    return cv2.flip(image, 1)


def wrapperfunctioncartonify():
    while True:
        success, frame = camera.read() 
        if not success:
            #print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        output1 = cartonify(frame)

        try:
            ret, buffer = cv2.imencode('.jpg', cv2.flip(output1,1))
            output1 = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + output1 + b'\r\n')
        except Exception as e:
            pass

def wrapperfunctionimg2sketch():
    while True:
        success, frame = camera.read() 
        if not success:
            #print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        output1 = img2sketch(frame, k_size=151)

        try:
            ret, buffer = cv2.imencode('.jpg', cv2.flip(output1,1))
            output1 = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + output1 + b'\r\n')
        except Exception as e:
            pass

def wrapperfunctionholisticeatimate():
    while True:
        success, frame = camera.read() 
        if not success:
            #print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        output1 = holisticeatimate(frame)

        try:
            ret, buffer = cv2.imencode('.jpg', cv2.flip(output1,1))
            output1 = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + output1 + b'\r\n')
        except Exception as e:
            pass

def wrapperfunctionchangebackground():
    while True:
        success, frame = camera.read() 
        if not success:
            #print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        output1 = changebackground(frame, bg_image)

        try:
            ret, buffer = cv2.imencode('.jpg', cv2.flip(output1,1))
            output1 = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + output1 + b'\r\n')
        except Exception as e:
            pass


#instatiate flask app  
app = Flask(__name__, template_folder='./templates')

# create videocapture object to access the webcam
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

@app.route('/')
def index():
    return render_template('index.html')
    
    
@app.route('/video_feed_cartonify')
def video_feed_cartonify():
    return Response(wrapperfunctioncartonify(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_img2sketch')
def video_feed_img2sketch():
    return Response(wrapperfunctionimg2sketch(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_holisticeatimate')
def video_feed_holisticeatimate():
    return Response(wrapperfunctionholisticeatimate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_changebackground')
def video_feed_changebackground():
    return Response(wrapperfunctionchangebackground(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':
        if  request.form.get('stop') == 'Stop/Start':    
            if(switch==1):
                switch=0
                camera.release()
                cv2.destroyAllWindows()
                
            else:
                camera = cv2.VideoCapture(0)
                switch=1
                          
    elif request.method=='GET':
        return render_template('index.html')
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
    

# release the capture object and close all active windows 
camera.release()
cv2.destroyAllWindows()



