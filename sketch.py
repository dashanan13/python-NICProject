# DataFlair background removal 

# import necessary packages
import os
import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


# initialize mediapipe 
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)


# store background images in a list
image_path = os.getcwd()  + '\images'
images = os.listdir(image_path)

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

# create videocapture object to access the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    cartoonoutput = cartonify(image)
    output1 = changebackground(cartoonoutput, bg_image)
    
    output2 = img2sketch(image, k_size=151)

    output3 = holisticeatimate(image)

    outputa = np.concatenate((output1, output2), axis = 0)
    outputb = np.concatenate((output3, image), axis = 0)
    output = np.concatenate((outputa, outputb), axis = 1)

    # show outputs
    cv2.imshow("Output", output)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    # if 'd' key is pressed then change the background image
    elif key == ord('d'):

        if image_index != len(images)-1:
            image_index += 1
        else:
            image_index = 0
        bg_image = cv2.imread(image_path+'/'+images[image_index])


# release the capture object and close all active windows 
cap.release()
cv2.destroyAllWindows()



