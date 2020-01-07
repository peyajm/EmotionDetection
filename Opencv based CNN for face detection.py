#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time


# In[2]:




# DNN stands for OpenCV: Deep Neural Networks
DNN = "TF" # Or CAFFE, or any other suported framework
min_confidence = 0.5 # minimum probability to filter weak detections


# In[3]:


# load our serialized model from disk
print("[INFO] loading model...")

if DNN == "CAFFE":
    modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile= "deploy.prototxt"
    
    # Here we need to read our pre-trained neural net created using Caffe
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
else:
    modelFile = "/home/ws2/Desktop/ashima/opencv_face_detector_uint8.pb"
    configFile= "/home/ws2/Desktop/ashima/opencv_face_detector.pbtxt"
    
    # Here we need to read our pre-trained neural net created using Tensorflow
    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
    
print("[INFO] model loaded.")


# In[4]:


#filename1 ='/media/ws2/KINGSTON/Movie trailer classification/Dataset/Frames/Philauri/CNN based Hog/Detected'
#filename2 = '/media/ws2/KINGSTON/Movie trailer classification/Dataset/Frames/Philauri/CNN based Hog/Undetected'

filename1 ='/home/ws2/Desktop/ashima/Dataset/Frames/War/Detected'
filename2 = '/home/ws2/Desktop/ashima/Dataset/Frames/War/Undetected'


# In[ ]:


framecount = 0
count = 0
success=1


cap = cv2.VideoCapture('/home/ws2/Desktop/ashima/Dataset/Trailers/War _0.mp4')
fps = int(cap.get(cv2.CAP_PROP_FPS))
print("FPS:",fps)


start = time.time()
print("Start time:", start)

while success:
    
    # Read the frame
    success, frame = cap.read()
    print("Success:", success)
    if(success):
        print("Input frame:")
        plt.imshow(frame, cmap = 'gray', interpolation = 'bicubic')
        plt.show()
        # Our operations on the frame come here
        frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame1 = cv2.resize(frame,(int(600),int(400)))

        blob = cv2.dnn.blobFromImage(cv2.resize(frame1, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), 
                                     swapRB=False, crop=False)
        net.setInput(blob)
        detections = net.forward()
        print("Detections:")
        print(len(detections))

        # ensure at least one face was found
        if len(detections) > 0:
            # we're making the assumption that each image has only ONE
            # face, so find the bounding box with the largest probability
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]
            print("Confidence:",confidence)

            # ensure that the detection with the largest probability also
            # means our minimum probability test (thus helping filter out
            # weak detections)
            if confidence > min_confidence:
                # compute the (x, y)-coordinates of the bounding box for
                # the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")               
                
                # draw the bounding box of the face along with the associated
                # probability
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame1, (startX, startY), (endX, endY),
                    (0, 0, 255), 2)
       
                cv2.putText(frame1, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                cv2.imwrite(filename1+ '/'+ str(framecount)+ 'detect.jpg',frame1)
                print("Detected:",(filename1+ '/'+ str(framecount)+ 'detect.jpg'))
                framecount += 1
                plt.imshow(frame1, cmap = 'gray', interpolation = 'bicubic')
                plt.xticks([]), plt.yticks([]) 
                plt.show()
            else:
                cv2.imwrite(filename2+ '/'+ str(count)+ 'undetect.jpg',frame1)
                print("UnDetected:",(filename2+ '/'+ str(count)+ 'undetect.jpg'))

                count += 1
                #display resulting frame
                plt.imshow(frame1, cmap = 'gray', interpolation = 'bicubic')
                plt.xticks([]), plt.yticks([]) 
                plt.show()
        
    else:
        end = time.time()
        print("End time:", end)
        print("Time:",format(end - start, '.2f'))
        # Release the VideoCapture object
        cap.release()
        print("Done!!!!!!!!!!!!!!")
        cv2.destroyAllWindows()
        


# In[ ]:



            
        
        (h, w) = frame1.shape[:2]
        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (probability) associated with the prediction
            confidence = detections[0, 0, i, 2]
            print("Confidence:",confidence)

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > min_confidence:
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # draw the bounding box of the face along with the associated
                # probability
                
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame1, (startX, startY), (endX, endY),
(0, 0, 255), 2)
                cv2.putText(frame1, text, (startX, y),
cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                #cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 2)


# In[7]:


framecount = 0
count = 0
success=1


cap = cv2.VideoCapture('/home/ws2/Desktop/ashima/Dataset/Trailers/Ek thi dayan.mp4')
fps = int(cap.get(cv2.CAP_PROP_FPS))
print("FPS:",fps)
#start=time.process_time()

start = time.time()
print("Start time:", start)

while success:
    
    # Read the frame
    success, frame = cap.read()
    print("Success:", success)
    if(success):
        print("Input frame:")
        plt.imshow(frame, cmap = 'gray', interpolation = 'bicubic')
        plt.show()
        # Our operations on the frame come here
        frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame1 = cv2.resize(frame,(int(600),int(400)))

        blob = cv2.dnn.blobFromImage(cv2.resize(frame1, (300, 300)), 1.0, (300, 300), (103.93, 116.77, 123.68))
        net.setInput(blob)
        detections = net.forward()
        (h, w) = frame1.shape[:2]
        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (probability) associated with the prediction
            confidence = detections[0, 0, i, 2]
            print("Confidence:",confidence)

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > min_confidence:
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # draw the bounding box of the face along with the associated
                # probability
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame1, (startX, startY), (endX, endY),
                    (0, 0, 255), 2)
                cv2.putText(frame1, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                #cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 2)
                cv2.imwrite(filename1+ '/'+ str(framecount)+ 'detect.jpg',image)
                print("Detected:",(filename1+ '/'+ str(framecount)+ 'detect.jpg'))
                framecount += 1
                plt.imshow(image, cmap = 'gray', interpolation = 'bicubic')
                plt.xticks([]), plt.yticks([]) 
                plt.show()
            else:
                cv2.imwrite(filename2+ '/'+ str(count)+ 'undetect.jpg',image)
                print("UnDetected:",(filename2+ '/'+ str(count)+ 'undetect.jpg'))
                #print("Two:", count)
                #print(filename2+ '/'+ str(count)+ '.jpg')
                count += 1
                #display resulting frame
                plt.imshow(image, cmap = 'gray', interpolation = 'bicubic')
                plt.xticks([]), plt.yticks([]) 
                plt.show()
        
    else:
        end = time.time()
        print("End time:", end)
        print("Time:",format(end - start, '.2f'))
        # Release the VideoCapture object
        cap.release()
        print("Done!!!!!!!!!!!!!!")
        cv2.destroyAllWindows()
        


# In[ ]:


11:52

