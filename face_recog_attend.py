# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 18:11:37 2018

@author: Welkin
"""

import cv2
from align_custom import AlignCustom
from face_feature import FaceFeature
from mtcnn_detect import MTCNNDetect
from tf_graph import FaceRecGraph
import argparse
import json
import sys
import numpy as np
import pandas as pd
import time
import dlib
from collections import OrderedDict
from scipy.spatial import distance as dist

eye_ar_thresh = 0.3
eye_ar_frames = 3
FACIAL_LANDMARK_INDXS = OrderedDict([("mouth",(48,68)),("left_eyebrow",(22,27)),("right_eyebrow",(17,22)),("left_eye",(42,48)),('right_eye',(36,42)),('jaw',(0,17)),("nose",(27,36))])

(lStart,lEnd) = FACIAL_LANDMARK_INDXS['left_eye']
(rStart,rEnd) = FACIAL_LANDMARK_INDXS['right_eye']

start_time = time.time()
attendance = pd.read_csv(r'.\Attendance.csv',delimiter=',')
daily_attendance = pd.DataFrame(columns = ['Name','Attendance'],index=range(60))
daily_attendance['Name'][:2]=['Akash Yadav','Pawan Jha']
daily_attendance['Attendance'][:60]=[0]*60
path = r'.\shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(path)
def main(args):
    mode = args.mode
    #print(mode)
    if mode == "camera":
        face_recog()
    elif mode == 'input':
        new_data_entry()
    else:
        raise ValueError("Unimplemented Mode")

def face_recog():
    print("Camera is On.\nFace Recognition in process...")
    #for capturing Video from camera
    cap = cv2.VideoCapture(0)#0 is for idefinite time
    counter = 0
    total = 0
    while True:
        ret,frame = cap.read()
        
        #call detect face function it will return landmarks on face and bounding box
        rects,landmark = face_detect.detect_face(frame,80)#min face size set to 100x100
        if rects:
            
            temp_rects = dlib.rectangle(rects[0][0],rects[0][1],rects[0][2],rects[0][3])
            shape = predictor(frame,temp_rects)
            coords = np.zeros((shape.num_parts,2),dtype="int")
            for i in range(shape.num_parts):
                coords[i] = (shape.part(i).x,shape.part(i).y)
            shape = coords
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            
            leftEAR = e_a_r(leftEye)
            rightEAR = e_a_r(rightEye)
            
            #average eye aspect ratio of both eyes
            EAR = (leftEAR + rightEAR)/(2.0)
            if EAR < eye_ar_thresh:
                counter += 1
            else:
                if counter >= eye_ar_frames:
                    total += 1
                counter = 0
                
        
        aligns = []#this list conatin matrix of aligned images 
        positions = []#this list contain position of user's face :left ,right,center
        for (i,rect) in enumerate(rects):#rects contain list of bounding box(x1,y1,w,h)
            aligned_face,face_pos = aligner.align(160,frame,landmark[i])
            if len(aligned_face) == 160 and len(aligned_face[0]) == 160:
                aligns.append(aligned_face)
                positions.append(face_pos)
            else:
                print('Alignment of face is failed')
        if len(aligns)>0:
            #feed align or preprocessed image to function it will return 128D vector which contain unique feature
            #of faces and also called generate embedding function take input as image return 128D vector features
            feat_atrb = extract_feature.get_features(aligns)
            #this function return name and percentage of accuracy with image
            detected_face = findPeople(feat_atrb,positions)
            count = 0#this is for counting attendance of each student
            for i,rect in enumerate(rects):
                count = 0
                #this will draw bounding box on detected face
                cv2.rectangle(frame,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(0,255,0),)
                cv2.putText(frame,detected_face[i][0]+"-"+str(detected_face[i][1])+"%",(rect[0],rect[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)
                if total == 0:
                    cv2.putText(frame,"Fake",(int((rect[0]+rect[2])/2),rect[1]+rect[3]),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                else:
                    cv2.putText(frame,"Blink:{}".format(total),(int((rect[0]+rect[2])/2),rect[1]+rect[3]),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                if detected_face[i][0].lower() == 'akash' and count==0 and total >=1:
                    count+=1
                    daily_attendance.loc[daily_attendance['Name']=='Akash Yadav','Attendance']=count
                elif detected_face[i][0].lower() == 'pawan' and count==0:
                    count+=1
                    daily_attendance.loc[daily_attendance['Name']=='Pawan Jha','Attendance']=count
        cv2.imshow("Face_Recognition",frame)
        ch = cv2.waitKey(1)
        #end_time = time.time()
        #(end_time - start_time)/60 >= 2 or
        
        if ch == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            print("Face recognition completed.")
            break
    attendance['Attendance']+=daily_attendance['Attendance']
    attendance.to_csv(r'.\Attendance.csv',index=False)


def e_a_r(eye):
    #vertical distance
    A = dist.euclidean(eye[1],eye[5])
    B = dist.euclidean(eye[2],eye[4])
    #horizontal distance
    C = dist.euclidean(eye[0],eye[3])
    #eye aspect ratio
    ear = (A + B)/(C*2.0)
    return ear



def findPeople(feature_attr,positions,thres = 0.6,percent_thres = 70):
    '''
    feature_attr:- it contain 128D / vector values of face
    positions:- it contain position of face on screen it can be either left,right or center
    thres:-it is distance threshold
    percent_thres = at least 70% feature match with store data
    return name/ID and percentage
    '''
    file = open(r'.\face_feature_dataset_128D.txt','r')
    face_dataset = json.loads(file.read())
    results = []
    for i,feat_128D in enumerate(feature_attr):
        result = "Unkonwn"
        smallest = sys.maxsize #it hold maxsize of any object or limit that is store by list,str,dict
        for person in face_dataset.keys():
            person_data = face_dataset[person][positions[i]]#it hold 128D vector data of dataset or stored data in database/disk
            for data in person_data:#it will extract list of list of data from list of list
                distance = np.sqrt(np.sum(np.square(data-feat_128D)))#it return euclidian distance of original data that is stored in disk -(minus) current data(128D) from camera
                if distance < smallest:
                    smallest = distance #it check for euclidian distance if differece of distance is high than Unkonwn person or if in range of sys.maxsize than it will known person
                    result = person 
        percentage = min(100,100*thres/smallest)
        if percentage <= percent_thres:
            result = "Unkonwn"
        results.append((result,percentage))
    return results

def new_data_entry():
    cap = cv2.VideoCapture(0)
    name_id = input("Enter name of ID or yours=> ")
    file = open(r'.\face_feature_dataset_128D.txt','r')
    dataset = json.loads(file.read())
    image_matrix = {'Left':[],'Right':[],'Center':[]}
    feature_128D = {'Left':[],'Right':[],'Center':[]}
    print("For Training turn your face slowly from left to right and after completion press q to save in file=>")
    while True:
        ret,frame = cap.read()
        rects,landmarks = face_detect.detect_face(frame,80)#80 is min size
        for (i,rects) in enumerate(rects):
            aligned_face,pos = aligner.align(160,frame,landmarks[i])
            if len(aligned_face[0])==160 and len(aligned_face) == 160:
                image_matrix[pos].append(aligned_face)
                cv2.imshow("Captured Face",cv2.resize(aligned_face,(200,200)))
        ch = cv2.waitKey(1)
        if ch == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
    for pos in image_matrix:
        #this line is for multiple channel image :- to convert 128D vector
        feature_128D[pos] = [np.mean(extract_feature.get_features(image_matrix[pos]),axis=0).tolist()]
    dataset[name_id] = feature_128D
    file = open(r'.\face_feature_dataset_128D.txt','w')
    file.write(json.dumps(dataset))

     
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",type=str,help="Run camera for face recognition",default = 'camera')
    args = parser.parse_args(sys.argv[1:])
    default_graph = FaceRecGraph()#it take info of data flow in tensorflow's layer
    aligner = AlignCustom()
    extract_feature = FaceFeature(default_graph)
    face_detect = MTCNNDetect(default_graph,scale_factor = 2)#rescale image for fast detection of image
    main(args)


#-- the end--#
