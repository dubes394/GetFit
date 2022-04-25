#=====================================================
#Modified by: Augmented Startups & Geeky Bee AI
#Date : 22 April 2019
#Project: Yoga Angle Corrector/Plank Calc/Body Ratio
#Tutorial: http://augmentedstartups.info/OpenPose-Course-S
#=====================================================
import argparse
import logging
import time
from pprint import pprint
import cv2
import numpy as np
import sys
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import math

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0
def find_point(pose, p):
    for point in pose:
        try:
            body_part = point.body_parts[p]
            return (int(body_part.x * width + 0.5), int(body_part.y * height + 0.5))
        except:
            return (0,0)
    return (0,0)
def euclidian( point1, point2):
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2 )
def angle_calc(p0, p1, p2 ):
    '''
        p1 is center point from where we measured angle between p0 and
    '''
    try:
        a = (p1[0]-p0[0])**2 + (p1[1]-p0[1])**2
        b = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
        c = (p2[0]-p0[0])**2 + (p2[1]-p0[1])**2
        angle = math.acos( (a+b-c) / math.sqrt(4*a*b) ) * 180/math.pi
    except:
        return 0
    return int(angle)

def bicep_curl(a,b,c,d,e,f,g,h):
    '''
    a is angle between 2 3 4 
    b is angle between 5 6 7 
    c is distance between 2 4 
    d is distance between 5 7
    e 3 to 8
    f 6 to 11
    g 14 to 10
    h 15 to 13
    '''
    if a in range(10,45) and b in range(10,45) and c in range(20,70) and d in range(20,70) and e in range(75,110) and f in range(75,110) and g in range(20,1450) and h in range(20,1450):
        return True
    return False
def bicep_curl_relax(a,b,c,d,e,f,g,h):
    '''
    a is angle between 2 3 4 
    b is angle between 5 6 7 
    c is distance between 2 4 
    d is distance between 5 7
    e 3 to 8
    f 6 to 11
    g 14 to 10
    h 15 to 13
    '''
    if a in range(150,180) and b in range(150,180) and c in range(140,280) and d in range(140,280) and e in range(75,110) and f in range(75,110) and g in range(20,1450) and h in range(20,1450):
        return True
    return False
def shoulder_press(a,b,c,d,e,f,g,h,i):
    '''
    a is angle between 2 3 4 
    b is angle between 5 6 7
    c is angle between 1 2 3  
    d is angle between 1 5 6  
    e is distance between 4 7 
    f is distance between 4 8
    g 7 to 11
    h 2 to 9
    i 5 to 12
    '''
    if a in range(140,180) and b in range(140,180) and c in range(80,140) and d in range(80,140) and e in range(80,170) and f in range(180,450) and g in range (180,450) and h in range(20,1450) and i in range(20,1450):
        return True
    return False
def shoulder_press_relax(a,b,c,d,e,f,g):
    '''
    a is angle between 2 3 4 
    b is angle between 5 6 7
    c is angle between 1 2 3  
    d is angle between 1 5 6  
    e is distance between 4 7 
    f 2 to 9
    g 5 to 12
    '''
    if a in range(40,80) and b in range(40,80) and c in range(130,170) and d in range(100,170) and e in range(180,250) and f in range(20,1450) and g in range(20,1450):
        return True
    return False

def plank( a, b, c, d, e, f):
    #There are ranges of angle and distance to for plank. 
    '''
        a and b are angles of hands
        c and d are angle of legs
        e and f are distance between head to ankle because in plank distace will be maximum.
    '''
    if (a in range(50,100) or b in range(50,100)) and (c in range(135,175) or d in range(135,175)) and (e in range(50,250) or f in range(50,250)):
        return True
    return False

def stop_gesture(a,b,c,d):
    '''
    a is distance between 4 1 
    b is distance between 7 11
    '''
    if (a in range(5,100) and b in range(5,60)) or (c in range(5,100) and d in range(5,60)):
        return True
    return False

def draw_str(dst, xxx_todo_changeme, s, color, scale):
    
    (x, y) = xxx_todo_changeme
    if (color[0]+color[1]+color[2]==255*3):
        cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, scale, (0, 0, 0), thickness = 4, lineType=10)
    else:
        cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, scale, color, thickness = 4, lineType=10)
    #cv2.line    
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, scale, (255, 255, 255), lineType=11)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=str, default=0)
    parser.add_argument('--resize', type=str, default='432x368',
                        help='if provided, resize images before they are processed. default=432x368, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()
    
    # print("mode 0: Only Pose Estimation \nmode 1: People Counter \nmode 2: Fall Detection \nmode 3: Yoga pose angle Corrector \nmode 4: Planking/Push up Detection \nmode 5: Hourglass ratio")
    # mode = int(input("Enter a mode : "))
    
    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
    count = 0
    i = 0
    frm = 0
    y1 = [0,0]
    global height,width
    orange_color = (0,140,255)
    bc = False
    bcr = True
    count = 0
    mode = 0
    seconds = 0
    first_time = True
    set_mode = True
    while True:
        ret_val, image = cam.read()
        i =1
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        pose = humans
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        height,width = image.shape[0],image.shape[1]
        # distance calculations
        dist24 = int(euclidian(find_point(pose, 2), find_point(pose, 4)))
        dist57 = int(euclidian(find_point(pose, 5), find_point(pose, 7)))
        dist38 = int( euclidian(find_point(pose, 3), find_point(pose, 8)))
        dist611 = int( euclidian(find_point(pose, 6), find_point(pose, 11)))
        dist29 = int( euclidian(find_point(pose, 2), find_point(pose, 9)))
        dist512 = int( euclidian(find_point(pose, 5), find_point(pose, 12)))
        dist48 = int(euclidian(find_point(pose, 4), find_point(pose, 8)))
        dist47 = int(euclidian(find_point(pose, 4), find_point(pose, 7)))
        dist711 = int( euclidian(find_point(pose, 7), find_point(pose, 11)))
        dist41 = int(euclidian(find_point(pose, 4), find_point(pose, 1)))
        dist71 = int(euclidian(find_point(pose, 7), find_point(pose, 1)))
        dist07 = int(euclidian(find_point(pose, 0), find_point(pose, 7)))
        dist04 = int(euclidian(find_point(pose, 0), find_point(pose, 4)))
        # angle calcucations
        angle234 =  angle_calc(find_point(pose, 2), find_point(pose, 3), find_point(pose, 4))
        angle567 =  angle_calc(find_point(pose,5), find_point(pose,6), find_point(pose,7))
        angle123 =  angle_calc(find_point(pose, 1), find_point(pose, 2), find_point(pose, 3))
        angle156 =  angle_calc(find_point(pose,1), find_point(pose,5), find_point(pose,6))
        angle765 =  angle_calc(find_point(pose,7), find_point(pose,6), find_point(pose,5))
        angle111213 =  angle_calc(find_point(pose,11), find_point(pose,12), find_point(pose,13))
        angle432 =  angle_calc(find_point(pose,4), find_point(pose,3), find_point(pose,2))
        angle8910 =  angle_calc(find_point(pose,8), find_point(pose,9), find_point(pose,10))
        #mode_setter
        if set_mode and len(pose) > 0:
            cv2.putText(image,
                    "DO FIRST REP OF EXERCISE TO SELECT",
                    (20, 400),  cv2.FONT_HERSHEY_PLAIN, 1.5,
                    (255,0,0),thickness = 2, lineType=10)
            if first_time :
                cv2.putText(image,
                    "WELCOME TO TRAIN YOUR ANGLE",
                    (20, 50),  cv2.FONT_HERSHEY_PLAIN, 1.5,
                    (255,0,0),thickness = 2, lineType=10)

            if bicep_curl(angle234,angle567,dist24,dist57,dist38, dist611,dist29,dist512):
                mode = 1
                set_mode = False
            elif shoulder_press(angle234,angle567,angle123,angle156,dist47,dist48,dist711,dist29,dist512):
                mode = 2
                set_mode = False
            elif plank(angle765, angle432, angle111213, angle8910,dist04, dist07):
                mode = 3
                set_mode = False

        #bicep_curl        
        if set_mode == False and mode == 1 and len(pose) > 0:
            if bc and bicep_curl(angle234,angle567,dist24,dist57,dist38, dist611,dist29,dist512):
                        #draw_str(frame, (20, 220), " Bicep Curl", orange_color, 1.5)
                        #action = "Bicep Curl"
                        is_yoga = True
                        #if prev_action == 'Unknown' or prev_action == "Unknown_First":
                        #    yoga_duration = time.time()
                        # draw_str(image, (20, 50), action, orange_color, 2)
                        # logger.debug("*** Bicep Curl ***")
                        bcr = True
                        bc = False
                        count+=1
                        logger.debug(count)
            if bcr and bicep_curl_relax(angle234,angle567,dist24,dist57,dist38, dist611,dist29,dist512):
                        #draw_str(frame, (50, 220), " Bicep Curl Relax", orange_color, 1.5)
                        #action = "Bicep Curl Relax"
                        is_yoga = True
                        #if prev_action == 'Unknown' or prev_action == "Unknown_First":
                        #    yoga_duration = time.time()
                        # draw_str(image, (50, 50), action, orange_color, 2)
                        # logger.debug("*** Bicep Curl Relax***")
                        bcr = False
                        bc = True
            cv2.putText(image,
                    "BICEP CURL REP COUNT: %d" % (count),
                    (20, 50),  cv2.FONT_HERSHEY_PLAIN, 1.5,
                    (0,140,255), thickness = 2, lineType=10)
            cv2.putText(image,
                    "RAISE YOUR SINGLE HAND TO STOP",
                    (20, 400),  cv2.FONT_HERSHEY_PLAIN, 1,
                    (255,0,0), thickness = 2, lineType=10)


        #shoulder_press
        if set_mode == False and mode == 2 and len(pose) > 0:
            if bc and shoulder_press_relax(angle234,angle567,angle123,angle156,dist47,dist29,dist512):
                        #draw_str(frame, (20, 220), " Bicep Curl", orange_color, 1.5)
                        #action = "Shoulder Press Relax"
                        #if prev_action == 'Unknown' or prev_action == "Unknown_First":
                        #    yoga_duration = time.time()
                        #draw_str(image, (20, 50), action, orange_color, 2)
                        logger.debug("*** Shoulder Press Relax ***")
                        bcr = True
                        bc = False
                        count+=1
                        logger.debug(count)
                        
            if bcr and shoulder_press(angle234,angle567,angle123,angle156,dist47,dist48,dist711,dist29,dist512):
                        #draw_str(frame, (50, 220), " Bicep Curl Relax", orange_color, 1.5)
                        #action = "Shoulder Press"
                        is_yoga = True
                        #if prev_action == 'Unknown' or prev_action == "Unknown_First":
                        #    yoga_duration = time.time()
                        #draw_str(image, (50, 50), action, orange_color, 2)
                        logger.debug("*** Shoulder Press ***")
                        bcr = False
                        bc = True
            cv2.putText(image,
                    "SHOULDER PRESS REP COUNT: %d" % (count),
                    (20, 50),  cv2.FONT_HERSHEY_PLAIN, 1.5,
                    (0,140,255), thickness = 2, lineType=10)
            cv2.putText(image,        
                    "RAISE YOUR SINGLE HAND TO STOP",
                    (20, 400),  cv2.FONT_HERSHEY_PLAIN, 1,
                    (255,0,0), thickness = 2, lineType=10)


        #plank
        if set_mode == False and mode == 3 and len(pose) > 0:

            if plank(angle765, angle432, angle111213, angle8910,dist04, dist07):
                        action = "Plank"
                        is_yoga = True
                        #if prev_action == 'Unknown' or prev_action == "Unknown_First":
                        #    yoga_duration = time.time()
                        #logger.debug("*** Plank ***")
                        # draw_str(image, (20, 50), " Plank", orange_color, 2)
                        seconds +=1
                        time.sleep(1)
            cv2.putText(image,
                    "PLANK DURATION %d" % (seconds),
                    (20, 50),  cv2.FONT_HERSHEY_PLAIN, 1.5,
                    (0,140,255), thickness = 2, lineType=10)
            cv2.putText(image,        
                      "RAISE YOUR SINGLE HAND TO STOP",
                      (20, 400),  cv2.FONT_HERSHEY_PLAIN, 1,
                      (255,0,0), thickness = 2, lineType=10)
        #stop_gesture
        if set_mode == False and len(pose) > 0 :
            if stop_gesture(dist41,dist711,dist71,dist48):
                        #draw_str(frame, (20, 220), " Bicep Curl", orange_color, 1.5)
                        #action = "stop_gesture"
                        is_yoga = True
                        #if prev_action == 'Unknown' or prev_action == "Unknown_First":
                        #    yoga_duration = time.time()
                        #draw_str(image, (20, 50), action, orange_color, 2)
                        logger.debug("STOP")
                        count = 0
                        mode = 0
                        seconds =0
                        first_time = False
                        set_mode = True
        

        # cv2.putText(image,
        #             "FPS: %f" % (1.0 / (time.time() - fps_time)),
        #             (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #             (0, 255, 0), 2)
        #image =   cv2.resize(image, (720,720))
        if(frm==0):
            out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (image.shape[1],image.shape[0]))
            print("Initializing")
            frm+=1
        cv2.imshow('Train Your Angles', image)
        if i != 0:
            out.write(image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
