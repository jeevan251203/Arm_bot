from ultralytics import YOLO
import cv2
#import tensorflow as tf
from scipy.optimize import minimize
import numpy as np
import math
calib_data_path = r"X:\Computer-vision\RealWorldCoordinates\MultiMatrix.npz"

calib_data = np.load(calib_data_path)
print(calib_data.files)

cmtx = calib_data["camMatrix"]
dist = calib_data["distCoef"]
r_vectors = calib_data["rVector"]
t_vectors = calib_data["tVector"]

model = YOLO("best8.pt")

cap=cv2.VideoCapture("http://192.168.43.109:4747/video")

qr = cv2.QRCodeDetector()

while True:
    ret,frame=cap.read()
    def Boxdetect(frame):
        results = model(frame)
        detection = False
        for i in range(len(results)):
            if len(results[i].boxes) > 0:
                detection = True
                break

        if detection:
            for i in range(len(results)):
                    boxes = results[i].boxes
                    box = boxes[i]
                    conf = box.conf[0].item()
                    if conf>0.8:
                        cords = box.xyxy[0].tolist()
                        cords = [round(x) for x in cords]
                        print(cords)
                        x=cords[0]
                        y=cords[1]
                        w=cords[2]
                        h=cords[3]
                        cv2.rectangle(frame,(x,y),(w,h),(255,0,0),2)
                        center_x = (x + w) / 2
                        center_y = (y + h) / 2
                        return [center_y,center_x]
                    
    def kinematics(x1,y1,z):
        L1= 26.5
        L2 = 13.3
        L3 = 20.7 

        # Define the desired end-effector angle (in radians)
        desired_theta_e = np.pi/2  # Example: 45 degrees

        # Define the joint angle limits (in radians)
        theta1_min, theta1_max = 0, np.pi
        theta2_min, theta2_max = 0, np.pi
        theta3_min, theta3_max = 0, np.pi

        # Function to calculate the forward kinematics
        def forward_kinematics(theta):
            
            y = L1 * np.cos(theta[0]) + L2 * np.cos(theta[0] + theta[1]) + L3 * np.cos(theta[0] + theta[1] + theta[2])
            z = L1 * np.sin(theta[0]) + L2 * np.sin(theta[0] + theta[1]) + L3 * np.sin(theta[0] + theta[1] + theta[2])
            
            return np.array([y, z])

        # Objective function to minimize the error in end-effector position and angle
        def objective_function(theta):
            end_effector_position = forward_kinematics(theta)
            end_effector_angle = theta[0] + theta[1] + theta[2]
            error_position = (end_effector_position[0] - y_target)**2 + (end_effector_position[1] - z_target)**2
            error_angle = (end_effector_angle - desired_theta_e)**2
            return error_position + error_angle

        # Initial guess for joint angles
        initial_guess = [0, 0, 0]  # You can start with different initial angles

        x_target =x 
        y_target =y#y-x,z-y
        z_target= z
        # Set up bounds for joint angles
        bounds = ((theta1_min, theta1_max), (theta2_min, theta2_max), (theta3_min, theta3_max))

        # Perform the optimization to find joint angles
        result = minimize(objective_function, initial_guess, bounds=bounds)
        base_angle=math.atan2(z_target, y_target)

                        
    def detection(frame):

        ret_qr,decoded_info,points,_ = qr.detectAndDecodeMulti(frame)
        #print(points)
        if ret_qr:
            #axis_points, rvec, tvec = get_qr_coords(cmtx, dist, points)
            if points is not None:
                print(points)
            #print(tvec)
                cv2.polylines(frame, points.astype(int), True, (255, 0, 0), 3)           
 
        else:
            kinecord=Boxdetect(frame)
            print(kinecord)
            print(kinecord[0] * 2.54 / 96)


    detection(frame)

    cv2.imshow("frame",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
