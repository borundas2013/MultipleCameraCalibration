import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tqdm



#we considered cheeseboard which have 9 corners vertically and 6 corners horizontally
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
img_ptsL = []
img_ptsR = []
obj_pts = []
pathL = "left/"
pathR = "right/"

#read left and right camera images
#find corners
#draw corners
for i in (range(1,12)): 
    imgL = cv2.imread(pathL+"img%d.png"%i)
    imgR = cv2.imread(pathR+"img%d.png"%i)
    imgL_gray = cv2.imread(pathL+"img%d.png"%i,0)
    imgR_gray = cv2.imread(pathR+"img%d.png"%i,0)

    
    outputL=imgL.copy()
    outputR=imgR.copy()
    
    retR, cornersR =  cv2.findChessboardCorners(outputR,(9,6),None)
    retL, cornersL = cv2.findChessboardCorners(outputL,(9,6),None)

    if  retL and retR:
        obj_pts.append(objp)
        cv2.cornerSubPix(imgR_gray,cornersR,(11,11),(-1,-1),criteria)
        cv2.cornerSubPix(imgL_gray,cornersL,(11,11),(-1,-1),criteria)
        cv2.drawChessboardCorners(outputR,(9,6),cornersR,retR)
        cv2.drawChessboardCorners(outputL,(9,6),cornersL,retL)
        #cv2.imshow('cornersR',outputR)
        #cv2.imshow('cornersL',outputL)
        #cv2.waitKey(0)
        
        img_ptsL.append(cornersL)
        img_ptsR.append(cornersR)
    
   
 
#calibrate left images points   and find optimal points
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(obj_pts,img_ptsL,imgL_gray.shape[::-1],None,None)
hL,wL= imgL_gray.shape[:2]
new_mtxL, roiL= cv2.getOptimalNewCameraMatrix(mtxL,distL,(wL,hL),1,(wL,hL))

#calibrate right images and find optimal points
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(obj_pts,img_ptsR,imgR_gray.shape[::-1],None,None)
hR,wR= imgR_gray.shape[:2]
new_mtxR, roiR= cv2.getOptimalNewCameraMatrix(mtxR,distR,(wR,hR),1,(wR,hR))

print("Optimal Camera matrix:") 
print(new_mtxR) 
 
print("\n Distortion coefficient:") 
print(distR) 
   
print("\n Rotation Vectors:") 
print(rvecsR) 
   
print("\n Translation Vectors:") 
print(tvecsR) 



#combined two new matrix and apply sterep calibrate 
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC

criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(obj_pts,
                                                          img_ptsL,
                                                          img_ptsR,
                                                          new_mtxL,
                                                          distL,
                                                          new_mtxR,
                                                          distR,
                                                          imgL_gray.shape[::-1],
                                                          criteria_stereo,
                                                          flags)




#rectify new matrix
rectify_scale= 1
rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR= cv2.stereoRectify(new_mtxL, distL, new_mtxR, distR,
                                                 imgL_gray.shape[::-1], Rot, Trns,
 
#get undistorted rectify map for left images
Left_Stereo_Map= cv2.initUndistortRectifyMap(new_mtxL, distL, rect_l, proj_mat_l,
                                             imgL_gray.shape[::-1], cv2.CV_16SC2)
#get undistorted rectify map for right images                                             
Right_Stereo_Map= cv2.initUndistortRectifyMap(new_mtxR, distR, rect_r, proj_mat_r,
                                              imgR_gray.shape[::-1], cv2.CV_16SC2)
                                                                         
                                            



#save undistorted parameters for both images
print("Saving paraeters ......")
cv_file = cv2.FileStorage("improved_params2.xml", cv2.FILE_STORAGE_WRITE)
cv_file.write("Left_Stereo_Map_x",Left_Stereo_Map[0])
cv_file.write("Left_Stereo_Map_y",Left_Stereo_Map[1])
cv_file.write("Right_Stereo_Map_x",Right_Stereo_Map[0])
cv_file.write("Right_Stereo_Map_y",Right_Stereo_Map[1])
cv_file.release()




#apply those points in real life scene
CamL= cv2.VideoCapture(0)
CamR= cv2.VideoCapture(1)

print("Reading parameters ......")
cv_file = cv2.FileStorage("improved_params2.xml", cv2.FILE_STORAGE_READ)

Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
cv_file.release()


while True:
	retR, imgR= CamR.read()
	retL, imgL= CamL.read()
	
	if retL and retR:
		imgR_gray = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
		imgL_gray = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)

		Left_nice= cv2.remap(imgL,Left_Stereo_Map_x,Left_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
		Right_nice= cv2.remap(imgR,Right_Stereo_Map_x,Right_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

		output = Right_nice.copy()
		output[:,:,0] = Right_nice[:,:,0]
		output[:,:,1] = Right_nice[:,:,1]
		output[:,:,2] = Left_nice[:,:,2]

		# output = Left_nice+Right_nice
		output = cv2.resize(output,(700,700))
		cv2.namedWindow("3D movie",cv2.WINDOW_NORMAL)
		cv2.imshow("3D movie",output)

		cv2.waitKey(1)
	
	else:
		break
