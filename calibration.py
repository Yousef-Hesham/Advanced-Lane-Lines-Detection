import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import os.path as path
import pickle


def calibrate_camera(calib_images_dir, testing = False):
    """
    Description: This funtion checks if the calibration file already exists, if so then just return it.
    If not then call the compute calibration function.

    Parameter - calib_images_dir: directory containing chessboard frames
    Parameter - testing: if True, draw and show chessboard corners
    Rreturn   - calibration parameters

    """
    calibration_cache = 'camera_cal/calibration_data.pickle'

    if path.exists(calibration_cache): #Calibration file already exists.
        print('Loading cached camera calibration...',)
        with open(calibration_cache, 'rb') as dump_file:
            calibration = pickle.load(dump_file)
    else:                              #Calibration file dosent exists. it has to be calculated
        print('Computing camera calibration...',)

         # Debugging condition to check for images directory.
        assert path.exists(calib_images_dir), '"{}" must exist and contain calibration images.'.format(calib_images_dir)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros( (6 * 9, 3) , np.float32)         # create 2D array of 6*9 rows and 3 cols with zeros
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) # fill the X,Y cols of the 2D array with their valeus, leave Z as is.

        # Make a list of calibration images
        images = glob.glob(path.join(calib_images_dir, 'calibration*.jpg'))

        # Step through the list and search for chessboard corners
        for filename in images:

            #Reading the current img from our calibration images, then applying grayscale to detect corners.
            img = cv2.imread(filename)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            pattern_found, corners = cv2.findChessboardCorners(gray, (9, 6), None)

            if pattern_found is True:
                imgpoints.append(corners)
                objpoints.append(objp)


                if testing:
                    # Draw and display the corners
                    img = cv2.drawChessboardCorners(img, (9, 6), corners, pattern_found)
                    cv2.imshow('img',img)
                    cv2.waitKey(700)

        if testing:
            cv2.destroyAllWindows()

        # This openCV function returns the distortion coefficient and the camera matrix that we need to transform 3D objectes to 2D.
            #ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        calibration = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        with open(calibration_cache, 'wb') as dump_file:
            pickle.dump(calibration, dump_file)

    print('Done.')
    return calibration


def undistort(frame, mtx, dist, testing = False):
    """
    Description: Undistort a frame given camera matrix and distortion coefficients.

    Parameter - frame: input frame
    Parameter - mtx: camera matrix
    Parameter - dist: distortion coefficients
    Parameter - testing: if True, show frame before/after distortion correction
    Rreturn   - undistorted frame
    """
    frame_undistorted = cv2.undistort(frame, mtx, dist, newCameraMatrix=mtx)

    if testing:
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].set_title('input_frame')
        ax[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        ax[1].set_title('frame_undistorted')
        ax[1].imshow(cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2RGB))
        plt.show()

    return frame_undistorted

# Module Testing code
if __name__ == '__main__':

    ret, mtx, dist, rvecs, tvecs = calibrate_camera(calib_images_dir='camera_cal')

    #img = cv2.imread('test_images/test2.jpg')
    img = cv2.imread('camera_cal/calibration1.jpg')

    img_undistorted = undistort(img, mtx, dist, testing = True)
