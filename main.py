import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from calibration import calibrate_camera, undistort
from thresholding import binarize
from prespective_view import birdeye
from moviepy.editor import VideoFileClip
from line_detection import get_fits_by_sliding_windows, draw_back_onto_the_road, Line, get_fits_by_previous_fits
from globals import xm_per_pix, time_window


#Global Variables
processed_frames = 0                       # counter of frames processed (when processing video)
left_line = Line(buffer_len=time_window)   # line on the left of the lane
right_line = Line(buffer_len=time_window)  # line on the right of the lane

show_processing = True;

def blend (blend_on_road, binary_img, birdeye_img, img_fit, right_line, left_line, offset_meter):
    """
    Description: Prepare the final output blend, given all intermediate pipeline images

    Parameter - blend_on_road: color image of lane blend onto the road
    Parameter - binary_img: thresholded binary image
    Parameter - birdeye_img: bird's eye view of the thresholded binary image
    Parameter - img_fit: bird's eye view with detected lane-lines highlighted
    Parameter - right_line: detected left lane-line
    Parameter - left_line: detected right lane-line
    Parameter - offset_meter: offset from the center of the lane

    Return - blended image with all processing shown.
    """
    h, w = blend_on_road.shape[:2]

    thumb_ratio = 0.2
    thumb_h, thumb_w = int(thumb_ratio * h), int(thumb_ratio * w)

    off_x, off_y = 20, 15

    # add a gray rectangle to highlight the upper area
    mask = blend_on_road.copy()
    mask = cv2.rectangle(mask, pt1=(0, 0), pt2=(w, thumb_h+2*off_y), color=(0, 0, 0), thickness=cv2.FILLED)
    blend_on_road = cv2.addWeighted(src1=mask, alpha=0.2, src2=blend_on_road, beta=0.8, gamma=0)

    # add thumbnail of binary image
    thumb_binary = cv2.resize(binary_img, dsize=(thumb_w, thumb_h))
    thumb_binary = np.dstack([thumb_binary, thumb_binary, thumb_binary]) * 255
    blend_on_road[off_y:thumb_h+off_y, off_x:off_x+thumb_w, :] = thumb_binary

    # add thumbnail of bird's eye view
    thumb_birdeye = cv2.resize(birdeye_img, dsize=(thumb_w, thumb_h))
    thumb_birdeye = np.dstack([thumb_birdeye, thumb_birdeye, thumb_birdeye]) * 255
    blend_on_road[off_y:thumb_h+off_y, 2*off_x+thumb_w:2*(off_x+thumb_w), :] = thumb_birdeye

    # add thumbnail of bird's eye view (lane-line highlighted)
    thumb_img_fit = cv2.resize(img_fit, dsize=(thumb_w, thumb_h))
    blend_on_road[off_y:thumb_h+off_y, 3*off_x+2*thumb_w:3*(off_x+thumb_w), :] = thumb_img_fit

    # add text (curvature and offset info) on the upper right of the blend
    mean_curvature_meter = np.mean([right_line.curvature_meter, left_line.curvature_meter])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blend_on_road, 'Curvature radius: {:.02f}m'.format(mean_curvature_meter), (860, 60), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(blend_on_road, 'Offset from center: {:.02f}m'.format(offset_meter), (860, 130), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return blend_on_road

def compute_offset_from_center(right_line, left_line, frame_width):
    """
    Description: Compute offset from center of the inferred lane.
    The offset from the lane center can be computed under the hypothesis that the camera is fixed
    and mounted in the midpoint of the car roof. In this case, we can approximate the car's deviation
    from the lane center as the distance between the center of the image and the midpoint at the bottom
    of the image of the two lane-lines detected.

    Parameter - right_line: detected left lane-line
    Parameter - left_line: detected right lane-line
    Parameter - frame_width: width of the undistorted frame
    Return - inferred offset
    """
    if right_line.detected and left_line.detected:
        right_line_bottom = np.mean(right_line.all_x[right_line.all_y > 0.95 * right_line.all_y.max()])
        left_line_bottom = np.mean(left_line.all_x[left_line.all_y > 0.95 * left_line.all_y.max()])
        lane_width = left_line_bottom - right_line_bottom
        midpoint = frame_width / 2
        offset_pix = abs((right_line_bottom + lane_width / 2) - midpoint)
        offset_meter = xm_per_pix * offset_pix
    else:
        offset_meter = -1

    return offset_meter


def process_pipeline(frame, keep_state):
    """
    Description: Apply whole lane detection pipeline to an input color frame.
    Parameter - Frame: input color frame
    Parameter - keep_state: if True, lane-line state is conserved (this permits to average results)

    Rreturn - output blend with detected lane overlaid
    """
    global left_line, right_line, processed_frames, show_processing

    # undistort the image using coefficients found in calibration
    undistorted_img = undistort(frame, mtx, dist, testing = False)

    # binarize the frame s.t. lane lines are highlighted as much as possible
    binary_img = binarize(undistorted_img, testing = False)

    # compute perspective transform to obtain bird's eye view
    birdeye_img, M, Minv = birdeye(binary_img, testing = False)

    # fit 2-degree polynomial curve onto lane lines found
    if processed_frames > 0 and keep_state and left_line.detected and right_line.detected:
        left_line, right_line, img_fit = get_fits_by_previous_fits(birdeye_img, left_line, right_line, verbose=False)
    else:
        left_line, right_line, img_fit = get_fits_by_sliding_windows(birdeye_img, left_line, right_line, n_windows=9, verbose=False)

    # compute offset in meter from center of the lane
    offset_meter = compute_offset_from_center(left_line, right_line, frame_width=frame.shape[1])

    # draw the surface enclosed by lane lines back onto the original frame
    blend_on_road = draw_back_onto_the_road(undistorted_img, Minv, left_line, right_line, keep_state)

    processed_frames += 1

    if show_processing:
        # stitch on the top of final output images from different steps of the pipeline
        blend_on_road = blend(blend_on_road, binary_img, birdeye_img, img_fit, left_line, right_line, offset_meter)

    #combined = blend(undistorted_img, binary_img, birdeye_img)
    return blend_on_road

#Main Logic

cap = cv2.VideoCapture("Videos/project_video.mp4")
#cap = cv2.VideoCapture("Videos/toranto.mp4")
if cap.isOpened():
    # get vcap property
    video_width = cap.get(3)   # float
    video_height = cap.get(4) # float

# first things first: calibrate the camera, it is only done once.
ret, mtx, dist, rvecs, tvecs = calibrate_camera(calib_images_dir='camera_cal')

while(cap.isOpened()):
    ret, current_frame = cap.read() #cap.read returns a boolean, and an image [frame] currently displayed.
    if not ret:
        break

    final = process_pipeline(current_frame, True)
    cv2.imshow('processesed.', final)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()
