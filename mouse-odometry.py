import cv2
import numpy as np
import math

# Global variable to store the previous frame for optical flow calculation
prev_frame = None

# Camera specifications
# Field of view in degrees (horizontal, vertical)
field_of_view = (63.3, 49.7)
# Camera height from the ground in meters
cam_height = 0.05
# Camera resolution in pixels (width, height)
cam_resolution = (640, 480)

def calculate_pixel_to_meter_ratio():
    """
    Calculate the conversion ratio from pixels to meters based on camera specifications.
    
    This function uses trigonometry to determine how many meters each pixel represents
    in the image, accounting for camera height and field of view.
    
    Returns:
    - px_to_meter_x: Conversion ratio for x-axis (horizontal)
    - px_to_meter_y: Conversion ratio for y-axis (vertical)
    """
    # Calculate tangent of half the field of view for both x and y axes
    height_tan_x = math.tan(math.radians(field_of_view[0] / 2))
    height_tan_y = math.tan(math.radians(field_of_view[1] / 2))

    # Calculate pixel to meter ratio using camera height and resolution
    # Formula derives from trigonometric relationships in camera geometry
    px_to_meter_x = 2 * cam_height * height_tan_x / cam_resolution[0]
    px_to_meter_y = 2 * cam_height * height_tan_y / cam_resolution[1]

    return px_to_meter_x, px_to_meter_y

# Calculate pixel to meter conversion ratios
px_to_meter_x, px_to_meter_y = calculate_pixel_to_meter_ratio()

def calculate_robot_movement(prev_frame, current_frame):
    """
    Calculate robot movement and rotation using optical flow between two consecutive frames.

    Args:
    - prev_frame: Previous video frame
    - current_frame: Current video frame

    Returns:
    - distance_meters: Total movement distance in meters
    - rotation_radians: chaneg of angle in radians
    """
    # Convert frames to grayscale for optical flow calculation
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow using Farneback method
    # This detects pixel movement between two frames
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, current_gray,
        flow=None,
        pyr_scale=0.5,      # Image scale (<1) for pyramid construction
        levels=3,            # Number of pyramid layers
        winsize=15,          # Averaging window size
        iterations=3,        # Number of iterations at each pyramid level
        poly_n=5,            # Size of pixel neighborhood
        poly_sigma=1.2,      # Gaussian standard deviation for polynomial expansion
        flags=0              # No additional flags
    )

    # Calculate mean displacement of pixels
    displacement = np.mean(flow, axis=(0,1))

    # Convert pixel displacement to meters using previously calculated ratios
    distance_x_meters = displacement[0] * px_to_meter_x
    distance_y_meters = displacement[1] * px_to_meter_y

     # Calculate flow distribution
    # Calculate the average angle of flow vectors
    flow_angles = np.arctan2(flow[:,:,1], flow[:,:,0])

    # Calculate total distance and movement angle
    distance_meters = np.linalg.norm([distance_x_meters, distance_y_meters])
    # Calculate standard deviation of flow angles to estimate rotation
    rotation_radians = np.std(flow_angles)

    return distance_meters, rotation_radians

def runPipeline(image, llrobot):
    """
    Main processing pipeline for robot movement tracking.
    
    Args:
    - image: Current video frame
    - llrobot: Low-level robot parameters (not used in this implementation)
    
    Returns:
    - Processed contours, image, and robot movement parameters
    """
    global prev_frame

    # Handle first frame scenario
    if prev_frame is None:
        prev_frame = image
        return image, [0, 0, 0, 0, 0, 0, 0, 0]

    # Calculate robot movement between previous and current frames
    distance, rotation_radians = calculate_robot_movement(prev_frame, image)

    # Update previous frame for next iteration
    prev_frame = image

    # Find contours in the previous frame
    contours, _ = cv2.findContours(prev_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Prepare low-level python parameters (movement distance and angle)
    llpython = [distance, rotation_radians, 0, 0, 0, 0, 0, 0]

    return contours, image, llpython