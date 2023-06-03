#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Pose, Point
from std_msgs.msg import Header

class ImageProcessingNode:
    def __init__(self):
        rospy.init_node('image_processing_node')
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('image_topic', Image, self.occupancy_grid_callback)
        self.path_pub = rospy.Publisher('path_topic', Path, queue_size=1)
        self.resolution = None # Resolution of the grid [m/cell]
        self.length_x = None  # Length in x-direction [m]
        self.length_y = None  # Length in y-direction [m]
        self.grid_map_center = Pose()  # Pose of the grid map center

    def occupancy_grid_callback(self, msg):
        self.resolution = msg.info.resolution
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_origin = msg.info.origin

        self.path_generator(msg.data)

    def path_generator(self, msg):
        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')

        # Binarize the image
        ret, binary_image = cv2.threshold(cv_image, 127, 255, cv2.THRESH_BINARY)

        # Apply Euclidean distance transform
        dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 3)

        # Skeletonize the image
        skeleton = self.skeletonize(binary_image)

        # Convert skeleton image to path
        path = self.image_to_path(skeleton, msg.header.frame_id)

        # Publish the path
        self.path_pub.publish(path)

    def skeletonize(self, binary_image):
        # Perform skeletonization using an appropriate algorithm
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        size = np.size(binary_image)
        skeleton = np.zeros(binary_image.shape, np.uint8)

        while True:
            eroded = cv2.erode(binary_image, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(binary_image, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            binary_image = eroded.copy()

            zeros = size - cv2.countNonZero(binary_image)
            if zeros == size:
                break

        return skeleton

    def image_to_path(self, image, frame_id):
        path = Path()
        path.header.frame_id = frame_id

        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            for point in contour:
                pose = PoseStamped()
                pose.header.frame_id = frame_id

                # Calculate position using resolution and grid map center
                pose.pose.position.x = point[0][0] * self.resolution + self.grid_map_center.position.x
                pose.pose.position.y = point[0][1] * self.resolution + self.grid_map_center.position.y

                path.poses.append(pose)

        # Set orientation of each pose towards the next point
        for i in range(len(path.poses) - 1):
            curr_pose = path.poses[i].pose
            next_pose = path.poses[i + 1].pose

            orientation = np.arctan2(next_pose.position.y - curr_pose.position.y,
                                     next_pose.position.x - curr_pose.position.x)
            curr_pose.orientation.z = np.sin(orientation / 2.0)
            curr_pose.orientation.w = np.cos(orientation / 2.0)

        # Set the orientation of the last pose to the previous
        path.poses[-1].pose.orientation = path.poses[-2].pose.orientation

        return path

if __name__ == '__main__':
    image_processing_node = ImageProcessingNode()
    rospy.spin()
