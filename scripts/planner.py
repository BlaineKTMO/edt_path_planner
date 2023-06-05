#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from nav_msgs.msg import Path, OccupancyGrid
import itertools as it
import math
from geometry_msgs.msg import PoseStamped, Pose, Point
from std_msgs.msg import Header

COST_THRESHOLD = 0.6


class ImageProcessingNode:
    def __init__(self):
        rospy.init_node('image_processing_node')
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/composite_costmap/costmap/costmap', OccupancyGrid, self.occupancy_grid_callback)
        self.path_pub = rospy.Publisher('/path', Path, queue_size=1)
        self.rate = rospy.Rate(1)
        self.resolution = None  # Resolution of the grid [m/cell]
        self.length_x = None  # Length in x-direction [m]
        self.length_y = None  # Length in y-direction [m]
        self.grid_map_center = Pose()  # Pose of the grid map center

    def occupancy_grid_callback(self, msg):
        self.resolution = msg.info.resolution
        self.map_width = round(msg.info.width)
        self.map_height = round(msg.info.height)
        self.map_origin = msg.info.origin
        self.frame_id = "base_link"
        print(self.frame_id)

        print(msg.info.width)
        print(msg.info.height)
        print()
        print(self.map_width)
        print(self.map_height)

        image = np.zeros((self.map_height, self.map_width, 3), dtype=np.uint8)
        for y in range(self.map_height):
            for x in range(self.map_width):
                index = y * self.map_width + x

                # print(index)
                cost = msg.data[index]
                if cost >= COST_THRESHOLD:
                    color = (0, 0, 0)
                else:
                    color = (255, 255, 255)

                image[self.map_height - x - 1, self.map_width - y - 1] = color

        self.path_generator(image)

    def path_generator(self, msg):

        # Convert ROS Image message to OpenCV image
        # cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')

        img = np.copy(msg)

        # cv2.imwrite("img.jpg", msg)
        # cv2.imshow("image", msg)
        # cv2.waitKey()

        # Binarize the image
        # ret, binary_image = cv2.threshold(msg, 127, 255, cv2.THRESH_BINARY)

        # Grey the image
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Euclidean distance transform
        dist_transform = cv2.distanceTransform(grayimg, cv2.DIST_C, 3, dstType=cv2.CV_8U)
        normalized = cv2.normalize(dist_transform, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

        # Skeletonize the image
        skeleton = self.skeletonize(normalized)
        # skeleton_gray = cv2.cvtColor(skeleton, cv2.COLOR_BAYER_BG2GRAY)

        cv2.imshow("skeleton", skeleton)
        # cv2.waitKey(0)

        # Convert skeleton image to path
        path = self.image_to_path(skeleton)

        # Publish the path
        self.path_pub.publish(path)

        self.rate.sleep()

    def skeletonize(self, binary_image):
        # Perform skeletonization using an appropriate algorithm
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        size = np.size(binary_image)
        skeleton = np.zeros(binary_image.shape, np.uint8)

        while True:
            eroded = cv2.erode(binary_image, element)
            temp = cv2.dilate(eroded, element)
            # temp = cv2.normalize(temp, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
            temp = cv2.subtract(binary_image, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            binary_image = eroded.copy()

            zeros = size - cv2.countNonZero(binary_image)
            if zeros == size:
                break

        return skeleton

    def distance(self, x, y):
        return math.hypot(y[0] - x[0], y[1] - x[1])

    def find_minimum_hamiltonian_path(self, path_msg):
        print("starting calculations")
        x_coords = [pose.pose.position.x for pose in path_msg.poses]
        y_coords = [pose.pose.position.y for pose in path_msg.poses]
        coords = [(x, y) for x, y in zip(x_coords, y_coords)]

        permutations = it.permutations(coords)

        print(permutations)

        # Calculate the length of each path and choose the shortest one
        shortest_path = min(permutations, key=lambda path: sum(self.distance(path[i], path[i + 1]) for i in range(len(path) - 1)))

        return shortest_path

    def image_to_path(self, image):
        path = Path()
        path.header.frame_id = self.frame_id

        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        longest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(image, longest_contour, -1, (255, 0, 255))
        cv2.imshow("contours", image)
        cv2.waitKey()

        for point in longest_contour:
            pose = PoseStamped()
            pose.header.frame_id = self.frame_id

            # Calculate position using resolution and grid map center
            print(self.grid_map_center.position)
            pose.pose.position.y = point[0][1] * self.resolution
            pose.pose.position.x = point[0][0] * self.resolution

            path.poses.append(pose)

        # self.find_minimum_hamiltonian_path(path)

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
