#!/usr/bin/env python3

import math
import rospy
import actionlib
import numpy as np
import tf

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal


class PolynomialTaskSpaceNode:
    def __init__(self):
        rospy.init_node("polynomial_task_space_node")

        self.joint_names = ["joint1", "joint2", "joint3", "joint4"]

        self.current_joint_state = None
        rospy.Subscriber("/joint_states", JointState, self.joint_state_callback)

        self.tf_listener = tf.TransformListener()

        self.client = actionlib.SimpleActionClient(
            "/arm_controller/follow_joint_trajectory",
            FollowJointTrajectoryAction
        )

        rospy.loginfo("Waiting for /arm_controller/follow_joint_trajectory...")
        self.client.wait_for_server()
        rospy.loginfo("Connected to arm trajectory controller.")

        # Approximate OpenManipulator-X link lengths [m]
        self.L1 = 0.077
        self.L2 = 0.130
        self.L3 = 0.124
        self.L4 = 0.126

        self.output_csv = "/workspaces/lab2-3-team-3/catkin_ws/src/lab3/data/poly_task_desired.csv"

    def joint_state_callback(self, msg):
        self.current_joint_state = msg

    def get_current_ee_pose(self):
        """
        Read current end-effector position from TF.

        Frame:
            base_link -> gripper_link
        """

        self.tf_listener.waitForTransform(
            "base_link",
            "gripper_link",
            rospy.Time(0),
            rospy.Duration(2.0)
        )

        trans, rot = self.tf_listener.lookupTransform(
            "base_link",
            "gripper_link",
            rospy.Time(0)
        )

        return np.array([trans[0], trans[1], trans[2]])

    def quintic(self, q0, qf, T, t):
        """
        Quintic polynomial trajectory.

        q(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5

        Boundary conditions:
            q(0) = q0
            q(T) = qf
            qdot(0) = qdot(T) = 0
            qddot(0) = qddot(T) = 0

        Coefficients:
            a0 = q0
            a1 = 0
            a2 = 0
            a3 = 10(qf-q0)/T^3
            a4 = -15(qf-q0)/T^4
            a5 = 6(qf-q0)/T^5
        """

        a0 = q0
        a1 = 0.0
        a2 = 0.0
        a3 = 10.0 * (qf - q0) / (T**3)
        a4 = -15.0 * (qf - q0) / (T**4)
        a5 = 6.0 * (qf - q0) / (T**5)

        q = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
        qd = a1 + 2*a2*t + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4
        qdd = 2*a2 + 6*a3*t + 12*a4*t**2 + 20*a5*t**3

        return q, qd, qdd

    def inverse_kinematics(self, x, y, z, tool_pitch=0.0):
        """
        Position-only analytical IK.

        Returns:
            [joint1, joint2, joint3, joint4]
        """

        joint1 = math.atan2(y, x)

        r_total = math.sqrt(x**2 + y**2)

        # Approximate wrist center
        r = r_total - self.L4 * math.cos(tool_pitch)
        z_wrist = z - self.L1 - self.L4 * math.sin(tool_pitch)

        D = math.sqrt(r**2 + z_wrist**2)

        if D > self.L2 + self.L3:
            raise ValueError("Target outside maximum reach.")

        if D < abs(self.L2 - self.L3):
            raise ValueError("Target inside minimum reach.")

        cos_joint3 = (D**2 - self.L2**2 - self.L3**2) / (2.0 * self.L2 * self.L3)
        cos_joint3 = max(min(cos_joint3, 1.0), -1.0)

        joint3 = math.acos(cos_joint3)

        alpha = math.atan2(z_wrist, r)
        beta = math.atan2(
            self.L3 * math.sin(joint3),
            self.L2 + self.L3 * math.cos(joint3)
        )

        joint2 = alpha - beta
        joint4 = tool_pitch - joint2 - joint3

        return np.array([joint1, joint2, joint3, joint4])

    def generate_task_space_trajectory(self, p_start, p_goal, T=4.0, dt=0.05):
        """
        Generate task-space polynomial trajectory.

        Steps:
            1. Generate x(t), y(t), z(t).
            2. Compute IK for each Cartesian point.
            3. Create JointTrajectoryPoint list.
            4. Save desired x,y,z trajectory to CSV.
        """

        points = []
        times = np.arange(0.0, T + dt, dt)

        previous_q = None

        with open(self.output_csv, "w") as f:
            f.write("time,desired_x,desired_y,desired_z,desired_vx,desired_vy,desired_vz\n")

            for t in times:
                desired_pos = []
                desired_vel = []

                for i in range(3):
                    q, qd, qdd = self.quintic(p_start[i], p_goal[i], T, t)
                    desired_pos.append(q)
                    desired_vel.append(qd)

                x, y, z = desired_pos

                q_joints = self.inverse_kinematics(x, y, z)

                if previous_q is None:
                    qd_joints = np.zeros(4)
                else:
                    qd_joints = (q_joints - previous_q) / dt

                previous_q = q_joints.copy()

                point = JointTrajectoryPoint()
                point.positions = q_joints.tolist()
                point.velocities = qd_joints.tolist()
                point.time_from_start = rospy.Duration.from_sec(t)

                points.append(point)

                f.write("{:.4f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}\n".format(
                    t,
                    desired_pos[0],
                    desired_pos[1],
                    desired_pos[2],
                    desired_vel[0],
                    desired_vel[1],
                    desired_vel[2]
                ))

        rospy.loginfo("Saved desired trajectory to %s", self.output_csv)
        return points

    def send_trajectory(self, points):
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = self.joint_names
        goal.trajectory.header.stamp = rospy.Time.now() + rospy.Duration(1.0)
        goal.trajectory.points = points

        rospy.loginfo("Sending task-space polynomial trajectory with %d points.", len(points))
        self.client.send_goal(goal)
        self.client.wait_for_result()
        rospy.loginfo("Trajectory execution complete.")

    def run(self):
        rospy.sleep(1.0)

        p_start = self.get_current_ee_pose()

        rospy.loginfo("Current EE pose: x=%.4f, y=%.4f, z=%.4f",
                      p_start[0], p_start[1], p_start[2])

        # Small safe target close to current pose.
        # Adjust only after first test succeeds.
        p_goal = np.array([0.030, 0.020, 0.210])

        rospy.loginfo("Desired EE goal: x=%.4f, y=%.4f, z=%.4f",
                      p_goal[0], p_goal[1], p_goal[2])

        points = self.generate_task_space_trajectory(
            p_start=p_start,
            p_goal=p_goal,
            T=4.0,
            dt=0.05
        )

        self.send_trajectory(points)


if __name__ == "__main__":
    try:
        node = PolynomialTaskSpaceNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
