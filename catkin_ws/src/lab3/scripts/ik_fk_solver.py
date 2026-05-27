#!/usr/bin/env python3

import math
import numpy as np

"""
ME571 Lab 3: OpenManipulator-X IK/FK Solver

Method:
- Position-only analytical IK approximation.
- joint1 controls base yaw.
- joint2, joint3, joint4 solve the planar arm geometry.
- FK is used to verify the IK result.

All distances are in meters.
All joint angles are in radians.
Base frame is treated as the world frame.
End-effector pose is defined with respect to base_link.
"""

class OpenManipulatorIKFK:
    def __init__(self):
        # Approximate OpenManipulator-X link lengths [m]
        self.L1 = 0.077   # base height offset
        self.L2 = 0.130   # shoulder to elbow
        self.L3 = 0.124   # elbow to wrist
        self.L4 = 0.126   # wrist/gripper offset

    def inverse_kinematics(self, x, y, z, tool_pitch=0.0):
        theta1 = math.atan2(y, x)

        r_total = math.sqrt(x**2 + y**2)

        # Approximate wrist center
        r = r_total - self.L4 * math.cos(tool_pitch)
        z_wrist = z - self.L1 - self.L4 * math.sin(tool_pitch)

        D = math.sqrt(r**2 + z_wrist**2)

        if D > self.L2 + self.L3:
            raise ValueError("Target is outside maximum reach.")

        if D < abs(self.L2 - self.L3):
            raise ValueError("Target is inside minimum reach.")

        cos_theta3 = (D**2 - self.L2**2 - self.L3**2) / (2 * self.L2 * self.L3)
        cos_theta3 = max(min(cos_theta3, 1.0), -1.0)

        theta3 = math.acos(cos_theta3)

        alpha = math.atan2(z_wrist, r)
        beta = math.atan2(
            self.L3 * math.sin(theta3),
            self.L2 + self.L3 * math.cos(theta3)
        )

        theta2 = alpha - beta
        theta4 = tool_pitch - theta2 - theta3

        return np.array([theta1, theta2, theta3, theta4])

    def forward_kinematics(self, q):
        theta1, theta2, theta3, theta4 = q

        planar_r = (
            self.L2 * math.cos(theta2)
            + self.L3 * math.cos(theta2 + theta3)
            + self.L4 * math.cos(theta2 + theta3 + theta4)
        )

        z = (
            self.L1
            + self.L2 * math.sin(theta2)
            + self.L3 * math.sin(theta2 + theta3)
            + self.L4 * math.sin(theta2 + theta3 + theta4)
        )

        x = planar_r * math.cos(theta1)
        y = planar_r * math.sin(theta1)

        return np.array([x, y, z])

    def verify(self, target, q):
        fk_pos = self.forward_kinematics(q)
        error = np.linalg.norm(np.array(target) - fk_pos)

        print("Desired pose [x y z] =", np.round(target, 4))
        print("IK joints [rad]      =", np.round(q, 4))
        print("FK pose [x y z]      =", np.round(fk_pos, 4))
        print("Position error [m]   =", round(error, 6))

        return fk_pos, error


if __name__ == "__main__":
    solver = OpenManipulatorIKFK()

    # Use small safe poses close to your verified current gripper pose.
    target_poses = {
        "pose1": [0.03,  0.02, 0.21],
        "pose2": [0.04,  0.04, 0.20],
        "pose3": [0.04, -0.02, 0.22],
    }

    print("pose,desired_x,desired_y,desired_z,joint1,joint2,joint3,joint4,fk_x,fk_y,fk_z,error")

    for name, target in target_poses.items():
        try:
            q = solver.inverse_kinematics(target[0], target[1], target[2])
            fk_pos, error = solver.verify(target, q)

            print("{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.6f}".format(
                name,
                target[0], target[1], target[2],
                q[0], q[1], q[2], q[3],
                fk_pos[0], fk_pos[1], fk_pos[2],
                error
            ))

            print("-" * 70)

        except ValueError as e:
            print(name, "IK failed:", e)