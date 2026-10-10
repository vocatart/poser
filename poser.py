import time

import cv2
import numpy as np
from mediapipe.tasks.python import vision, BaseOptions
import mediapipe


def kinect_v1_skeleton(data):
    points = lambda *pts: np.mean([[data[p].x, data[p].y] for p in pts], axis=0)

    return {
        "kJointHipCenter": points(23, 24),
        "kJointShoulderCenter": points(11, 12),
        "kJointSpine": points(23, 24) + 0.25 * (points(11, 12) - points(23, 24)),
        "kJointHead": points(7, 8),
        "kJointShoulderLeft": points(11),
        "kJointElbowLeft": points(13),
        "kJointWristLeft": points(15),
        "kJointHandLeft": points(17, 19),
        "kJointShoulderRight": points(12),
        "kJointElbowRight": points(14),
        "kJointWristRight": points(16),
        "kJointHandRight": points(18, 20),
        "kJointHipLeft": points(23),
        "kJointKneeLeft": points(25),
        "kJointAnkleLeft": points(27),
        "kJointHipRight": points(24),
        "kJointKneeRight": points(26),
        "kJointAnkleRight": points(28),
        "kJointFootLeft": points(31),
        "kJointFootRight": points(32),
    }

kinect_v1_bones = {
    "kBoneHead": ["kJointHead", "kJointShoulderCenter"],
    "kBoneCollarRight": ["kJointShoulderCenter", "kJointShoulderRight"],
    "kBoneArmUpperRight": ["kJointShoulderRight", "kJointElbowRight"],
    "kBoneArmLowerRight": ["kJointElbowRight", "kJointWristRight"],
    "kBoneHandRight": ["kJointWristRight", "kJointHandRight"],
    "kBoneCollarLeft": ["kJointShoulderCenter", "kJointShoulderLeft"],
    "kBoneArmUpperLeft": ["kJointShoulderLeft", "kJointElbowLeft"],
    "kBoneArmLowerLeft": ["kJointElbowLeft", "kJointWristLeft"],
    "kBoneHandLeft": ["kJointWristLeft", "kJointHandLeft"],
    "kBoneLegUpperRight": ["kJointHipRight", "kJointKneeRight"],
    "kBoneLegLowerRight": ["kJointKneeRight", "kJointAnkleRight"],
    "kBoneLegUpperLeft": ["kJointHipLeft", "kJointKneeLeft"],
    "kBoneLegLowerLeft": ["kJointKneeLeft", "kJointAnkleLeft"],
    "kBoneBackUpper": ["kJointSpine", "kJointShoulderCenter"],
    "kBoneBackLower": ["kJointSpine", "kJointHipCenter"],
    "kBoneHipRight": ["kJointHipCenter", "kJointHipRight"],
    "kBoneHipLeft": ["kJointHipCenter", "kJointHipLeft"],
    "kBoneFootLeft": ["kJointAnkleLeft", "kJointFootLeft"],
    "kBoneFootRight": ["kJointAnkleRight", "kJointFootRight"]
}

if __name__ == "__main__":
    landmarker = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="pose_landmarker_heavy.task"),
        running_mode=vision.RunningMode.VIDEO)
    )

    cap = cv2.VideoCapture(0)

    last_ts = 0
    cv2.namedWindow("poser", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("poser", 1280, 720)

    while cv2.waitKey(1) != ord("q"):
        ok, frame = cap.read()
        if not ok:
            exit(0)

        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        image = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        last_ts = max(int(time.monotonic() * 1000), last_ts + 1)
        result = landmarker.detect_for_video(image, last_ts)

        canvas = np.zeros_like(frame)

        for skeleton in result.pose_landmarks:
            joints = {k: (int(x * width), int(y * height)) for k, (x, y) in kinect_v1_skeleton(skeleton).items()}

            for jointName, joint in joints.items():
                cv2.circle(canvas, joint, 5, (255, 255, 255), -1)
                cv2.putText(canvas, jointName, (joint[0] + 10, joint[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

            for boneName, (joint1, joint2) in kinect_v1_bones.items():
                cv2.line(canvas, joints[joint1], joints[joint2], (0, 255, 0), 3)


        cv2.imshow("poser", canvas)

    cap.release()
    cv2.destroyAllWindows()