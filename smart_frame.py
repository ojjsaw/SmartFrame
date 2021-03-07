
from openvino.inference_engine import IECore
# from matplotlib import pyplot as plt # only for notebook display
import cv2, datetime
import glob

filenames = glob.glob("./result/*.png")
filenames.sort()
photo_frames = [cv2.imread(img) for img in filenames]
length = len(photo_frames)
index = 0

# Plugin Initialization
ie = IECore()
net = ie.read_network(model="./models/intel/pedestrian-detection-adas-0002/FP16/pedestrian-detection-adas-0002.xml", \
                      weights="./models/intel/pedestrian-detection-adas-0002/FP16/pedestrian-detection-adas-0002.bin")

# Input/Output Preparation
input_blob = next(iter(net.input_info))
output_blob = next(iter(net.outputs))
n, c, h, w = net.input_info[input_blob].input_data.shape

# Load the network - replace "CPU" with "MYRIAD" for ncs2
exec_net = ie.load_network(network=net, device_name="MYRIAD")

#cap = cv2.VideoCapture("testnew.mp4") # replace with 0 for live camera
cap = cv2.VideoCapture(0) # replace with 0 for live camera

cv2.namedWindow("Preview", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Preview", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cv2.imshow("Preview", photo_frames[index])
index += 1
cv2.waitKey(1)

while (cap.isOpened()):
    has_frame, orig_frame = cap.read()

    if not has_frame:
        break
    
    in_frame = cv2.resize(orig_frame, (w, h))
    in_frame = in_frame.transpose((2, 0, 1))
    in_frame = in_frame.reshape((n, c, h, w))
    
    results = exec_net.infer(inputs={input_blob: in_frame})
    
    detections = results[output_blob][0][0]
    max_width_of_person = -1
    for detection in detections:
        if detection[2] > 0.95: # threshold
            curr_person_width = detection[5] - detection[3]
            if curr_person_width > max_width_of_person:
                max_width_of_person = curr_person_width

    if max_width_of_person > -1 and max_width_of_person > 0.15:
        cv2.imshow("Preview", photo_frames[index])
        index += 1

    if index >= length:
        index = 0

    # stop on 'q' keyboard input
    if cv2.waitKey(1) & 0xFF == ord('q'):
           break      

cap.release()
cv2.destroyAllWindows()
