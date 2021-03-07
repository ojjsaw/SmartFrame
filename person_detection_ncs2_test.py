
from openvino.inference_engine import IECore
# from matplotlib import pyplot as plt # only for notebook display
import cv2, datetime

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

cap = cv2.VideoCapture("testnew.mp4") # replace with 0 for live camera
#cap = cv2.VideoCapture(0) # replace with 0 for live camera

while (cap.isOpened()):
    has_frame, orig_frame = cap.read()
    
    if not has_frame:
        break
    
    in_frame = cv2.resize(orig_frame, (w, h))
    in_frame = in_frame.transpose((2, 0, 1))
    in_frame = in_frame.reshape((n, c, h, w))
    
    start_time = datetime.datetime.now()
    results = exec_net.infer(inputs={input_blob: in_frame})
    end_time = datetime.datetime.now()
    
    detections = results[output_blob][0][0]
    for detection in detections:
        if detection[2] > 0.90: # threshold
            xmin = int(detection[3] * orig_frame.shape[1]) # shape[1] - width
            ymin = int(detection[4] * orig_frame.shape[0]) # shape[0] - height
            xmax = int(detection[5] * orig_frame.shape[1])
            ymax = int(detection[6] * orig_frame.shape[0])
            cv2.rectangle(orig_frame, (xmin, ymin), (xmax, ymax), (0, 125, 255), 3)
    
    time_st = 'NCS2 Inference time: {:.2f} ms'.format(round((end_time - start_time).total_seconds() * 1000), 2)
    cv2.putText(orig_frame,time_st, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 125, 255), 6)
    #cv2.imwrite("result.png", orig_frame)
    
    # # cv2.imshow not compatible in notebook, use matplotlib
    # orig_frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
    # plt.imshow(orig_frame)
    # plt.show()
    # break # only show one frame and stop expanding cell output in notebook
    
    cv2.imshow("Output ", orig_frame)
    # stop on 'q' keyboard input
    if cv2.waitKey(1) & 0xFF == ord('q'):
           break      

cap.release()
cv2.destroyAllWindows()
