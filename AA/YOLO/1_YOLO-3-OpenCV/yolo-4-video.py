

"""
Course:  Training YOLO v4 for Objects Detection with Custom Data

Section-2
Objects Detection on Video with YOLO v4 and OpenCV
File: yolo-4-video.py
"""


# Detecting Objects on Video with OpenCV deep learning library
#
# Algorithm:
# Reading input video --> Loading YOLO v4 Network -->
# --> Reading frames in the loop --> Getting blob from the frame -->
# --> Implementing Forward Pass --> Getting Bounding Boxes -->
# --> Non-maximum Suppression --> Drawing Bounding Boxes with Labels -->
# --> Writing processed frames
#
# Result:
# New video file with Detected Objects, Bounding Boxes and Labels


# Importing needed libraries
import numpy as np
import cv2
import time


"""
Reading input video
"""

# Defining 'VideoCapture' object and reading video from a file
# Pay attention! If you're using Windows, the path might looks like:
# r'videos\traffic-cars.mp4' or: 'videos\\traffic-cars.mp4'
video = cv2.VideoCapture('videos/traffic-cars-and-people.mp4')

# Preparing variable for writer that we will use to write processed frames
## Se iniciliza a None porque no podemos inicializar sin conocer el ancho y alto de cada frame.
writer = None

# Preparing variables for spatial dimensions of the frames
h, w = None, None





"""
Loading YOLO v4 network
"""

# Loading COCO class labels from file
# Pay attention! If you're using Windows, yours path might looks like:
# r'yolo-coco-data\coco.names' or:  'yolo-coco-data\\coco.names'
with open('yolo-coco-data/coco.names') as f:
    # Getting labels reading every line and putting them into the list
    labels = [line.strip() for line in f] ## Leemos las diferentes clases de COCO

# # Check point
# print('List with labels names:')
# print(labels)

# Loading trained YOLO v4 Objects Detector
# with the help of 'dnn' library from OpenCV
# Pay attention! If you're using Windows, yours paths might look like:
# r'yolo-coco-data\yolov4.cfg' or 'yolo-coco-data\\yolov4.cfg'
# r'yolo-coco-data\yolov4.weights' or yolo-coco-data\\yolov4.weights'
## Cargamos la red ya entrenada de YOLO 4, su configuración y pesos
network = cv2.dnn.readNetFromDarknet('yolo-coco-data/yolov4.cfg',
                                     'yolo-coco-data/yolov4.weights')

# Getting list with names of all layers from YOLO v4 network
layers_names_all = network.getLayerNames()

# # Check point
# print()
# print(layers_names_all)

# Getting only output layers' names that we need from YOLO v4 algorithm
# with function that returns indexes of layers with unconnected outputs
# layers_names_output = \
#     [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]
layers_names_output = []
for i in network.getUnconnectedOutLayers():
    layers_names_output.append(layers_names_all[i - 1])

# # Check point
# print()
# print(layers_names_output)  # ['yolo_82', 'yolo_94', 'yolo_106']

# Setting minimum probability to eliminate weak predictions
probability_minimum = 0.5

# Setting threshold for filtering weak bounding boxes
# with non-maximum suppression
threshold = 0.3

# Generating colours for representing every detected object
# with function randint(low, high=None, size=None, dtype='l')
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

# # Check point
# print()
# print(type(colours))  # <class 'numpy.ndarray'>
# print(colours.shape)  # (80, 3)
# print(colours[0])  # [172  10 127]



"""
Reading frames in the loop
"""

# Defining variable for counting frames
# At the end we will show total amount of processed frames
f = 0

# Defining variable for counting total time
# At the end we will show time spent for processing all frames
t = 0

# Defining loop for catching frames
while True:
    # Capturing frame-by-frame
    ## Nos devuelve en ret un boolean de éxito (true) o no (false) en la lectura del frame
    ## y en frame la lectura del frame correspondiente que vamos a procesar.
    ret, frame = video.read()

    # If the frame was not retrieved then we break the loop
    ## Por ejemplo cuando hemos terminado de leer el video
    if not ret:
        break

    # Getting spatial dimensions of the fram, we do it only once from the very beginning
    ## Se hace una única vez porque todos los frames se presupone que tienen las mismas dimensiones
    if w is None or h is None:
        # Slicing from tuple only first two elements
        h, w = frame.shape[:2]




    """
    Getting blob from current frame
    """

    # Getting blob from current frame
    # The 'cv2.dnn.blobFromImage' function returns 4-dimensional blob from current
    # frame after mean subtraction, normalizing, and RB channels swapping
    # Resulted shape has number of frames, number of channels, width and height
    # E.G.: blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)
    ## Obtenemos un tensor de 4D con el número de frames, número de canales, ancho y alto.
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)





    """
    Implementing Forward pass
    """

    # Implementing forward pass with our blob and only through output layers
    # Calculating at the same time, needed time for forward pass
    network.setInput(blob)  # setting blob as input to the network
    start = time.time()
    ## Mediante la función forward obtenemos los resultados unicamentre de los 'layers' especificados.
    output_from_network = network.forward(layers_names_output)
    end = time.time()

    # Increasing counters for frames and total time
    f += 1
    t += end - start

    # Showing spent time for single current frame
    print('Frame number {0} took {1:.5f} seconds'.format(f, end - start))




    """
    Getting bounding boxes
    """

    # Preparing lists for detected bounding boxes,
    # obtained confidences and class's number
    bounding_boxes = []
    confidences = []
    class_numbers = []

    # Going through all output layers after feed forward pass
    for result in output_from_network:
        # Going through all detections from current output layer
        for detected_objects in result:
            # Getting 80 classes' probabilities for current detected object
            scores = detected_objects[5:]
            # Getting index of the class with the maximum value of probability
            class_current = np.argmax(scores)
            # Getting value of probability for defined class
            confidence_current = scores[class_current]

            # # Check point
            # # Every 'detected_objects' numpy array has first 4 numbers with
            # # bounding box coordinates and rest 80 with probabilities
            #  # for every class
            # print(detected_objects.shape)  # (85,)

            # Eliminating weak predictions with minimum probability
            if confidence_current > probability_minimum:
                # Scaling bounding box coordinates to the initial frame size
                # YOLO data format keeps coordinates for center of bounding box
                # and its current width and height
                # That is why we can just multiply them elementwise
                # to the width and height
                # of the original frame and in this way get coordinates for center
                # of bounding box, its width and height for original frame
                ## Los dos primeros parámetros de detected objects son los centros X e Y
                ## y los dos últimos el ancho y alto, que lo multiplicamos por el array para
                ## escalarlos al tamaño original tras el blob.
                box_current = detected_objects[0:4] * np.array([w, h, w, h])

                # Now, from YOLO data format, we can get top left corner coordinates
                # that are x_min and y_min
                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))

                # Adding results into prepared lists
                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)





    """
    Non-maximum suppression
    """

    # Implementing non-maximum suppression of given bounding boxes
    # With this technique we exclude some of bounding boxes if their
    # corresponding confidences are low or there is another
    # bounding box for this region with higher confidence

    # It is needed to make sure that data type of the boxes is 'int'
    # and data type of the confidences is 'float'
    # https://github.com/opencv/opencv/issues/12789v

    ## Aplicando el algoritmo de Supresión No Máxima excluimos los recuadros que no cumplan
    ## con los parámetros de probabilidad y threshold que hayamos indicado. Además si encuentra
    ## más de una clase que coincida con un objeto, se quedará con aquella con mayor probabilidad.
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)


    """
    Drawing bounding boxes and labels
    """

    # Checking if there is at least one detected object
    # after non-maximum suppression
    if len(results) > 0:
        # Going through indexes of results
        for i in results.flatten():
            # Getting current bounding box coordinates,
            # its width and height
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

            # Preparing colour for current bounding box
            # and converting from numpy array to list
            colour_box_current = colours[class_numbers[i]].tolist()

            # # # Check point
            # print(type(colour_box_current))  # <class 'list'>
            # print(colour_box_current)  # [172 , 10, 127]

            # Drawing bounding box on the original current frame
            cv2.rectangle(frame, (x_min, y_min),  (x_min + box_width, y_min + box_height), colour_box_current, 2)

            # Preparing text with label and confidence for current bounding box
            text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])], confidences[i])

            # Putting text with label and confidence on the original image
            cv2.putText(frame, text_box_current, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)


    """
    Writing processed frame into the file
    """

    # Initializing writer
    # we do it only once from the very beginning
    # when we get spatial dimensions of the frames
    if writer is None:
        # Constructing code of the codec to be used in the function VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Writing current processed frame into the video file
        # Pay attention! If you're using Windows, yours path might looks like:
        # r'videos\result-traffic-cars.mp4' or: 'videos\\result-traffic-cars.mp4'
        writer = cv2.VideoWriter('videos/result-traffic-cars-and-people.mp4', fourcc, 30, (frame.shape[1], frame.shape[0]), True)

    # Write processed current frame to the file
    writer.write(frame)






# Printing final results
print()
print('Total number of frames', f)
print('Total amount of time {:.5f} seconds'.format(t))
print('FPS:', round((f / t), 1))


# Releasing video reader and writer
video.release()
writer.release()


"""
Some comments

What is a FOURCC?
    FOURCC is short for "four character code" - an identifier for a video codec,
    compression format, colour or pixel format used in media files.
    http://www.fourcc.org


Parameters for cv2.VideoWriter():
    filename - Name of the output video file.
    fourcc - 4-character code of codec used to compress the frames.
    fps	- Frame rate of the created video.
    frameSize - Size of the video frames.
    isColor	- If it True, the encoder will expect and encode colour frames.
"""
