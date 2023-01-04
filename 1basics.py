#We will use the mediapipe library with opencv which is a cross-platform library developed by Google that provides many 
#ready-to-use ML solutions for computer vision tasks. It has many pre-trained models and it requires minimal resources.

import cv2
import mediapipe as mp                 #The __init__ file of mediapipe package imports all required modules and we just need 
                                       #to mention it as abosolute path in order to use.
import time

p_time = 0
c_time = 0

cap = cv2.VideoCapture(0)

#Set the hands module to a variable (for easy access)
mpHands = mp.solutions.hands

#Set the drawing_utils module to a varaible (for same reason)
mpDraw = mp.solutions.drawing_utils

#Creating an instance of the Hands class, it initialize the MediaPipe Hands object with some parameters but deafult
#works good. To read more on the parameters: (https://google.github.io/mediapipe/solutions/hands#solution-apis)
hands = mpHands.Hands()

while True:
    ret, frame = cap.read()
    if ret:

        h, w, d = frame.shape
#As hands object only processes RGB image we need to convert the frame from BGR to RGB using cvtColor()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#Using process() which takes RGB image array as argument and returns the hand landmarks and handedness of each detected hand.
#Read more:(https://google.github.io/mediapipe/solutions/hands#output)

        result = hands.process(frame_rgb)
#Actually, it returns a namedtuple object (it's a container in collection module), using this we can access values 
#using dot notation like below:

        # print(result.multi_hand_landmarks)          #Detected hands are represented as a list of 21 or less hand landmarks
        # print(result.multi_hand_world_landmarks)    #Same as above, but represented in world coords
        # print(result.multi_handedness)              #Detected hand is classified as left or right hand with conf. score
        
        if result.multi_hand_landmarks:               #Checks if hand landmarks are detected
            
            for hand_landmarks in result.multi_hand_landmarks:
                for id, lm in enumerate(hand_landmarks.landmark):    #Get the normalized coords in the range 0 to 1 for all landmarks
                        
                    cx, cy = int(lm.x * w), int(lm.y * h)        #convert the nrmalized landmark coords to actaul usable 
                                                                 #coords in frame image by multiplying with width & heigh.                       
                
                #ids are the landmarks location, all these locations are of finger tip    
                    if id in [4, 8, 12, 16, 20]:
                        cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)           #draw circle on all finger tips

#Using draw_landmarks() function from drawing_utils module to draw the landmark point on frame image (should be BGR). 
#We are also passing mpHands.HAND_CONNECTIONS to draw the lines between the landmark points.
                mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)


#Get the FPS of the video capture
        c_time = time.time()
        fps = int(1/(c_time - p_time))
        p_time = c_time

#Putting FPS on frame:
        frame = cv2.putText(frame, f"FPS: {fps}", (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ( 0, 255, 0), 1)
        
        cv2.imshow("Video", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    else:
        print("Problem with video frame :(")


cap.release()
cv2.destroyAllWindows()
