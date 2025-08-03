import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import math

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

bpoints = [deque(maxlen=1024)]  
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]
ppoints = [deque(maxlen=1024)]  # Purple
opoints = [deque(maxlen=1024)]  # Orange
cpoints = [deque(maxlen=1024)]  # Cyan
pinkpoints = [deque(maxlen=1024)]  # Pink

blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0
purple_index = 0
orange_index = 0
cyan_index = 0
pink_index = 0

kernel = np.ones((5, 5), np.uint8)

colors = [
    (255, 0, 0),     # Blue
    (0, 255, 0),     # Green
    (0, 0, 255),     # Red
    (0, 255, 255),   # Yellow
    (255, 0, 255),   # Purple
    (0, 165, 255),   # Orange
    (255, 255, 0),   # Cyan
    (203, 192, 255)  # Pink
]
color_names = ["BLUE", "GREEN", "RED", "YELLOW", "PURPLE", "ORANGE", "CYAN", "PINK"]
colorIndex = 0

brush_size = 5  
eraser_mode = False
canvas_color = (255, 255, 255)  
def draw_color_palette(frame, selected_color):
    global brush_size, eraser_mode
    
    palette_height = 100
    color_radius = 30
    spacing = 20
    start_x = spacing
    start_y = spacing
    
    # palette background
    cv2.rectangle(frame, (0, 0), (WINDOW_WIDTH, palette_height), (240, 240, 240), -1)
    cv2.line(frame, (0, palette_height), (WINDOW_WIDTH, palette_height), (200, 200, 200), 2)
    
    # color circles
    for i, color in enumerate(colors):
        center_x = start_x + i * (color_radius * 2 + spacing) + color_radius
        center_y = start_y + color_radius
        
        # outer circle if selected
        if i == selected_color and not eraser_mode:
            cv2.circle(frame, (center_x, center_y), color_radius + 5, (255, 255, 255), 2)
        
        # Draw color circle
        cv2.circle(frame, (center_x, center_y), color_radius, color, -1)
        
        # color name
        text_size = cv2.getTextSize(color_names[i], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = center_x - text_size[0] // 2
        cv2.putText(frame, color_names[i], (text_x, center_y + color_radius + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    # brush size slider
    slider_start_x = start_x + len(colors) * (color_radius * 2 + spacing) + spacing
    slider_width = 200
    slider_height = 20
    slider_y = start_y + color_radius - slider_height // 2
    
    # slider background
    cv2.rectangle(frame, (slider_start_x, slider_y), 
                 (slider_start_x + slider_width, slider_y + slider_height), (200, 200, 200), -1)
    
    # slider position
    position = int(slider_start_x + (brush_size / 25) * slider_width)
    cv2.circle(frame, (position, slider_y + slider_height // 2), 10, (0, 0, 0), -1)
    
    cv2.putText(frame, f"Brush Size: {brush_size}", (slider_start_x, slider_y - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    
    # eraser button
    eraser_x = slider_start_x + slider_width + spacing
    eraser_y = start_y + color_radius - 15
    cv2.rectangle(frame, (eraser_x, eraser_y), (eraser_x + 100, eraser_y + 30), 
                 (50, 50, 50) if eraser_mode else (200, 200, 200), -1)
    cv2.putText(frame, "ERASER", (eraser_x + 15, eraser_y + 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255) if eraser_mode else (0, 0, 0), 1, cv2.LINE_AA)
    
    # clear button
    clear_x = eraser_x + 120
    clear_y = eraser_y
    cv2.rectangle(frame, (clear_x, clear_y), (clear_x + 100, clear_y + 30), (240, 100, 100), -1)
    cv2.putText(frame, "CLEAR ALL", (clear_x + 10, clear_y + 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    # save button
    save_x = clear_x + 120
    save_y = clear_y
    cv2.rectangle(frame, (save_x, save_y), (save_x + 100, save_y + 30), (100, 240, 100), -1)
    cv2.putText(frame, "SAVE", (save_x + 30, save_y + 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    return frame

def draw_brush_preview(frame, position, color, size):
    cv2.circle(frame, position, size, color, -1)
    cv2.circle(frame, position, size, (0, 0, 0), 1)  # Outline
    return frame

def is_point_in_rect(point, rect_start, rect_end):
    return rect_start[0] <= point[0] <= rect_end[0] and rect_start[1] <= point[1] <= rect_end[1]

def draw_quadratic_bezier(img, p0, p1, p2, color, thickness):
    for t in np.linspace(0, 1, 20):  # Adjust number of steps for smoothness
        x = int((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0])
        y = int((1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1])
        cv2.circle(img, (x, y), thickness, color, -1)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

canvas = np.ones((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8) * 255

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)

cursor_visible = True
cursor_timer = 0
is_drawing = False
snap_to_shape_mode = False
shape_points = []
snap_threshold = 50  # distance to consider the shape closed
previous_finger_position = None
finger_up = True
save_counter = 0

status_message = "Welcome to Air Canvas! Move your hand to draw."
status_timer = 0

ret = True
while ret:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
    
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    frame = draw_color_palette(frame, colorIndex)
    
    canvas_height = frame.shape[0] - 100
    canvas_width = frame.shape[1]
    if canvas.shape[:2] != (canvas_height, canvas_width):
      # Resize canvas to match the required dimensions
      canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    frame[100:, :] = cv2.addWeighted(frame[100:, :], 0.3, canvas[:frame.shape[0]-100, :frame.shape[1]], 0.7, 0)    
    # Get hand landmark prediction
    result = hands.process(framergb)
    
    if status_timer > 0:
        cv2.rectangle(frame, (WINDOW_WIDTH // 2 - 200, WINDOW_HEIGHT - 50), 
                     (WINDOW_WIDTH // 2 + 200, WINDOW_HEIGHT - 20), (0, 0, 0, 100), -1)
        cv2.putText(frame, status_message, (WINDOW_WIDTH // 2 - 190, WINDOW_HEIGHT - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        status_timer -= 1
    
    index_finger_tip = None
    middle_finger_tip = None
    thumb_tip = None
    ring_finger_tip = None


    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * WINDOW_WIDTH)
                lmy = int(lm.y * WINDOW_HEIGHT)
                landmarks.append([lmx, lmy])
            
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS,
                                 mpDraw.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
                                 mpDraw.DrawingSpec(color=(0, 153, 255), thickness=1, circle_radius=1))
        
        index_finger_tip = (landmarks[8][0], landmarks[8][1])
        middle_finger_tip = (landmarks[12][0], landmarks[12][1])
        thumb_tip = (landmarks[4][0], landmarks[4][1])


        index_middle_distance = math.sqrt((index_finger_tip[0] - middle_finger_tip[0])**2 + 
                                          (index_finger_tip[1] - middle_finger_tip[1])**2)

        ring_finger_tip = landmarks[16]

        three_fingers_up = (
            index_finger_tip[1] < landmarks[6][1] and
            middle_finger_tip[1] < landmarks[10][1] and
            ring_finger_tip[1] < landmarks[14][1]
        )

        if three_fingers_up:
            if not snap_to_shape_mode:
                snap_to_shape_mode = True
                status_message = "Snap-to-shape: ON"
                status_timer = 30
        else:
            if snap_to_shape_mode:
                snap_to_shape_mode = False
                status_message = "Snap-to-shape: OFF"
                status_timer = 30

        cursor_visible = True
        cursor_timer = 10

        if index_finger_tip[1] < 100:
            for i in range(len(colors)):
                center_x = 20 + i * (60 + 20) + 30
                center_y = 20 + 30
                distance = math.sqrt((index_finger_tip[0] - center_x)**2 + (index_finger_tip[1] - center_y)**2)
                if distance < 30:
                    colorIndex = i
                    eraser_mode = False
                    status_message = f"Selected color: {color_names[i]}"
                    status_timer = 30

            slider_start_x = 20 + len(colors) * (60 + 20) + 20
            slider_width = 200
            slider_y = 20 + 30 - 10
            if is_point_in_rect(index_finger_tip, (slider_start_x, slider_y), 
                                (slider_start_x + slider_width, slider_y + 20)):
                rel_x = index_finger_tip[0] - slider_start_x
                brush_size = max(1, min(25, int((rel_x / slider_width) * 25)))
                status_message = f"Brush size: {brush_size}"
                status_timer = 30

            eraser_x = slider_start_x + slider_width + 20
            eraser_y = 20 + 30 - 15
            if is_point_in_rect(index_finger_tip, (eraser_x, eraser_y), (eraser_x + 100, eraser_y + 30)):
                eraser_mode = not eraser_mode
                status_message = "Eraser mode: " + ("ON" if eraser_mode else "OFF")
                status_timer = 30

            clear_x = eraser_x + 120
            clear_y = eraser_y
            if is_point_in_rect(index_finger_tip, (clear_x, clear_y), (clear_x + 100, clear_y + 30)):
                canvas = np.ones((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8) * 255
                status_message = "Canvas cleared!"
                status_timer = 30

            save_x = clear_x + 120
            save_y = clear_y
            if is_point_in_rect(index_finger_tip, (save_x, save_y), (save_x + 100, save_y + 30)):
                save_filename = f"air_canvas_{save_counter}.png"
                cv2.imwrite(save_filename, canvas)
                status_message = f"Saved as {save_filename}"
                status_timer = 60
                save_counter += 1

            # Reset drawing state
            is_drawing = False
            previous_finger_position = None

        else:  # Drawing below color palette area
            current_color = (255, 255, 255) if eraser_mode else colors[colorIndex]
            draw_brush_preview(frame, index_finger_tip, current_color, brush_size)

            if previous_finger_position is not None:
                if snap_to_shape_mode:
                    shape_points.append(index_finger_tip)
                else:
                    shape_points = []

                p0 = previous_finger_position
                p2 = index_finger_tip
                p1 = ((p0[0] + p2[0]) // 2, (p0[1] + p2[1]) // 2)

                if eraser_mode:
                    draw_quadratic_bezier(canvas, p0, p1, p2, (255, 255, 255), brush_size * 2)
                else:
                    draw_quadratic_bezier(canvas, p0, p1, p2, colors[colorIndex], brush_size)

            previous_finger_position = index_finger_tip
            is_drawing = True

            # Detect closed shape and snap to polygon
            if snap_to_shape_mode and len(shape_points) >= 3:
                dist = math.hypot(shape_points[0][0] - shape_points[-1][0],
                                  shape_points[0][1] - shape_points[-1][1])
                if dist < snap_threshold:
                    # Erase user sketch
                    for i in range(len(shape_points) - 1):
                        cv2.line(canvas, shape_points[i], shape_points[i + 1], (255, 255, 255), brush_size * 2)
                    cv2.line(canvas, shape_points[-1], shape_points[0], (255, 255, 255), brush_size * 2)

                    # Approximate and redraw as clean shape
                    approx = cv2.approxPolyDP(np.array(shape_points), epsilon=5.0, closed=True)
                    for i in range(len(approx)):
                        pt1 = tuple(approx[i][0])
                        pt2 = tuple(approx[(i + 1) % len(approx)][0])
                        draw_color = (255, 255, 255) if eraser_mode else colors[colorIndex]
                        cv2.line(canvas, pt1, pt2, draw_color, brush_size)

                    status_message = f"Shape snapped with {len(approx)} sides"
                    status_timer = 30
                    shape_points.clear()
    else:
        previous_finger_position = None
        is_drawing = False

        if index_finger_tip is not None and thumb_tip is not None:

            thumb_index_distance = math.hypot(index_finger_tip[0] - thumb_tip[0], 
                                          index_finger_tip[1] - thumb_tip[1])
            if thumb_index_distance < 30 and cursor_timer <= 0:
                cursor_visible = not cursor_visible
                cursor_timer = 30
                status_message = "Cursor visibility toggled"
                status_timer = 30
    
            if snap_to_shape_mode and len(shape_points) >= 3:
                dist = math.hypot(shape_points[0][0] - shape_points[-1][0],
                              shape_points[0][1] - shape_points[-1][1])
                if dist < snap_threshold:
                    # Shape is closed
                    approx = cv2.approxPolyDP(np.array(shape_points, dtype=np.int32), 0.04 * cv2.arcLength(np.array(shape_points, dtype=np.int32), True), True)
                    for i in range(len(approx)):
                        pt1 = tuple(approx[i][0])
                        pt2 = tuple(approx[(i + 1) % len(approx)][0])
                        color = (255, 255, 255) if eraser_mode else colors[colorIndex]
                        cv2.line(canvas, pt1, pt2, color, brush_size)
                shape_points = []
   
    if cursor_timer > 0:
        cursor_timer -= 1
    
    if is_drawing:
        cv2.putText(frame, "DRAWING", (WINDOW_WIDTH - 150, WINDOW_HEIGHT - 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    
    tool_info = f"Tool: {'Eraser' if eraser_mode else color_names[colorIndex]} | Size: {brush_size}"
    cv2.putText(frame, tool_info, (10, WINDOW_HEIGHT - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    
    snap_status = "ON" if snap_to_shape_mode else "OFF"
    cv2.putText(frame, f"Snap-to-Shape: {snap_status}", (10, WINDOW_HEIGHT - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)


    cv2.putText(frame, "Index finger to draw, pinch to toggle cursor", (WINDOW_WIDTH // 2 - 200, 130), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    
    cv2.imshow("Air Canvas", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.ones((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8) * 255
        status_message = "Canvas cleared!"
        status_timer = 30
    elif key == ord('s'):
        save_filename = f"air_canvas_{save_counter}.png"
        cv2.imwrite(save_filename, canvas)
        status_message = f"Saved as {save_filename}"
        status_timer = 60
        save_counter += 1

cap.release()
cv2.destroyAllWindows()                                            