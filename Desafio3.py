#!/usr/bin/env python

import sys
import argparse
import pyglet
from pyglet.window import key
import numpy as np
import gym
import math
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper
import cv2


import time

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default="Duckietown-udem1-v1")
parser.add_argument('--map-name', default='udem1')
parser.add_argument('--distortion', default=False, action='store_true')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
parser.add_argument('--seed', default=1, type=int, help='seed')
args = parser.parse_args()

# Parametros para el detector de lineas blancas
white_filter_1 = np.array([0, 0, 105]) 
white_filter_2 = np.array([185, 46, 254])

# Filtros para el detector de lineas amarillas
yellow_filter_1 = np.array([22, 68, 132]) 
yellow_filter_2 = np.array([31, 255, 255])
window_filter_name = "filtro"
#Colores RGB#
white_RGB = (255, 255, 255)
yellow_RGB = (0, 255, 255)

# Constantes
DUCKIE_MIN_AREA = 0 #editar esto si es necesario
RED_LINE_MIN_AREA = 0 #editar esto si es necesario
RED_COLOR = (0,0,255)
MAX_DELAY = 20
MAX_TOLERANCE = 200
AREA_MAX = 500 
MAX_HISTORY_SIZE = 10  


# Variables globales
duckie_area = 0 
red_line_detected = False
red_line_stop_time = 0
red_line_stop_duration = 6 

Pix_frame = (640,480) 
Reference_point = (Pix_frame[0] // 2 , Pix_frame[1])


if args.env_name and args.env_name.find('Duckietown') != -1:
    env = DuckietownEnv(
        seed=args.seed,
        map_name=args.map_name,
        draw_curve=args.draw_curve,
        draw_bbox=args.draw_bbox,
        domain_rand=args.domain_rand,
        frame_skip=args.frame_skip,
        distortion=args.distortion,
    )
else:
    env = gym.make(args.env_name)

env.reset()
env.render(mode="top_down")


# Funciones interesantes para hacer operaciones interesantes
def box_area(box):
    return abs(box[2][0] - box[0][0]) * abs(box[2][1] - box[0][1])

def bounding_box_height(box):
    return abs(box[2][0] - box[0][0])

def get_angle_degrees2(x1, y1, x2, y2):
    return get_angle_degrees(x1, y1, x2, y2) if y1 < y2 else get_angle_degrees(x2, y2, x1, y1)
    
def get_angle_degrees(x1, y1, x2, y2):
    ret_val = math.atan2(y2 - y1, x2 - x1) * 180 / math.pi
    if ret_val < 0:
        return 180.0 + ret_val
    return ret_val

def get_angle_radians(x1, y1, x2, y2):
    ret_val = math.atan2(y2 - y1, x2 - x1)
    if ret_val < 0:
        return math.pi + ret_val
    return ret_val


def line_intersect(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
    """ returns a (x, y) tuple or None if there is no intersection """
    d = (by2 - by1) * (ax2 - ax1) - (bx2 - bx1) * (ay2 - ay1)
    if d:
        uA = ((bx2 - bx1) * (ay1 - by1) - (by2 - by1) * (ax1 - bx1)) / d
        uB = ((ax2 - ax1) * (ay1 - by1) - (ay2 - ay1) * (ax1 - bx1)) / d
    else:
        return None, None
    if not (0 <= uA <= 1 and 0 <= uB <= 1):
        return None, None
    x = ax1 + uA * (ax2 - ax1)
    y = ay1 + uA * (ay2 - ay1)

    return x, y

def yellow_conds(x1, y1, x2, y2):
    '''
    Condiciones para omitir el procesamiento de una línea amarilla detectada:
    si su ángulo es cercano a 0 o 180, así como si es cercano a recto.
    '''
    angle = get_angle_degrees2(x1, y1, x2, y2)
    return (angle < 30 or angle > 160) or (angle < 110 and angle > 90)

def white_conds(x1, y1, x2, y2):
    '''
    Condiciones para omitir el procesamiento de una línea blanca detectada:
    si se encuentra en el primer, segundo o tercer cuadrante, en otras palabras,
    se retorna False solo si la línea está en el cuarto cuadrante.
    '''
    return (min(x1,x2) < 320) or (min(y1,y2) < 320)


def distance_to_line(x_1, y_1, x_2, y_2):

    A = y_2 - y_1
    B = x_1 - x_2
    C = x_2 * y_1 - x_1 * y_2

    if B != 0:
        m = A / B
        d = abs(A * Reference_point[0] + B * Reference_point[1] + C) / math.sqrt(A**2 + B**2)
    else:
        d = abs(Reference_point[0] - x_1)
        m = 0

    print("valor distance_To line", d, m)

    return (d,m)





def duckie_detection(obs, converted, frame):
    '''
    Detectar patos, retornar si hubo detección y el ángulo de giro en tal caso 
    para lograr esquivar el duckie y evitar la colisión.
    '''
    #asignar el valor del area global#
    global duckie_area
    #converted_copy = converted.copy()

    # Se asume que no hay detección
    detection = False
    angle_duck = 0
    

    '''
    Para lograr la detección, se puede utilizar lo realizado en el desafío 1
    con el freno de emergencia, aunque con la diferencia que ya no será un freno,
    sino que será un método creado por ustedes para lograr esquivar al duckie.
    '''

    # Implementar filtros
    filtro_1_Duck = np.array([10, 239, 165]) 
    filtro_2_Duck = np.array([360, 255, 255]) 
    converted = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)

    #converted = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)
    mask_duckie = cv2.inRange(converted, filtro_1_Duck, filtro_2_Duck)
    segment_image = cv2.bitwise_and(converted, converted, mask= mask_duckie)
    image = cv2.cvtColor(segment_image, cv2.COLOR_HSV2BGR)
    kernel = np.ones((5,5),np.uint8)

    # Realizar la segmentación con operaciones morfológicas (erode y dilate) 
    # y buscar los contornos
    image_out = cv2.erode(mask_duckie, kernel, iterations = 2)    
    image_out = cv2.dilate(image_out, kernel, iterations = 10)
    contours, hierarchy = cv2.findContours(image_out, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # Revisar los contornos identificados y dibujar el rectángulo correspondiente
    # a la detección, además, dentro de este for, se establece la detección = verdadera
    # además del ángulo de giro angle = 'ángulo'
    for cnt in contours:
        # Obtener rectangulo
        x, y, w, h = cv2.boundingRect(cnt)
        duckie_area = w * h

        # Filtrar por area minima
        if w*h > DUCKIE_MIN_AREA:

            x2 = x + w  # obtener el otro extremo
            y2 = y + h
            # Dibujar un rectangulo en la imagen
            cv2.rectangle(frame, (int(x), int(y)), (int(x2),int(y2)), (0,0,255), 3)
            #definir angulo #
            center_x = x + w // 2
            center_y = y + h // 2
            reference_x = frame.shape[1] // 2  # Punto de referencia en el centro del eje horizontal
            reference_y = frame.shape[0] - 1  # Punto de referencia en el borde inferior

            delta_x = center_x - reference_x
            delta_y = reference_y - center_y  # Invertir el eje y

            # Calcular el ángulo en radianes
            angle_rad = math.atan2(delta_y, delta_x)

            # Convertir el ángulo a grados
            angle_deg = math.degrees(angle_rad)

            # Actualizar el ángulo
            angle_duck = angle_deg

            if w*h < AREA_MAX:

                detection = True
    

    return detection



def red_line_detection(converted, frame):
    # Se asume que no hay detección
    detection = False

    # Implementar filtros
    filtro_1_Red = np.array([111, 79, 81]) 
    filtro_2_Red = np.array([360, 239, 186]) 
    mask_red_line = cv2.inRange(converted, filtro_1_Red, filtro_2_Red)

    # Realizar la segmentación con operaciones morfológicas (erode y dilate) 
    # y buscar los contornos
    kernel = np.ones((5,5),np.uint8)
    image_out_line = cv2.erode(mask_red_line, kernel, iterations = 1)   
    image_out_line = cv2.dilate(image_out_line, kernel, iterations = 1)
 
    # Revisar los contornos identificados y dibujar el rectángulo correspondiente
    # a la detección, además, Si hay detección, detection = True
    contours, hierarchy = cv2.findContours(image_out_line, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        # Obtener rectangulo
        x, y, w, h = cv2.boundingRect(cnt)

        # Filtrar por area minima
        if w*h > RED_LINE_MIN_AREA:

            x2 = x + w  # obtener el otro extremo
            y2 = y + h
            # Dibujar un rectangulo en la imagen
            cv2.rectangle(frame, (int(x), int(y)), (int(x2),int(y2)), (255,0,0), 3)
            if w*h > AREA_MAX:
                print("true")
                detection = True
    return detection



def get_line(converted, frame ,filter_1, filter_2, Color_RGB, line_color):
    '''
    Determina el ángulo al que debe girar el duckiebot dependiendo
    del filtro aplicado, y de qué color trata, si es "white"
    y se cumplen las condiciones entonces gira a la izquierda,
    si es "yellow" y se cumplen las condiciones girar a la derecha.
    '''

    coord_history = []
    mask = cv2.inRange(converted, filter_1, filter_2)
    segment_image = cv2.bitwise_and(converted, converted, mask=mask)
    # Obtener la altura de la pantalla
    height, width = frame.shape[:2]
    
    # Definir las coordenadas y de la región de interés en la parte inferior de la pantalla
    y1 = int(0.5 * height)  # Inicio de la región de interés (70% de la altura)
    y2 = height  # Fin de la región de interés (parte inferior de la pantalla)
    
    # Dibujar un rectángulo en la imagen para visualizar la región de interés
    cv2.rectangle(frame, (0, y1), (width, y2), (255, 0, 0), 3)
    # Erosionar la imagen
    image = cv2.cvtColor(segment_image, cv2.COLOR_HSV2BGR)
    kernel = np.ones((5,5),np.uint8)
    image_lines = cv2.erode(image, kernel, iterations = 2)    
    # Detectar líneas
    gray_lines = cv2.cvtColor(image_lines, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_lines, 50, 200, None, 3)

    # Detectar lineas usando houghlines y lo aprendido en el desafío 2.
    lines = cv2.HoughLines(edges, 1, np.pi/180, 70)
    x1, y1, x2, y2 = 0, 0, 0, 0  # Valores predeterminados para x1, y1, x2, y2
    avg_x1, avg_y1, avg_x2, avg_y2 = 0, 0, 0, 0 
    if lines is not None:
        lista_r_theta = []  # Crear una lista vacía para almacenar los valores de r y theta
        for r_theta in lines[:6]:
            arr = np.array(r_theta[0], dtype=np.float64)
            r, theta = arr           
            lista_r_theta.append((r, theta))

            if lista_r_theta:
                avg_r_theta = np.mean(lista_r_theta, axis=0)  # Calcular el promedio de r y theta
                avg_r, avg_theta = avg_r_theta
                a = np.cos(avg_theta)
                b = np.sin(avg_theta)
                x0 = a * avg_r
                y0 = b * avg_r
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                # Agregar las coordenadas al historial
                coord_history.append(((x1, y1), (x2, y2)))

            # Calcular el promedio ponderado de las coordenadas
            avg_coords = np.average(coord_history, axis=0)

            # Obtener las coordenadas promediadas
            avg_x1, avg_y1 = avg_coords[0]
            avg_x2, avg_y2 = avg_coords[1]

            cv2.line(frame, (int(avg_x1), int(avg_y1)), (int(avg_x2), int(avg_y2)), Color_RGB, 2)
            cv2.imshow("borde detectado" , edges )
            cv2.imshow("Frame" , frame )
        
        return (avg_coords)
 

def line_follower(vel, angle, obs):
    global red_line_detected

    converted = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)
    frame = obs[:, :, [2, 1, 0]]
    frame = cv2.UMat(frame).get()

    # Detección de duckies
    #detection = duckie_detection(obs=obs, frame=frame, converted=converted)
    '''
    Implementar evasión de duckies en el camino, variado la velocidad angular del robot
    '''
    #if detection:
    #    duckie_area_max = 2000  # Definir el área máxima para activar la evasión
    #    if duckie_area > duckie_area_max:
    #        evasion_angle = angle - 30  # Girar 30 grados en dirección opuesta a la línea blanca
    

    # Detección de líneas rojas
    detection = red_line_detection(converted=converted, frame=frame)
    ''' 
    Implementar detención por un tiempo determinado del duckiebot
    al detectar una linea roja en el camino, luego de este tiempo,
    el duckiebot debe seguir avanzando
    '''
    red_line_stop_time = 5

    if red_line_detection is True:
        red_line_detected = True  # Activar la detención por línea roja
        red_line_stop_time = time.time()  # Guardar el tiempo de inicio de la detención

    if red_line_detected:
        # Detener el duckiebot por un tiempo determinado
        stop_duration = time.time() - red_line_stop_time
        if stop_duration < red_line_stop_duration:
            env.step([0.0 , 0.0])
        else:
            red_line_detected = False  # Reiniciar la detección de línea roja

    # Obtener el ángulo propuesto por cada color
    avg_coords_white = get_line(converted, frame , white_filter_1, white_filter_2, white_RGB,"white")
    avg_coords_yellow = get_line(converted, frame , yellow_filter_1, yellow_filter_2, yellow_RGB, "yellow")

    avg_m_white = 0.0

    if np.all(avg_coords_white): 
        avg_dist_white, avg_m_white = distance_to_line(avg_coords_white[0][0],avg_coords_white[0][1], avg_coords_white[1][0], avg_coords_white[1][1])
        if avg_dist_white < MAX_TOLERANCE:
            env.step([0.0 , 0.0])
            if avg_m_white <0.0:
                env.step([0.0 , 7.8])
            else:
                env.step([0.0 , -7.8])
    if np.all(avg_coords_yellow):
        avg_dist_yellow, avg_m_yellow = distance_to_line(avg_coords_yellow[0][0],avg_coords_yellow[0][1], avg_coords_yellow[1][0], avg_coords_yellow[1][1])
        if avg_dist_yellow < MAX_TOLERANCE:
            env.step([0.0 , 0])
            if avg_m_white <0.0:
                env.step([0.0 , -6.0])
            else:
                env.step([0.0 , +6.0])

    return np.array([vel, angle]) # Implementar nuevo ángulo de giro controlado
    


@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    Handler para reiniciar el ambiente
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.render(mode="top_down")

    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0

    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

    env.render()


# Registrar el handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

action = np.array([0.0, 0.0])

def update(dt):
    """
    Funcion que se llama en step.
    """
    action = np.array([0.6, 0.0])
    # Aquí se controla el duckiebot
    if key_handler[key.UP]:
        action[0] += 0.44
    if key_handler[key.DOWN]:
        action[0] -= 0.44
    if key_handler[key.LEFT]:
        action[1] += 1
    if key_handler[key.RIGHT]:
        action[1] -= 1
    if key_handler[key.SPACE]:
        action = np.array([0, 0])

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 7

    ''' Aquí se obtienen las observaciones y se setea la acción
    Para esto, se debe utilizar la función creada anteriormente llamada line_follower,
    la cual recibe como argumentos la velocidad lineal, la velocidad angular y 
    la ventana de la visualización, en este caso obs.
    Luego, se setea la acción del movimiento implementado con el controlador
    con action[i], donde i es 0 y 1, (lineal y angular)
    '''

    # obs consiste en un imagen de 640 x 480 x 3
    obs, reward, done, info = env.step(action)
    vel, angle = line_follower(action[0], action[1], obs)
    action[0] = vel
    action[1] = angle

    if done:
        print('done!')
        env.reset()
        env.render(mode="top_down")

    env.render(mode="top_down")

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()