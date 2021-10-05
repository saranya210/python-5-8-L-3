import sys

import cv2, time, dlib, math
import numpy as np
from imutils import face_utils, rotate_bound
from os import listdir
from os.path import isfile, join

def calc_incl(p1, p2):
	x1,x2,y1,y2 = p1[0], p2[0], p1[1], p2[1]
	incl = 180/math.pi*math.atan((float(y2-y1))/(x2-x1))
	return incl

def apply_sprite(frame, p2sprite, w, x, y, angle, ontop = True):
	sprite = cv2.imread(p2sprite, -1)
	sprite = rotate_bound(sprite, angle)
	(sprite, y_final) = adjust_sprite2head(sprite, w, y, ontop)
	frame = draw_sprite(frame, sprite, x, y_final)

def adjust_sprite2head(sprite, head_w, head_y, ontop = True):
	(h_sprite, w_sprite) = (sprite.shape[0], sprite.shape[1])
	factor = 1.0*head_w/w_sprite
	sprite = cv2.resize(sprite, (0,0), fx=factor, fy=factor)
	(h_sprite, w_sprite) = (sprite.shape[0], sprite.shape[1])
	y_orig = head_y-h_sprite if ontop else head_y
	if y_orig < 0:
		sprite = sprite[abs(y_orig)::,:,:]
		y_orig = 0
	return (sprite, y_orig)

def draw_sprite(frame, sprite, x_off, y_off):
	(h,w) = (sprite.shape[0], sprite.shape[1])
	(imH,imW) = (frame.shape[0], frame.shape[1])

	if y_off+h >= imH:
		sprite = sprite[0:imH-y_off,:,:]
	if x_off+w >= imW:
		sprite = sprite[:,0:imW-x_off,:]
	if x_off < 0:
		sprite = sprite[:,abs(x_off)::,:]
		w = sprite.shape[1]
		x_off = 0

	for c in range(3):
		frame[y_off:y_off+h, x_off:x_off+w, c] = sprite[:,:,c] * (sprite[:,:,3]/255.0) + frame[y_off:y_off+h, x_off:x_off+w,c] * (1.0 - sprite[:,:,3]/255.0)
	return frame

def calc_bbox(coords):
	x = min(coords[:,0])
	y = min(coords[:,1])
	w = max(coords[:,0]) - x
	h = max(coords[:,1]) - y
	return (x,y,w,h)

def get_bbox(points, part):
	if part == 1:
		(x,y,w,h) = calc_bbox(points[17:22])	#left eyebrow
	elif part == 2:
		(x,y,w,h) = calc_bbox(points[22:27])	#right eyebrow
	elif part == 3:
		(x,y,w,h) = calc_bbox(points[36:42])	#left eye
	elif part == 4:
		(x,y,w,h) = calc_bbox(points[42:48])	#right eye
	elif part == 5:
		(x,y,w,h) = calc_bbox(points[29:36])	#nose
	elif part == 6:
		(x,y,w,h) = calc_bbox(points[48:68])	#mouth
	return (x,y,w,h)

dir_ = './sprites/googlies/'
googlies = [f for f in listdir(dir_) if isfile(join(dir_, f))]
i = 0
dir1 = './sprites/clouds/'
clouds = [g for g in listdir(dir1) if isfile(join(dir1, g))]
j = 0


vs = cv2.VideoCapture(0)
time.sleep(1.5)

detector = dlib.get_frontal_face_detector()
model = 'filters/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(model)

while True:
	ret, frame = vs.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = detector(gray, 0)

	for face in faces:
		(x,y,w,h) = (face.left(), face.top(), face.width(), face.height())
		shape = predictor(gray, face)
		shape = face_utils.shape_to_np(shape)
		incl = calc_incl(shape[17], shape[26])

		mouth_open = (shape[66][1] - shape[62][1]) >= 10
		(x0,y0,w0,h0) = get_bbox(shape, 6)	#mouth
		(x1,y1,w1,h1) = get_bbox(shape, 1)	#eyes
		(x2,y2,w2,h2) = get_bbox(shape, 5)	#nose
		apply_sprite(frame, dir_+googlies[i], w, x, y1, incl, ontop = False)
		i += 1
		i = 0 if i >= len(googlies)	else i
		apply_sprite(frame, dir1+clouds[j], w, x, y-100, incl, ontop = False)
		j += 1
		j = 0 if j >= len(clouds) else j
		apply_sprite(frame, './sprites/nosepick.png', w2, x2-15, y2+25, incl, ontop = False)
		if mouth_open:
			apply_sprite(frame, './sprites/rainbow.png', w0, x0, y0, incl, ontop = False)

	cv2.imshow('Snap', frame)
	
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break

vs.release()
cv2.destroyAllWindows()
