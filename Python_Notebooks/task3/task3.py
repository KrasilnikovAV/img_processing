import cv2
import statistics as st
import math
import numpy as np
import time

# Функция для вывода изображения
def viewImage(image, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# метрика подобия изображений
def psnr(image_1, image_2):
	image_change = (image_1 - image_2) ** 2
	
	MSE_R = st.fmean([pixel[0] for row in image_change for pixel in row])
	MSE_G = st.fmean([pixel[1] for row in image_change for pixel in row])
	MSE_B = st.fmean([pixel[2] for row in image_change for pixel in row])

	MSE = (MSE_R + MSE_G + MSE_B) / 3

	MAX_I = 2 ** 8

	PSNR = 10 * math.log10(MAX_I ** 2 / MSE);
	return PSNR

# функция увеличения яркости для BGR
def bright_bgr(source_image, value):
	B, G, R = cv2.split(source_image)
	for i in range(0, source_image.shape[0]):
		for j in range(0, source_image.shape[1]):
			B[i, j] += value
			if B[i, j] > 255: 
				B[i, j] = 255
			G[i, j] += value
			if G[i, j] > 255: 
				G[i, j] = 255
			R[i, j] += value
			if R[i, j] > 255: 
				R[i, j] = 255
	dest_image = cv2.merge((B, G, R))
	return dest_image
	
# функция увеличения яркости для YUV
def bright_yuv(source_image, value):
	Y, U, V = cv2.split(source_image)
	for i in range(0, source_image.shape[0]):
		for j in range(0, source_image.shape[1]):
			Y[i, j] += value
			if Y[i, j] > 255: 
				Y[i, j] = 255
	dest_image = cv2.merge((Y, U, V))
	return dest_image
	
# загружаем исходное изображение	
image = cv2.imread("cat.JPG")
viewImage(image, "Source image")

##################### ПРЕОБРАЗОВАНИЕ ИЗ BGR В YUV #####################

# конвертируем его из BGR в YUV с помощью OpenCV
image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
viewImage(image_yuv, "CV YUV image")

# теперь реализуем конвертацию. высчитываем компоненты
B, G, R = cv2.split(image)

Y = np.ndarray(shape=B.shape, dtype=B.dtype)
U = np.ndarray(shape=B.shape, dtype=B.dtype)
V = np.ndarray(shape=B.shape, dtype=B.dtype)

for i in range(0, image.shape[0]):
	for j in range(0, image.shape[1]):
		Y[i, j] = int(0.299 * R[i, j] + 0.587 * G[i, j] + 0.114 * B[i, j])
		U[i, j] = int(-0.147 * R[i, j] - 0.289 * G[i, j] + 0.437 * B[i, j] + 128)
		V[i, j] = int(0.615 * R[i, j] - 0.515 * G[i, j] - 0.09 * B[i, j] + 128)

# объединяем их в одно изображение
my_image_yuv = cv2.merge((Y, U, V))
viewImage(my_image_yuv, "My YUV image")

# вычисляем меру сходства
PSNR = psnr(image_yuv, my_image_yuv)
print(PSNR)

##################### ПРЕОБРАЗОВАНИЕ ИЗ YUV В BGR #####################

# конвертируем ищображение из YUV в RGB с помощью OpenCV
image_bgr = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)
viewImage(image_bgr, "CV BGR image")

# теперь реализуем конвертацию. высчитываем компоненты
Y, U, V = cv2.split(my_image_yuv)
B = np.ndarray(shape=Y.shape, dtype=Y.dtype)
G = np.ndarray(shape=Y.shape, dtype=Y.dtype)
R = np.ndarray(shape=Y.shape, dtype=Y.dtype)
for i in range(0, my_image_yuv.shape[0]):
	for j in range(0, my_image_yuv.shape[1]):
		R[i, j] = int(Y[i, j] + 1.13983 * (V[i, j] - 128))
		G[i, j] = int(Y[i, j] - 0.39465 * (U[i, j] - 128) - 0.5806 * (V[i, j] - 128))
		B[i, j] = int(Y[i, j] + 2.03211 * (U[i, j] - 128))

# объединяем их в одно изображение
my_image_bgr = cv2.merge((B, G, R))
viewImage(my_image_bgr, "My BGR image")

# вычисляем меру сходства
PSNR = psnr(image_bgr, my_image_bgr)
print(PSNR)

##################### ЯРКОСТЬ С ЗАМЕРОМ ВРЕМЕНИ#####################

start_time = time.time() 
viewImage(bright_bgr(image, 15), "bright BGR image")
end_time = time.time()
print(end_time - start_time)

start_time = time.time() 
viewImage(bright_yuv(image_yuv, 20), "bright YUV image")
end_time = time.time()
print(end_time - start_time)