import os
import cv2
from PIL import Image
import random

BUFFER_PERCENTAGE = 0.05


def find_faces(image):
    cascade_path = 'C:/dev/mask-off/generateMask/assets/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(
        image,
        scaleFactor=1.21,
        minNeighbors=3,
        minSize=(18, 18),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    return faces


def find_eye_line(face):
    eye_cascade = cv2.CascadeClassifier('C:/dev/mask-off/generateMask/assets/haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(face, scaleFactor=1.2, minNeighbors=4)
    eye_line = max(map(lambda eye: int(eye[1] + eye[-1] / 2), eyes[:1]))

    return eye_line


def main():
    images_path = '../pair_face/without_mask/'
    images = os.listdir(images_path)
    for image_name in images:
        image = cv2.imread(images_path + image_name)
        height, width, depth = image.shape
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = find_faces(image)

        if random.choice([0, 1]) == 0:
            mask_path = 'assets/black-mask.png'
        else:
            mask_path = 'assets/aligned-blue-mask.png'
        mask = Image.open(mask_path).convert('RGBA')

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 235, 0), 2)
            (startX, startY, endX, endY) = (x, y, x + w, y + h)
            startX = int(max(startX - width * BUFFER_PERCENTAGE, 0))
            startY = int(max(startY - height * BUFFER_PERCENTAGE, 0))
            endX = int(min(endX + width * BUFFER_PERCENTAGE, width))
            endY = int(min(endY + height * BUFFER_PERCENTAGE, height))
            face = image[startY:endY, startX:endX]

            eye_line = find_eye_line(face)
            if eye_line != -1:
                background = Image.fromarray(face)
                b, g, r = background.split()
                background = Image.merge('RGB', (r, g, b))

                new_mask_size = (background.size[0], background.size[1] - eye_line)
                resized_mask = mask.copy().resize(new_mask_size)

                background.paste(resized_mask, box=(0, eye_line), mask=resized_mask)

                fullbackground = Image.fromarray(img)
                fullbackground.paste(background, (startX, startY))
                fullbackground.save('../pair_face/with_different_masks/' + image_name)


if __name__ == '__main__':
    main()
