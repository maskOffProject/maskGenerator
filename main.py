import os
import cv2
from PIL import Image


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
    images_path = 'C:/mask/without_mask/'
    images = os.listdir(images_path)[:10]
    for image_name in images:
        image = cv2.imread(images_path + image_name)
        faces = find_faces(image)

        mask_path = 'C:/dev/mask-off/generateMask/assets/black-mask.png'
        mask = Image.open(mask_path).convert('RGBA')

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 235, 0), 2)
            (startX, startY, endX, endY) = (x, y, x + w, y + h)
            face = image[startY:endY, startX:endX]

            eye_line = find_eye_line(face)
            background = Image.fromarray(face)
            b, g, r = background.split()
            background = Image.merge('RGB', (r, g, b))

            new_mask_size = (background.size[0], background.size[1] - eye_line)
            resized_mask = mask.copy().resize(new_mask_size)

            background.paste(resized_mask, box=(0, eye_line), mask=resized_mask)
            background.save('C:/mask/with_black_mask/' + image_name)


if __name__ == '__main__':
    main()