import dlib
from skimage import io
import numpy as np

def euclidean_distance(descriptor1, descriptor2):
    return np.linalg.norm(np.array(descriptor1) - np.array(descriptor2))

def compare_images(image_path1, image_path2):
    # Load the images
    img1 = io.imread(image_path1)
    img2 = io.imread(image_path2)

    # Create a face recognition model
    face_detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    face_recognition_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

    # Detect faces in the images
    faces1 = face_detector(img1)
    faces2 = face_detector(img2)

    # Check if at least one face is detected in each image
    if len(faces1) == 0 or len(faces2) == 0:
        print("No faces found in one or both images.")
        return False

    # Get facial features (landmarks) and face descriptors
    shape1 = shape_predictor(img1, faces1[0])
    face_descriptor1 = face_recognition_model.compute_face_descriptor(img1, shape1)

    shape2 = shape_predictor(img2, faces2[0])
    face_descriptor2 = face_recognition_model.compute_face_descriptor(img2, shape2)

    # Compare the face descriptors
    distance = euclidean_distance(face_descriptor1, face_descriptor2)

    # Set a threshold for similarity (you may need to adjust this value)
    threshold = 0.6

    # Check if the distance is below the threshold
    if distance < threshold:
        print("The photos are of the same person.")
        return True
    else:
        print("The photos are of different people.")
        return False

# Example usage
image_path1 = "rubeyy.jpg"
image_path2 = "123.jpg"
compare_images(image_path1, image_path2)
