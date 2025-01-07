import os
import subprocess
import cv2
import numpy as np
from scipy.spatial.distance import directed_hausdorff

# Function to perform feature extraction using SIFT
def extract_sift_features(image_path):
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image from {image_path}")
    
    # SIFT feature extraction
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors

# Function to calculate Hausdorff distance between two sets of features
def hausdorff_distance(des1, des2):
    # Convert descriptors to numpy arrays
    des1 = np.float32(des1)
    des2 = np.float32(des2)
    
    # Calculate the directed Hausdorff distance
    dist, _, _ = directed_hausdorff(des1, des2)
    return dist

# Function to register a user
def register_user(register_image_path, user_id):
    # Run 1.py (subprocess) to capture the image and save it as register_image_path
    subprocess.run(["python", "1.py", register_image_path], check=True)
    
    # Check if the image was saved correctly
    if not os.path.exists(register_image_path):
        print(f"Error: Image not saved at {register_image_path}")
        return
    
    # Extract features from the captured register image
    kp_register, des_register = extract_sift_features(register_image_path)
    
    # Save the features to a database (simulating with file storage)
    database_path = f"database/user_{user_id}_register_features.npy"
    np.save(database_path, des_register)
    print(f"Registration complete. Features saved to {database_path}")

# Function to login a user
def login_user(login_image_path, user_id):
    # Run 1.py (subprocess) to capture the image and save it as login_image_path
    subprocess.run(["python", "1.py", login_image_path], check=True)
    
    # Check if the image was saved correctly
    if not os.path.exists(login_image_path):
        print(f"Error: Image not saved at {login_image_path}")
        return
    
    # Extract features from the captured login image
    kp_login, des_login = extract_sift_features(login_image_path)
    
    # Retrieve the registered features from the database
    database_path = f"database/user_{user_id}_register_features.npy"
    
    if not os.path.exists(database_path):
        print("No registered features found for the user.")
        return
    
    des_register = np.load(database_path)
    
    # Compare the features using Hausdorff distance
    dist = hausdorff_distance(des_register, des_login)
    print(f"Hausdorff distance between registration and login image: {dist}")
    
    # If the Hausdorff distance is within a threshold, consider the login as successful
    threshold = 0.5  # Define an appropriate threshold for feature matching
    if dist < threshold:
        print("Login successful!")
    else:
        print("Login failed. Feature mismatch detected.")

# Example usage:
user_id = 1  # Example user ID

# Register a user (save the register image)
register_image_path = "captured_palm_C_register.jpg"
register_user(register_image_path, user_id)

# Login the user (capture the login image and compare)
login_image_path = "captured_palm_C_login.jpg"
login_user(login_image_path, user_id)
#Threshold Tuning: You may need to adjust the Hausdorff threshold based on experimentation. A smaller threshold will increase the precision but may reduce the tolerance for variability in user hand positioning.
#Database Integration: The feature extraction result (des_register) is saved as a .npy file in the database directory. This can be extended to a database for scalability.





