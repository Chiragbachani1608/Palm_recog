import tkinter as tk
from tkinter import messagebox
import subprocess
import cv2
import numpy as np
from tinydb import TinyDB
from scipy.spatial.distance import directed_hausdorff

# Database Initialization
db = TinyDB('palm_data.json')

# Feature Extraction Function using SIFT
def extract_palm_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors

# Hausdorff Distance Calculation
def calculate_hausdorff_distance(features1, features2):
    h1, _, _ = directed_hausdorff(features1, features2)
    h2, _, _ = directed_hausdorff(features2, features1)
    return max(h1, h2)

# Save Captured Image
def capture_image(image_path):
    # Run the external palm detection script (1.py) and save the image to the specified path
    subprocess.run(["python", "C:\\Users\\chira\\OneDrive\\Desktop\\Palm 2 Pay (1)\\1.py", image_path], check=True)

# Register Function
def register_user():
    name = entry_name.get().strip()
    if not name:
        messagebox.showerror("Error", "Please enter a name to register.")
        return

    # Define the registration image path
    register_image_path = f"captured_palm_{name}_register.jpg"

    # Capture the image for registration
    capture_image(register_image_path)

    # Load the captured image
    image = cv2.imread(register_image_path)
    if image is None:
        messagebox.showerror("Error", "Failed to load the captured palm image.")
        return

    # Extract features
    descriptors = extract_palm_features(image)
    if descriptors is None:
        messagebox.showerror("Error", "No palm features detected. Try again.")
        return

    # Save user data (name + features) in the database
    db.insert({"name": name, "descriptors": descriptors.tolist()})
    messagebox.showinfo("Success", f"User '{name}' registered successfully!")

# Login Function
def login_user():
    # Define the login image path
    login_image_path = "captured_palm_image_login.jpg"

    # Capture the image for login
    capture_image(login_image_path)

    # Load the captured image for login
    image = cv2.imread(login_image_path)
    if image is None:
        messagebox.showerror("Error", "Failed to load the captured palm image.")
        return

    # Extract features from the login image
    descriptors_login = extract_palm_features(image)
    if descriptors_login is None:
        messagebox.showerror("Error", "No palm features detected. Try again.")
        return

    # Compare with registered users using Hausdorff Distance
    users = db.all()
    if not users:
        messagebox.showinfo("Info", "No registered users found.")
        return

    matched_user = None
    min_distance = float('inf')

    for user in users:
        registered_descriptors = np.array(user['descriptors'], dtype=np.float32)
        distance = calculate_hausdorff_distance(descriptors_login, registered_descriptors)
        if distance < min_distance:
            min_distance = distance
            matched_user = user['name']

    # Set a threshold for recognition accuracy
    threshold = 10.0  # Adjust based on testing
    if matched_user and min_distance < threshold:
        messagebox.showinfo("Success", f"Welcome, {matched_user}!")
    else:
        messagebox.showerror("Error", "Palm does not match any registered users.")

# GUI Setup
root = tk.Tk()
root.title("Palm 2 Pay - Attendance System")
root.geometry("400x300")

# Title
title_label = tk.Label(root, text="Palm 2 Pay Attendance System", font=("Arial", 16))
title_label.pack(pady=10)

# Name Entry
name_label = tk.Label(root, text="Enter Name (for Registration):")
name_label.pack(pady=5)
entry_name = tk.Entry(root)
entry_name.pack(pady=5)

# Register Button
register_button = tk.Button(root, text="Register Palm", command=register_user, bg="lightgreen", width=20)
register_button.pack(pady=10)

# Login Button
login_button = tk.Button(root, text="Login with Palm", command=login_user, bg="lightblue", width=20)
login_button.pack(pady=10)

# Exit Button
exit_button = tk.Button(root, text="Exit", command=root.quit, bg="red", width=20)
exit_button.pack(pady=10)

# Run the GUI
root.mainloop()
