import os
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # pretrained YOLOv8n model

# Specify the directory containing images
image_directory = r"C:\Users\17049\Desktop\FinalProject\bike_test"

# Normalize the path
image_directory = os.path.normpath(image_directory)

# Debugging: Print the directory path
print(f"Checking directory: {image_directory}")

# Check if the directory exists
if not os.path.exists(image_directory):
    raise FileNotFoundError(f"The directory {image_directory} does not exist.")
else:
    print(f"Directory exists: {image_directory}")

# Supported image extensions
supported_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

# Get a list of all image files in the directory
image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(supported_extensions)]

# Debugging: Print the list of image files
print(f"Found image files: {image_files}")

# Run batched inference on each image in the directory
for image_file in image_files:
    image_path = os.path.join(image_directory, image_file)
    results = model([image_path])  # run inference on the image

    # Process results
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()  # display to screen
        result.save(filename=os.path.join(image_directory, f"result_{image_file}"))  # save to disk with a new filename

