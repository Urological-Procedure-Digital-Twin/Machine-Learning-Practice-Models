import cv2
import numpy as np

# Load original ground truth frame and neural network prediction
ground_truth = cv2.imread('predicted_scope_frame.png')  # Predicted output from train_model.py
cap = cv2.VideoCapture('scope_video.mp4')
ret, original_frame = cap.read()
cap.release()

if ret and ground_truth is not None:
    # Add text labels
    cv2.putText(original_frame, "Original Ground Truth", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(ground_truth, "Model Synthetic Frame", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Combine side-by-side into a single image
    comparison = np.hstack((original_frame, ground_truth))
    
    # Save comparison image
    cv2.imwrite('side_by_side_comparison.png', comparison)
    print("Saved 'side_by_side_comparison.png'! Open this file to inspect your model's prediction.")
else:
    print("Could not load frames. Make sure train_model.py finished successfully.")