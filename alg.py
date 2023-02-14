import cv2
import numpy as np
import matplotlib.pyplot as plt

# Open the camera
cap = cv2.VideoCapture(0)

# Define the video codec and create VideoWriter object for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (640, 480))

# Initialize the previous frame
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Create a figure to display the frames and difference graphs
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

# Create a loop for processing each frame
while True:
    # Read a new frame from the camera
    ret, frame2 = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate the optical flow using Lucas-Kanade method
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Apply the motion compensation to the current frame
    rows, cols = next.shape
    map_x, map_y = np.meshgrid(np.arange(cols), np.arange(rows))
    dst_x = map_x - flow[...,0]
    dst_y = map_y - flow[...,1]
    map_x = dst_x.astype(np.float32)
    map_y = dst_y.astype(np.float32)
    stabilized = cv2.remap(frame2, map_x, map_y, cv2.INTER_LINEAR)

    # Compute the difference between the original and stabilized frames
    diff = cv2.absdiff(frame2, stabilized)
    diff_x = np.mean(diff, axis=0)
    diff_y = np.mean(diff, axis=1)

    # Write the stabilized frame to the output video
    out.write(stabilized)

    # Display the original and stabilized frames
    cv2.imshow('Original', frame2)
    cv2.imshow('Stabilized', stabilized)

    # Display the difference graphs
    ax1.clear()
    ax1.imshow(frame2)
    ax1.set_title('Original Frame')
    ax2.clear()
    ax2.imshow(stabilized)
    ax2.set_title('Stabilized Frame')
    ax3.clear()
    ax3.plot(diff_x)
    ax3.set_title('Horizontal Difference')
    ax4.clear()
    ax4.plot(diff_y)
    ax4.set_title('Vertical Difference')
    plt.pause(0.001)

    # Update the previous frame
    prvs = next

    # Check for key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and output video objects
cap.release()
out.release()

# Close all windows
cv2.destroyAllWindows()
