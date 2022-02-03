# Import modules
import codecademylib3_seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

# Load data
digits = datasets.load_digits()

# Print out data description
print(digits.DESCR)

# Print out the digits data
print(digits.data)

# Print out the target values
print(digits.target)

# Print out the target label at index 100
print(digits.target[100])

# Set colormap to gray
plt.gray() 

# Represent array as a matrix
plt.matshow(digits.images[100])

# Display plot 
plt.show()

# clear current plot
plt.clf()

# Figure size (width, height)
fig = plt.figure(figsize=(6, 6))
 
# Adjust the subplots 
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
 
# For each of the 64 images
for i in range(64):
    # Initialize the subplots: add a subplot in the grid of 8 by 8, at the i+1-th position
    ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])
    # Display an image at the i-th position
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    # Label the image with the target value
    ax.text(0, 7, str(digits.target[i]))
# Display plot 
plt.show()

# Create KMeans cluster model
model = KMeans(n_clusters=10, random_state=42)

# Train the model
model.fit(digits.data)

# Clear current plot
plt.clf()

# Create a figure of size 8x3
fig = plt.figure(figsize=(8,3))

# Add title
fig.suptitle("Cluser Center Images", fontsize=14, fontweight="bold")

# Create a for loop
for i in range(10):
  # Initialize subplots in a grid of 2X5, at i+1th position
  ax = fig.add_subplot(2, 5, 1 + i)
  # Display images
  ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)

# Show plot
plt.show()

# Create new_samples array
new_samples = np.array([
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.09,4.69,6.98,6.34,1.87,0.00,0.00,0.46,6.17,5.36,1.41,3.51,7.21,2.34,0.00,3.57,5.59,0.07,0.00,0.00,2.76,6.51,0.00,5.29,2.71,0.00,0.00,0.00,0.75,6.84,0.00,4.71,4.83,0.00,0.00,0.00,2.93,6.21,0.00,0.82,6.67,4.53,2.04,3.87,7.26,2.17,0.00,0.00,0.92,5.28,5.92,4.84,1.88,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.45,2.27,2.88,1.58,0.07,0.00,0.00,1.20,6.91,6.05,5.25,6.99,6.64,2.40,0.00,4.84,4.38,0.00,0.00,0.07,2.98,6.97,0.00,5.82,2.57,0.00,0.00,0.00,0.82,7.50,0.00,4.20,3.79,0.00,0.00,0.00,3.32,5.45,0.00,3.19,6.89,4.68,3.63,2.92,7.19,2.34,0.00,0.07,1.96,3.48,4.40,5.31,3.22,0.00],
[0.00,0.00,0.00,1.07,3.80,1.67,0.00,0.00,0.00,0.23,3.72,7.15,5.40,7.14,5.63,1.67,0.00,3.80,6.30,1.66,0.00,0.38,4.26,5.68,0.38,7.15,1.36,0.00,0.00,0.00,1.52,6.09,2.12,5.93,0.00,0.00,0.00,0.00,1.51,6.09,1.36,6.78,0.08,0.00,0.00,0.00,2.20,5.78,0.00,5.93,5.93,4.26,3.80,4.33,6.39,4.55,0.00,0.53,2.58,3.73,3.80,3.72,2.26,0.37],
[0.00,0.00,0.00,0.00,1.67,4.86,2.95,0.00,0.00,0.00,0.07,3.20,7.06,4.17,7.13,0.90,0.00,0.23,5.55,6.66,2.13,0.00,4.49,4.03,0.00,2.42,6.23,0.22,0.00,0.00,1.90,6.00,0.00,4.17,4.02,0.00,0.00,0.00,1.52,6.07,0.00,3.27,6.61,2.42,0.00,0.00,3.34,5.85,0.00,0.00,3.19,6.77,5.94,6.78,6.76,2.04,0.00,0.00,0.00,0.61,2.28,1.27,0.00,0.00]
])

# Predict new labels
new_labels = model.predict(new_samples)

# Print out new_labels
print(new_labels)

# A for loop to map out each of the labels with the digits we think it represents
for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(0, end='')
  elif new_labels[i] == 1:
    print(9, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(6, end='')
  elif new_labels[i] == 5:
    print(8, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(5, end='')
  elif new_labels[i] == 8:
    print(7, end='')
  elif new_labels[i] == 9:
    print(3, end='')