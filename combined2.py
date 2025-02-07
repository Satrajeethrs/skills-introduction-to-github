import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from heapq import heappush, heappop

# Global variables for coordinates
coords = []

# Event handler for mouse clicks to get coordinates
def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        # Append the clicked coordinates to the list
        coords.append((int(event.ydata), int(event.xdata)))  # Note the (y, x) format for OpenCV
        print(f"Clicked at: {coords[-1]}")
        
        # Display bounding box if 2 points are clicked
        if len(coords) == 2:
            x1, y1 = coords[0]
            x2, y2 = coords[1]
            bbox = [(x1, y1), (x2, y2)]
            print(f"Bounding Box: {bbox}")
            
            # Draw the bounding box on the image
            rect = plt.Rectangle((y1, x1), y2 - y1, x2 - x1,
                                 linewidth=2, edgecolor='red', facecolor='none')
            plt.gca().add_patch(rect)
            plt.draw()

# Load and display the image for coordinate selection
def select_coordinates(image_path):
    img = mpimg.imread(image_path)
    plt.imshow(img)
    plt.title("Click to select coordinates (2 clicks for bounding box)")
    
    # Connect the click event to the handler
    plt.gcf().canvas.mpl_connect('button_press_event', onclick)
    plt.show()

# Load image and preprocess for binary map
def load_image(image_path):
    """Load the image and handle errors."""
    floor_plan = cv2.imread(image_path)
    if floor_plan is None or floor_plan.size == 0:
        raise FileNotFoundError("Error: Unable to load the image. Check the file path.")
    return floor_plan

def preprocess_image(floor_plan):
    """Convert the image into a binary map."""
    gray_image = cv2.cvtColor(floor_plan, cv2.COLOR_BGR2GRAY)
    _, binary_map = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
    binary_map = cv2.bitwise_not(binary_map)  # Invert: walls -> 255, free space -> 0
    return binary_map

# A* Pathfinding algorithm
def astar(binary_map, start, goal):
    """A* algorithm to find the shortest path."""
    rows, cols = binary_map.shape
    visited = np.zeros((rows, cols), dtype=bool)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Straight movements only
    queue = []
    heappush(queue, (0, start, []))  # (cost, current position, path)

    while queue:
        cost, current, path = heappop(queue)
        x, y = current

        if visited[x, y]:
            continue
        visited[x, y] = True

        path = path + [current]
        if current == goal:
            return path

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny] and binary_map[nx, ny] == 0:
                heappush(queue, (cost + 1, (nx, ny), path))

    return None  # No path found

# Visualize the path on the floor plan
def visualize_path(floor_plan, path):
    """Visualize the path on the floor plan."""
    path_image = floor_plan.copy()

    for (x, y) in path:
        cv2.circle(path_image, (y, x), 2, (0, 255, 0), -1)  # Draw the path

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(path_image, cv2.COLOR_BGR2RGB))
    plt.title("Path Simulation")
    plt.axis("off")
    plt.show()

# Simulate wheelchair movement
def simulate_movement(binary_map, path):
    """Simulate movement along the path."""
    plt.figure(figsize=(10, 10))
    plt.title("Wheelchair Movement Simulation")
    plt.imshow(binary_map, cmap="gray")

    for pos in path:
        plt.scatter(pos[1], pos[0], c='red', s=10)
        plt.pause(0.01)

    plt.show()
    print("Simulation complete!")

# Main function
def main():
    image_path = "C:/Users/saisa/OneDrive/Desktop/walls.jpg"
    
    try:
        # Step 1: Select coordinates using the interactive map
        print("Step 1: Select the start and goal coordinates interactively.")
        select_coordinates(image_path)
        
        if len(coords) < 2:
            print("Error: Please select both start and goal coordinates.")
            return
        
        start, goal = coords[:2]
        print(f"Start: {start}, Goal: {goal}")
        
        # Step 2: Load and preprocess the image
        floor_plan = load_image(image_path)
        binary_map = preprocess_image(floor_plan)

        # Step 3: Pathfinding using A*
        path = astar(binary_map, start, goal)
        if path is None:
            print("No valid path found. Check start and goal positions.")
            return

        # Step 4: Visualize and simulate the path
        visualize_path(floor_plan, path)
        simulate_movement(binary_map, path)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
