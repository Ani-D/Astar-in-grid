import numpy as np
import heapq
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

# Adjustable grid size and obstacle density
GRID_SIZE = (24, 24)  # Larger grid
OBSTACLE_DENSITY = 0.3  # 30% of cells as obstacles

# Heuristic function (Manhattan distance)
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Generate random grid with obstacles
def create_grid(size, density):
    grid = np.zeros(size)
    obstacle_count = int(density * size[0] * size[1])
    obstacles = np.random.choice(size[0] * size[1], obstacle_count, replace=False)
    grid.ravel()[obstacles] = 1  # Place obstacles randomly
    return grid

# A* Algorithm with animation updates
def astar(grid, start, goal, ax):
    open_list = []
    heapq.heappush(open_list, (0, start))

    cost_so_far = {start: 0}
    came_from = {start: None}
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]

    frames = []  # Store frames for animation

    while open_list:
        current_cost, current_node = heapq.heappop(open_list)
        if current_node == goal:
            break

        # Visualize the current cell being explored
        if current_node != start:
            grid[current_node] = 0.5  # Mark as explored
            frames.append((grid.copy(), current_node))

        for dx, dy in neighbors:
            neighbor = (current_node[0] + dx, current_node[1] + dy)

            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                if grid[neighbor[0], neighbor[1]] == 1:
                    continue  # Skip obstacles

                new_cost = cost_so_far[current_node] + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, goal)
                    heapq.heappush(open_list, (priority, neighbor))
                    came_from[neighbor] = current_node

    # Verify if goal is reachable
    if goal not in came_from:
        print("Goal is unreachable. Please try again with a different configuration.")
        return [], frames

    # Reconstruct path from start to goal
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = came_from[node]

    return path[::-1], frames

# Plotting and Animation
def animate_path(grid, path, frames, start, goal):
    fig, ax = plt.subplots(figsize=(10, 10))  # Adjust figsize for larger plot
    cmap = ListedColormap(["white", "black", "cyan", "red", "green", "yellow"])

    def update(i):
        if i < len(frames):
            grid_frame, _ = frames[i]
        else:
            grid_frame = grid.copy()
            for (x, y) in path:
                grid_frame[x, y] = 4  # Path color
        ax.clear()
        ax.imshow(grid_frame, cmap=cmap)
        ax.plot(start[1], start[0], "ro")  # Start is red
        ax.plot(goal[1], goal[0], "go")    # Goal is green

        # Draw the path line
        if i >= len(frames):  # Only draw the line once the path is complete
            path_x = [y for x, y in path]
            path_y = [x for x, y in path]
            ax.plot(path_x, path_y, color="blue", linewidth=2)  # Path line is blue
            
        ax.set_title("A* Pathfinding")

    ani = animation.FuncAnimation(fig, update, frames=len(frames) + len(path), repeat=False)
    plt.show()

# Set up the grid, start, and goal
grid = create_grid(GRID_SIZE, OBSTACLE_DENSITY)
start = (0, 0)
goal = (GRID_SIZE[0] - 1, GRID_SIZE[1] - 1)

# Run A* and animate the pathfinding
path, frames = astar(grid, start, goal, None)
if path:
    animate_path(grid, path, frames, start, goal)
else:
    print("No path found. Please try again.")
