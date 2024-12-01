import numpy as np
import heapq
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

# Adjustable grid size and obstacle density
GRID_SIZE = (20, 20)
OBSTACLE_DENSITY = 0.2

# Directional Heuristic: Adds a directional bias to the Manhattan distance
def heuristic(a, b, bias=(1, 1)):
    return abs(a[0] - b[0]) * bias[0] + abs(a[1] - b[1]) * bias[1]

# Generate random grid with obstacles
def create_grid(size, density):
    grid = np.zeros(size)
    obstacle_count = int(density * size[0] * size[1])
    obstacles = np.random.choice(size[0] * size[1], obstacle_count, replace=False)
    grid.ravel()[obstacles] = 1  # Place obstacles randomly
    return grid

# A* Algorithm
def astar(grid, start, goal, bias):
    open_list = []
    heapq.heappush(open_list, (0, start))

    cost_so_far = {start: 0}
    came_from = {start: None}
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while open_list:
        _, current_node = heapq.heappop(open_list)
        if current_node == goal:
            break

        for dx, dy in neighbors:
            neighbor = (current_node[0] + dx, current_node[1] + dy)
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                if grid[neighbor[0], neighbor[1]] == 1:
                    continue  # Skip obstacles

                new_cost = cost_so_far[current_node] + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, goal, bias)
                    heapq.heappush(open_list, (priority, neighbor))
                    came_from[neighbor] = current_node

    # Reconstruct path
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = came_from.get(node)
    return path[::-1]

# Multi-goal Pathfinding
def find_multi_goal_path(grid, start, goals, bias):
    all_paths = []  # Full path covering all goals
    current_start = start  # Current position of the agent
    remaining_goals = list(goals)  # Goals left to visit

    while remaining_goals:
        # Find the closest unvisited goal (based on heuristic with bias)
        distances = {goal: heuristic(current_start, goal, bias) for goal in remaining_goals}
        next_goal = min(distances, key=distances.get)

        # Use A* to find the path to the next goal
        path_to_next_goal = astar(grid, current_start, next_goal, bias)

        # If no path is found, mark the goal as unreachable and continue
        if not path_to_next_goal:
            print(f"Goal {next_goal} is unreachable!")
            remaining_goals.remove(next_goal)
            continue

        # Append the path, excluding the starting point to avoid overlap
        if all_paths:
            all_paths.extend(path_to_next_goal[1:])
        else:
            all_paths.extend(path_to_next_goal)

        # Mark the current goal as visited and update the start position
        remaining_goals.remove(next_goal)
        current_start = next_goal

    return all_paths

# Plotting and Animation
def animate_multi_goal(grid, path, goals, start):
    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = ListedColormap(["white", "black", "cyan", "red", "green", "blue", "yellow", "magenta", "orange", "purple"])

    goal_colors = {goal: idx + 6 for idx, goal in enumerate(goals)}  # Unique color index for each goal

    def update(i):
        grid_frame = grid.copy()

        # Mark goals with their unique colors
        for goal, color_index in goal_colors.items():
            grid_frame[goal] = color_index

        # Mark the path up to the current frame
        for idx in range(i + 1):
            x, y = path[idx]
            grid_frame[x, y] = 4  # Path color

        ax.clear()
        ax.imshow(grid_frame, cmap=cmap)
        ax.plot(start[1], start[0], "ro")  # Start point in red

        # Draw the entire final path
        path_x = [y for x, y in path[:i + 1]]
        path_y = [x for x, y in path[:i + 1]]
        ax.plot(path_x, path_y, color="blue", linewidth=2)  # Path line is blue

        # Add text annotations for goal points
        for goal, color_index in goal_colors.items():
            ax.plot(goal[1], goal[0], "o", color=cmap(color_index))  # Goal markers
            ax.text(goal[1], goal[0], f"{goal}", color="black", fontsize=8, ha="center")

        ax.set_title("Multi-Goal A* Pathfinding with Directional Heuristic")

    ani = animation.FuncAnimation(fig, update, frames=len(path), repeat=False)
    plt.show()

# Set up grid, start, and goals
grid = create_grid(GRID_SIZE, OBSTACLE_DENSITY)
start = (0, 0)
goals = {(15, 15), (5, 10), (10, 5), (19, 19)}  # Multiple goal points

# Directional bias (e.g., prioritizing movement towards the bottom-right corner)
directional_bias = (1.5, 1)  # Adjust weights for x and y directions

# Find path covering all goals
path = find_multi_goal_path(grid, start, goals, directional_bias)

# Plot and animate the result
if path:
    animate_multi_goal(grid, path, goals, start)
else:
    print("No path found to cover all goals.")
