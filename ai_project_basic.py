import numpy as np
import heapq
import matplotlib.pyplot as plt

# Grid map definition (0 = free, 1 = obstacle)
grid = np.array([
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0]
])

# Define start and goal points
start = (0, 0)  # Top-left corner
goal = (5, 7)   # Bottom-right corner

# Heuristic function (Manhattan distance)
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# A* Algorithm Implementation
def astar(grid, start, goal):
    open_list = []
    heapq.heappush(open_list, (0, start))

    cost_so_far = {start: 0}
    came_from = {start: None}
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while open_list:
        current_cost, current_node = heapq.heappop(open_list)

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
                    priority = new_cost + heuristic(neighbor, goal)
                    heapq.heappush(open_list, (priority, neighbor))
                    came_from[neighbor] = current_node

    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = came_from[node]
    
    return path[::-1]

# Visualization
def plot_path(grid, path, start, goal):
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap=plt.cm.Dark2)

    ax.plot(start[1], start[0], "ro")  # Start is red
    ax.plot(goal[1], goal[0], "go")    # Goal is green

    for (x, y) in path:
        ax.plot(y, x, "bo")  # Path is blue

    plt.show()

# Test A* on the grid
path = astar(grid, start, goal)
plot_path(grid, path, start, goal)
