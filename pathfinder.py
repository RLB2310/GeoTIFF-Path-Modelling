# Provides the binary heap algorithm used for A*
import heapq

# Handles GeoTIFF
import rasterio

# Fast array operations, slope, matrix gradient
import numpy as np


# All spatial data and pathfinding logic
class SurfacePathfinder:
    def __init__(self, tif_path="tif_crops/Mars_HRSC_MOLA.tif", max_dim=400):

        self.tif_path = tif_path
        self.set_max_dim(max_dim)

    def set_max_dim(self, max_dim):
        """Re-scales matrix and re-computes terrain costs for a new grid resolution."""
        with rasterio.open(self.tif_path) as src:
            self.transform = src.transform

            # Calculate target dimensions
            self.scale = min(1.0, max_dim / max(src.width, src.height))
            target_h = int(src.height * self.scale)
            target_w = int(src.width * self.scale)

            # Read elevation array
            self.elevation = src.read(1, out_shape=(target_h, target_w)).astype(float)

            # IMPORTANT: Sync height and width directly from the actual loaded array shape
            self.height, self.width = self.elevation.shape

        # Adjusted pixel resolution in meters (assuming 200m base scale)
        pixel_size = 200.0 / self.scale
        dy, dx = np.gradient(self.elevation, pixel_size)
        slope_deg = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

        # Base traversal cost map (slopes > 15 degrees marked impassable)
        self.cost_map = 1.0 + (slope_deg / 3.0) ** 2
        self.cost_map[slope_deg > 15.0] = np.inf

    # Converts geographic degrees to downsampled matrix grid indices
    def latlon_to_grid(self, lat, lon):
        # Queries the rasterio transform, to find raw image coords
        row, col = rasterio.transform.rowcol(self.transform, lon, lat)

        # Scales raw coords down to match the downsampled grid size
        r = int(row * self.scale)
        c = int(col * self.scale)

        # Places indices inside matrix array boundaries
        return max(0, min(r, self.height - 1)), max(0, min(c, self.width - 1))

    # Inverts matrix indices back to geographic (lat/lon) degrees
    def grid_to_latlon(self, r, c):
        orig_r = int(r / self.scale)
        orig_c = int(c / self.scale)
        lon, lat = rasterio.transform.xy(self.transform, orig_r, orig_c)
        return lat, lon

    # Main solving method
    def find_path(self, start_lat, start_lon, end_lat, end_lon, effective):

        # Converts input degrees into starting and ending grid cells
        start = self.latlon_to_grid(start_lat, start_lon)
        goal = self.latlon_to_grid(end_lat, end_lon)

        # 8-neighbor movements | Defines 8-way directional offsets (dr, dc) and movement step distances (1.0 for orthognal,  sqr(2) for diagonal
        neighbors = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, 1.414),
            (-1, 1, 1.414),
            (1, -1, 1.414),
            (1, 1, 1.414),
        ]

        # Priority queue holding nodes to evaluate, ordered by f(n) = g(n) + h(n)
        open_set = []

        # Pushes starting node into queue with priority 0
        heapq.heappush(open_set, (0, start))

        # Hash map tracking parent pointers for path reconstruction
        came_from = {}

        # Hash map storing exact accumulated movement cost from start to each node
        g_score = {start: 0.0}

        # Collection for tracking frontier search history for animation
        explored_nodes = []
        step_count = 0

        # Main evaluation loop running while a candidate node exists
        while open_set:
            # Pops node with lowest estimated cost f(n)
            _, current = heapq.heappop(open_set)

            # Sample every 2nd evaluated node for the UI frontier display | generate frontier visualisation dots
            step_count += 1
            if step_count % 3 == 0:
                explored_nodes.append(self.grid_to_latlon(*current))

            # Triggers path reconstruction when destination reached
            if current == goal:
                # Reconstruct path
                path = []
                curr = goal

                # Backtracks through parent pointers from destination back to start
                while curr in came_from:
                    path.append(curr)
                    curr = came_from[curr]
                path.append(start)
                path.reverse()

                # Converts winning grid nodes into geographic degrees and returns both the explored nodes list and winning path list
                final_path_geo = [self.grid_to_latlon(r, c) for r, c in path]
                return explored_nodes, final_path_geo

            for dr, dc, move_cost in neighbors:
                # Computes neighbor grid coordinates
                nr, nc = current[0] + dr, current[1] + dc
                # Verifies neighbor is within grid boundaries
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    terrain_cost = self.cost_map[nr, nc]
                    # Skips >15 degree slopes
                    if np.isinf(terrain_cost):
                        continue

                    # Calculates path cost to reach neighbor
                    tentative_g = g_score[current] + (move_cost * terrain_cost)
                    if effective:
                        # Calculate elevation difference
                        z_start = self.elevation[current[0], current[1]]
                        z_end = self.elevation[nr, nc]

                        # Ensure points are valid elevation data (not NoData borders)
                        if z_start > -5000 and z_end > -5000:
                            dz = z_end - z_start
                            if dz > 0:
                                # Use a small scale weight (0.01 - 0.05) so A* doesn't lock up
                                climb_weight = 0.06
                                tentative_g += dz * climb_weight
                    neighbor = (nr, nc)
                    # If a cheaper route to neighbor is found
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        # Euclidean distance heuristic to goal
                        h = np.hypot(nr - goal[0], nc - goal[1])

                        # Pushes neighbor into priority queue
                        heapq.heappush(open_set, (tentative_g + h, neighbor))

        return explored_nodes, []
