# Custom GeoTIFF Path finding program by Rylan B (rlb23.dev)

import os
import tkinter as tk
import numpy as np
import threading
from PIL import Image, ImageTk
from pathfinder import SurfacePathfinder
import rasterio
import stringprep

# Hard coded QGIS map bounds
# For full_map
NORTH = 96.500170907
EAST = 182.148331752
SOUTH = -102.065846752
WEST = -177.006596362
# For map2
# WEST = -154.589984561
# NORTH = 27.283727603
# EAST = -89.914616581
# SOUTH = -15.771323677


class MainProgram:
    # Take the tkinter root as an arg
    def __init__(self, root, image_path):
        # Setup window
        self.root = root
        self.label = tk.Label(root, text="Rylan's A* GeoTIFF Path Finder")
        self.label.pack()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Need to instantiate the background solver here
        self.solver = SurfacePathfinder("tif_crops/Mars_HRSC_MOLA.tif")

        # Load the image for the map
        self.raw_image = Image.open(image_path)
        self.img_w, self.img_h = self.raw_image.size

        # Convert PIL to native TK format
        self.tk_image = ImageTk.PhotoImage(self.raw_image)

        # effective determines whether elevation should be avoided
        self.effective = False

        # State of explored paths showing
        self.show_explored_var = tk.BooleanVar(value=True)

        # Set the point variable

        self.start_point = None
        self.end_point = None
        self.point_elevation = None
        self.start_marker = None
        self.end_marker = None
        self.path_line = None
        self.path_points = []
        self.explored_canvas_pts = []

        # Toolbar for TK window, buttons, status label
        self.toolbar = tk.Frame(self.root)
        self.toolbar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        self.btn_reset = tk.Button(
            self.toolbar, text="Clear Points", command=self.reset_points
        )
        self.btn_reset.pack(side=tk.LEFT, padx=5)

        self.btn_run = tk.Button(
            self.toolbar, text="Run pathfinding", command=self.run_pathfinder
        )
        self.btn_run.pack(side=tk.LEFT, padx=5)
        self.status_label = tk.Label(
            self.toolbar,
            text="Select a start and end point",
            font=("Arial", 10, "bold"),
        )
        self.status_label.pack(side=tk.RIGHT, padx=10)

        # Effectiveness selection
        self.opt = tk.StringVar(value="Allow Incline/Declines")

        # Radio buttons
        for lang in ["Allow Incline/Declines", "Prioritise safety"]:
            tk.Radiobutton(self.toolbar, text=lang, variable=self.opt, value=lang).pack(
                side=tk.LEFT, padx=5
            )

        # Smoothing / Sampling

        self.lbl_res = tk.Label(self.toolbar, text="Scaling (higher = more detail):")
        self.lbl_res.pack(side=tk.LEFT, padx=(10, 2))

        # Spinbox with quick increments (100 to 2000, default 400)
        self.spin_res = tk.Spinbox(
            self.toolbar, from_=100, to=2000, increment=100, width=5
        )
        self.spin_res.delete(0, "end")
        self.spin_res.insert(0, "400")

        self.spin_res.pack(side=tk.LEFT, padx=5)

        # Add checkbutton to toolbar
        self.chk_explored = tk.Checkbutton(
            self.toolbar,
            text="Show All Search Paths",
            variable=self.show_explored_var,
            command=self.toggle_explored_nodes,
        )
        self.chk_explored.pack(side=tk.LEFT, padx=5)
        # Image explored_canvas
        self.canvas = tk.Canvas(self.root, width=self.img_w, height=self.img_h)
        self.canvas.pack(side=tk.BOTTOM)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

    def canvas_to_latlon(self, cx, cy):
        lon = WEST + (cx / self.img_w) * (EAST - WEST)
        lat = NORTH - (cy / self.img_h) * (NORTH - SOUTH)

        return lat, lon

    def latlon_to_canvas(self, lat, lon):
        cx = ((lon - WEST) / (EAST - WEST)) * self.img_w
        cy = ((NORTH - lat) / (NORTH - SOUTH)) * self.img_h

        return cx, cy

    def toggle_explored_nodes(self):
        """Hide or show all drawn search frontier dots instantly."""
        state = "normal" if self.show_explored_var.get() else "hidden"
        self.canvas.itemconfigure("explored_node", state=state)

    def on_canvas_click(self, event):

        # get click position
        x, y = event.x, event.y

        # convert to lat/lon
        lat, lon = self.canvas_to_latlon(x, y)

        # Check if its first or second click
        if self.start_point is None:
            # Set start point coords
            self.start_point = (x, y)

            # Radius of circle
            r = 6

            # Draw
            self.start_marker = self.canvas.create_oval(
                x - r, y - r, x + r, y + r, fill="lime", width=2
            )

            self.status_label.config(
                text=f"Start: {lat:.3f}°N, {lon:.3f}°E | Click END point"
            )
        elif self.end_point is None:
            # Set end point coords
            self.end_point = (x, y)

            # Radius of circle
            r = 6

            # Draw
            self.end_marker = self.canvas.create_oval(
                x - r, y - r, x + r, y + r, fill="red", width=2
            )

            self.status_label.config(
                text=f"End: {lat:.3f}°N, {lon:.3f}°E | Ready To Run"
            )

    def run_pathfinder(self):
        # Check if point are selected
        if not self.end_point:
            self.status_label.config(
                text="[ERROR] Select a start and end point before running"
            )
            return

        self.clear_drawn_paths()

        self.status_label.config(text="Exploring possible paths...")
        self.btn_run.config(state=tk.DISABLED)

        thread = threading.Thread(target=self._compute_path_thread)

        thread.daemon = True
        thread.start()

    def _compute_path_thread(self):
        try:
            user_max_dim = int(self.spin_res.get())
            user_max_dim = max(
                100, min(user_max_dim, 2000)
            )  # Clamp between 100 and 2000
        except ValueError:
            user_max_dim = 400
        self.solver.set_max_dim(user_max_dim)

        start_lat, start_lon = self.canvas_to_latlon(*self.start_point)
        end_lat, end_lon = self.canvas_to_latlon(*self.end_point)
        self.effective = self.opt.get()
        if self.effective == "Prioritise safety":
            self.effective = True
        else:
            self.effective = False

        explored_geo, geo_path = self.solver.find_path(
            start_lat,
            start_lon,
            end_lat,
            end_lon,
            self.effective,
        )

        self.root.after(0, lambda: self._on_path_computed(explored_geo, geo_path))

    def _on_path_computed(self, explored_geo, geo_path):
        self.btn_run.config(state=tk.NORMAL)

        self.explored_canvas_pts = [
            self.latlon_to_canvas(lat, lon) for lat, lon in explored_geo
        ]
        self.path_points = [self.latlon_to_canvas(lat, lon) for lat, lon in geo_path]

        # Animate expanding search (orange dots)
        self.animate_explored_paths(index=0)

    def animate_explored_paths(self, index=0):
        BATCH_SIZE = 50

        end_idx = min(index + BATCH_SIZE, len(self.explored_canvas_pts))

        initial_state = "normal" if self.show_explored_var.get() else "hidden"

        for cx, cy in self.explored_canvas_pts[index:end_idx]:
            r = 1.5
            self.canvas.create_oval(
                cx - r,
                cy - r,
                cx + r,
                cy + r,
                fill="orange",
                outline="",
                tag="explored_node",
            )
        if end_idx < len(self.explored_canvas_pts):
            self.root.after(8, lambda: self.animate_explored_paths(end_idx))
        else:
            # animate winning path
            self.status_label.config(text="Drawing best path...")
            self.path_line = self.canvas.create_line(
                self.start_point[0],
                self.start_point[1],
                self.start_point[0],
                self.start_point[1],
                fill="cyan",
                width=3,
            )

            # redraw marks
            if self.start_marker:
                self.canvas.tag_raise(self.start_marker)

            if self.end_marker:
                self.canvas.tag_raise(self.end_marker)
            self.animate_path_step(current_step=1)

    def animate_path_step(self, current_step):
        if current_step < len(self.path_points):
            flat_coords = [
                coord for pt in self.path_points[: current_step + 1] for coord in pt
            ]
            self.canvas.coords(self.path_line, *flat_coords)
            self.root.after(15, lambda: self.animate_path_step(current_step + 1))
        else:
            self.status_label.config(text="Pathing complete")

    def clear_drawn_paths(self):
        self.canvas.delete("explored_node")
        if self.path_line:
            self.canvas.delete(self.path_line)
            self.path_line = None

    def reset_points(self):
        if self.start_marker:
            self.canvas.delete(self.start_marker)
        if self.end_marker:
            self.canvas.delete(self.end_marker)

        self.clear_drawn_paths()

        self.start_point = None
        self.end_point = None
        self.start_marker = None
        self.end_marker = None
        self.path_points = []
        self.explored_canvas_pts = []
        self.status_label.config(text="Click map to set START point")

    def on_close(self):
        self.root.destroy()
        os._exit(0)


if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = MainProgram(root, "mars_full_map.png")
        root.mainloop()
    except KeyboardInterrupt:
        os._exit(0)
