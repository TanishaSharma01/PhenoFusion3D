import sys
sys.path.insert(0, ".")

import cv2
import numpy as np
import open3d as o3d
from processing.rgbd import rgbd2pcd
from processing.utils import clean_pcd

rgb_path   = "dataset/living_room_traj1_frei_png/rgb/834.png"
depth_path = "dataset/living_room_traj1_frei_png/depth/834.png"

K = np.array([
    [481.2,    0, 319.5],
    [   0, -480.0, 239.5],
    [   0,    0,     1  ]
])

# Debug: check images loaded correctly
color_raw = cv2.imread(rgb_path)
depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

print(f"RGB loaded:   {color_raw is not None}")
print(f"Depth loaded: {depth is not None}")

if color_raw is None:
    print(f"ERROR: Could not load RGB image at {rgb_path}")
    sys.exit(1)
if depth is None:
    print(f"ERROR: Could not load depth image at {depth_path}")
    sys.exit(1)

print(f"RGB shape:   {color_raw.shape}")
print(f"Depth shape: {depth.shape}")
print(f"Depth dtype: {depth.dtype}")
print(f"Depth min:   {depth.min()}")
print(f"Depth max:   {depth.max()}")

color = cv2.cvtColor(color_raw, cv2.COLOR_BGR2RGB)

pcd = rgbd2pcd(color, depth, K, depth_scale=5000.0, depth_trunc=5.0)
print(f"Points before clean: {len(pcd.points)}")

pcd = clean_pcd(pcd)
print(f"Points after clean:  {len(pcd.points)}")

if len(pcd.points) == 0:
    print("ERROR: Point cloud is empty. Check depth_scale and depth values.")
    sys.exit(1)

o3d.visualization.draw_geometries(
    [pcd],
    window_name="PhenoFusion3D — Single Frame Test",
    width=1024,
    height=768,
)