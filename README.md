# Computer Vision Lab

Classical computer vision algorithms implemented from scratch in Python.
The core math (least squares, sparse linear systems, homography, RANSAC) is written with NumPy/SciPy; OpenCV is used only for image I/O, SIFT keypoints and final warping.

| Project | What it does |
|---|---|
| [Photometric Stereo](photometric_stereo/) | Recovers surface normals and a 3D mesh from images taken under different lighting |
| [Image Stitching](image_stitching/) | Builds a panorama from overlapping photos, with blending and exposure correction |

## Photometric Stereo

| | Normal map | Depth map |
|---|---|---|
| bunny | ![](photometric_stereo/results/bunny/normal.png) | ![](photometric_stereo/results/bunny/depth.png) |
| star | ![](photometric_stereo/results/star/normal.png) | ![](photometric_stereo/results/star/depth.png) |
| venus | ![](photometric_stereo/results/venus/normal.png) | ![](photometric_stereo/results/venus/depth.png) |
| noisy venus | ![](photometric_stereo/results/noisy_venus/normal.png) | ![](photometric_stereo/results/noisy_venus/depth.png) |

## Image Stitching

Panorama with cylindrical projection and intensity-weighted blending:

![](image_stitching/results/base_cylindrical_intensity_weight.png)

Six photos with different exposures, stitched after gain compensation:

![](image_stitching/results/gain_compensation.png)

See each project folder for method details.
