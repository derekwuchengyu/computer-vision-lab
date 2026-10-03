# Photometric Stereo

Reconstruct a 3D surface from 6 grayscale images of the same object lit from 6 known directions (`data/<case>/LightSource.txt`).
Sample objects: `bunny` (120×120), `star` (240×240), `venus` (120×212), and `noisy_venus` (venus with Gaussian noise).

## Method

1. **Normal estimation.** Under the Lambertian model, each pixel gives 6 equations `I = L · (k_d n)` for 3 unknowns.
   I solve this over-determined system for all pixels at once with least squares, `(LᵀL)⁻¹LᵀI`, and normalize the result to get unit normals. Near-zero vectors (background) are left unscaled.
2. **Depth from normals.** A normal must be orthogonal to the surface tangents along x and y, which gives two linear equations per pixel:
   `n_z (z(x+1,y) − z(x,y)) = −n_x` and `n_z (z(x,y+1) − z(x,y)) = −n_y`.
   - Only pixels inside a foreground mask become unknowns, which shrinks the system.
   - At the right and bottom edges I switch to the backward neighbor, so boundary pixels still get constraints.
   - The `2S × S` matrix is stored as a SciPy sparse matrix and solved with `spsolve` on the normal equations. This is faster and more stable than a dense pseudo-inverse.
3. **Outlier removal.** Depth values whose z-score exceeds a per-object threshold are reset, which removes spikes caused by extreme normals (mainly on `venus`).
4. **Noisy input.**
   - Foreground mask from either an intensity threshold (> 60) or the clean `venus` mask.
   - Each normal is replaced by the mean of itself and its 4 neighbors before building the system. Without this smoothing the system could not be solved reliably.

## Run

```bash
pip install numpy scipy opencv-python open3d matplotlib
python photometric_stereo.py              # run all 4 objects
python photometric_stereo.py -t bunny     # or one: bunny / star / venus / noisy_venus
```

Outputs go to `results/<case>/`: `normal.png`, `depth.png`, `<case>.ply`, plus the raw `.npy` maps.
