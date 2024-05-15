## 3D Rotation
- 3-D Rotation around coordinate axes counter-clockwise
$$R_z (\theta) = \begin{bmatrix}
\cos(\theta) && -sin(\theta) && 0 \\
\sin(\theta) && \cos(\theta) && 0 \\
0 && 0 && 1
\end{bmatrix},
R_y (\theta) = \begin{bmatrix}
\cos(\theta) && 0 && \sin(\theta) \\
0 && 1 && 0 \\
-\sin(\theta) && 0 && \cos(\theta) 
\end{bmatrix},
R_x (\theta) = \begin{bmatrix}
1 && 0 && 0 \\
0 && \cos(\theta) && -sin(\theta) \\
0 && \sin(\theta) && \cos(\theta)
\end{bmatrix}
$$

### Rodrigues Formula

For a vector $\mathbf{x}$ and a unit vector $\mathbf{k}$, the rotation of $\mathbf{x}$ by an angle $\theta$ about $\mathbf{k}$ is given by
$$\mathbf{x_{rot}} = \mathbf{x\cos(\theta) + \sin(\theta)(k \times x) + (1-\cos\theta)(k \cdot x)k}$$
- Axis-Angle to $\mathbf{R}$: $\mathbf{R = I  + (\sin\theta)K} + (1-\cos\theta)K^2$, where $$K = \begin{bmatrix} 0 && -k_3 && k_2 \\ k_3 && 0 && -k_1 \\ -k_2 && k_1 && 0\end{bmatrix}$$
- $\mathbf{R}$ to Axis-Angle: $\theta = \cos^{-1}(\frac{Tr(R) - 1}{2})$, $\mathbf{k} = \frac{1}{2\sin\theta} \begin{bmatrix} R_{32} - R_{23} \\ R_{13} - R_{31} \\ R_{21} - R_{12} \end{bmatrix}$

## Camera parameters

- Homogeneous coordinates: $\mathbf{x} = \begin{bmatrix} x && y \end{bmatrix}^T \rightarrow \mathbf{x'} = \begin{bmatrix} x \cdot z && y \cdot z && z \end{bmatrix}^T, z \neq 0$
- Perspective Projection Transformation: $\mathbf{x'} = \begin{bmatrix} 1 && 0 && 0 && 0 \\ 0 && 1 && 0 && 0 \\ 0 && 0 && 1 && 0 \\ \end{bmatrix} \begin{bmatrix} x \\ y \\ z \\ 1\end{bmatrix}$

### Intrinsic Properties
- Principal Axis: Line from the camera centre perpendicular to the image plane. 
- Normalized Camera Coordinate: Camera centre at the origin $(C)$, $x$ and $y$ axes are aligned with the image axes and image plane ($P_z = f$); units in m/cm/ft etc.
- Principal point: point where the principal axis intersects the image plane (z=f)
- Pixel coordinate frame: Origin (0,0) is in the corner of an image; units are in pixels. 
- Skew: Not necessary that image plane axis are perpendicular
$$K = \begin{bmatrix} 1 && s && \\ && 1 && \\ && && 1 \end{bmatrix} \begin{bmatrix} m_x && && \\ && m_y && \\ && && 1 \end{bmatrix} \begin{bmatrix} f && && p_x \\ && f && p_y \\ && && 1 \end{bmatrix} = \begin{bmatrix} \alpha_x &&  s && \beta_x \\ && \alpha_y && \beta_y \\ && && 1\end{bmatrix} $$

$s$ is the skew parameter. Skew parameter is non-zero, only if x and y axes are non-orthogonal, i.e., pixels are not rectangular. $m_x$ and $m_y$ are the scaling factors in x and y directions. $f$ is the focal length, and $p_x$ and $p_y$ are the principal point coordinates.

### Steps to convert world view to pixels
1. World to Camera Coordinate Transformation: $\mathbf{X_{cam}} = \mathbf{R(X-C)}$ [Extrinsic parameters]
2. Perspective Projection: camera 3D to to image plane 2D
3. Scaling and shifting: image plane 2D to pixel 2D [Intrinsic parameters]

$$\mathbf{X' = K[R(X-C)] = K[R|t]X}$$

Here, $\mathbf{K}$ is the intrinsic matrix, $\mathbf{R}$ is the rotation matrix, $\mathbf{t}$ is the translation vector, and $\mathbf{X}$ is the 3D point in the world coordinate system.

### Degrees of Freedom
- Intrinsic: Focal length (2-dof) $f_x$ and $f_y$, Principal point (2-dof) $p_x, p_y$, Skew factor (1-dof) $s$
- Extrinsic: Rotation (3-dof): $R$, Translation (3-dof): $t$
- Total degrees of freedom: (3 + 3) + (2 + 2 + 1) = 11


## Projective Geometry

### Basics
- Planes passing through origin and $\perp$ to vector $\mathbf{n}$: $\mathbf{n}\cdot\mathbf{x} = 0$ i.e. $ax_1 + bx_2 + cx_3 = 0$
- Vector $\parallel$ to intersection of 2 planes $(a, b, c)$ and $(a', b', c')$: $\mathbf{n''} = \mathbf{n} \times \mathbf{n'}$
- Planes passing through two points $\mathbf{x}$ and $\mathbf{x'}$: $\mathbf{n} = \mathbf{x} \times \mathbf{x'}$
- To each point $m$ of the plane $P$ we can associate a single ray $\mathbf{x} = (x_1, x_2, x_3)$
- To each line $l$ of the plane $P$ we can associate a single point $\mathbf{l} = (l_1, l_2, l_3)$

We can go in reverse as well
- If we have a line $ax + by + c = 0$, then the point $\mathbf{l} = (a, b, c)$ is the corresponding plane in 3D, and if $x = \begin{bmatrix} x_1 && x_2 \end{bmatrix}^T \in l$, then 
$\begin{bmatrix} x_1 \\ x_2 \\ 1 \end{bmatrix}^T \begin{bmatrix} a \\ b \\ c \end{bmatrix} = 0$
- Point of intersection of 2 lines $x = l_1 \times l_2$
- Line at infinity: $l_{\infty} = (0, 0, 1)^T$, Point at infinity is always of the form: $\mathbf{x_{\infty}} = (a, b, 0)^T$
- For a line $l = \begin{bmatrix} a && b && c \end{bmatrix}^T$, the point at infinity is $\mathbf{x_{\infty}} = (b, -a, 0)^T$