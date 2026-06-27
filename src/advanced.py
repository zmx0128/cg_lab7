import taichi as ti
import math

ti.init(arch=ti.gpu)

N = 20
mass = 1.0
dt = 5e-4
k_s = 10000.0
k_shear = 5000.0
k_bending = 1000.0
k_d = 1.0
gravity = ti.Vector([0.0, -9.8, 0.0])
max_velocity = 50.0

sphere_center = ti.Vector.field(3, dtype=float, shape=())
sphere_radius = ti.field(dtype=float, shape=())

x = ti.Vector.field(3, dtype=float, shape=N * N)
v = ti.Vector.field(3, dtype=float, shape=N * N)
f = ti.Vector.field(3, dtype=float, shape=N * N)
is_fixed = ti.field(dtype=int, shape=N * N)

x_next = ti.Vector.field(3, dtype=float, shape=N * N)
v_next = ti.Vector.field(3, dtype=float, shape=N * N)
f_next = ti.Vector.field(3, dtype=float, shape=N * N)

max_springs = N * N * 8
spring_indices = ti.field(dtype=int, shape=max_springs * 2)
spring_pairs = ti.Vector.field(2, dtype=int, shape=max_springs)
spring_lengths = ti.field(dtype=float, shape=max_springs)
spring_stiffness = ti.field(dtype=float, shape=max_springs)
num_springs = ti.field(dtype=int, shape=())

sphere_resolution = 20
sphere_vertices = ti.Vector.field(3, dtype=float, shape=sphere_resolution * sphere_resolution * 2)
sphere_indices = ti.field(dtype=int, shape=sphere_resolution * sphere_resolution * 6)
num_sphere_triangles = ti.field(dtype=int, shape=())

def generate_sphere():
    global num_sphere_triangles
    idx = 0
    for i in range(sphere_resolution):
        theta = math.pi * i / (sphere_resolution - 1)
        for j in range(sphere_resolution):
            phi = 2 * math.pi * j / sphere_resolution
            x_sphere = math.sin(theta) * math.cos(phi)
            y_sphere = math.cos(theta)
            z_sphere = math.sin(theta) * math.sin(phi)
            sphere_vertices[idx] = ti.Vector([x_sphere, y_sphere, z_sphere])
            idx += 1
    tri_idx = 0
    for i in range(sphere_resolution - 1):
        for j in range(sphere_resolution):
            v0 = i * sphere_resolution + j
            v1 = (i + 1) * sphere_resolution + j
            v2 = (i + 1) * sphere_resolution + ((j + 1) % sphere_resolution)
            v3 = i * sphere_resolution + ((j + 1) % sphere_resolution)
            sphere_indices[tri_idx] = v0
            sphere_indices[tri_idx + 1] = v1
            sphere_indices[tri_idx + 2] = v2
            tri_idx += 3
            sphere_indices[tri_idx] = v0
            sphere_indices[tri_idx + 1] = v2
            sphere_indices[tri_idx + 2] = v3
            tri_idx += 3
    num_sphere_triangles[None] = tri_idx // 3

@ti.kernel
def update_sphere_render():
    center = sphere_center[None]
    radius = sphere_radius[None]
    for i in range(sphere_resolution * sphere_resolution):
        sphere_vertices[i] = sphere_vertices[i] * radius + center

@ti.kernel
def init_positions():
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        x[idx] = ti.Vector([i * 0.05 - 0.5, 0.8, j * 0.05 - 0.5])
        v[idx] = ti.Vector([0.0, 0.0, 0.0])
        f[idx] = ti.Vector([0.0, 0.0, 0.0])
        if j == 0 and (i == 0 or i == N - 1):
            is_fixed[idx] = 1
        else:
            is_fixed[idx] = 0
    sphere_center[None] = ti.Vector([0.0, -0.2, 0.0])
    sphere_radius[None] = 0.35

@ti.kernel
def init_springs():
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        if i < N - 1:
            idx_right = (i + 1) * N + j
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_right])
            spring_lengths[c] = (x[idx] - x[idx_right]).norm()
            spring_stiffness[c] = k_s
        if j < N - 1:
            idx_down = i * N + (j + 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_down])
            spring_lengths[c] = (x[idx] - x[idx_down]).norm()
            spring_stiffness[c] = k_s
        if i < N - 1 and j < N - 1:
            idx_diag = (i + 1) * N + (j + 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_diag])
            spring_lengths[c] = (x[idx] - x[idx_diag]).norm()
            spring_stiffness[c] = k_shear
            idx_diag2 = (i + 1) * N + (j - 1)
            if j > 0:
                c = ti.atomic_add(num_springs[None], 1)
                spring_pairs[c] = ti.Vector([idx, idx_diag2])
                spring_lengths[c] = (x[idx] - x[idx_diag2]).norm()
                spring_stiffness[c] = k_shear
        if i < N - 2:
            idx_skip = (i + 2) * N + j
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_skip])
            spring_lengths[c] = (x[idx] - x[idx_skip]).norm()
            spring_stiffness[c] = k_bending
        if j < N - 2:
            idx_skip_j = i * N + (j + 2)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_skip_j])
            spring_lengths[c] = (x[idx] - x[idx_skip_j]).norm()
            spring_stiffness[c] = k_bending

@ti.kernel
def init_spring_indices():
    for i in range(num_springs[None]):
        spring_indices[i * 2] = spring_pairs[i][0]
        spring_indices[i * 2 + 1] = spring_pairs[i][1]

def init_cloth():
    num_springs[None] = 0
    init_positions()
    init_springs()
    init_spring_indices()
    generate_sphere()
    update_sphere_render()

@ti.func
def compute_forces_on(pos: ti.template(), vel: ti.template(), force: ti.template()):
    for i in range(N * N):
        force[i] = gravity * mass - k_d * vel[i]
    for i in range(num_springs[None]):
        idx_a = spring_pairs[i][0]
        idx_b = spring_pairs[i][1]
        pos_a = pos[idx_a]
        pos_b = pos[idx_b]
        d = pos_a - pos_b
        dist = d.norm()
        if dist > 1e-6:
            d_normalized = d / dist
            f_spring = -spring_stiffness[i] * (dist - spring_lengths[i]) * d_normalized
            ti.atomic_add(force[idx_a], f_spring)
            ti.atomic_add(force[idx_b], -f_spring)

@ti.func
def clamp_velocity(vel: ti.template(), idx: int):
    vel_norm = vel[idx].norm()
    if vel_norm > max_velocity:
        vel[idx] = vel[idx] / vel_norm * max_velocity

@ti.func
def handle_sphere_collision(pos: ti.template(), vel: ti.template()):
    center = sphere_center[None]
    radius = sphere_radius[None]
    for i in range(N * N):
        if is_fixed[i] == 0:
            diff = pos[i] - center
            dist = diff.norm()
            if dist < radius:
                normal = diff / dist
                pos[i] = center + normal * radius
                vel[i] -= ti.min(vel[i].dot(normal), 0.0) * normal

@ti.kernel
def step_explicit():
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            x[i] += v[i] * dt
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)
    handle_sphere_collision(x, v)

@ti.kernel
def step_semi_implicit():
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)
            x[i] += v[i] * dt
    handle_sphere_collision(x, v)

@ti.kernel
def step_implicit_iter():
    for i in range(N * N):
        v_next[i] = v[i]
        x_next[i] = x[i]
    for _ in ti.static(range(3)):
        compute_forces_on(x_next, v_next, f_next)
        for i in range(N * N):
            if is_fixed[i] == 0:
                v_next[i] = v[i] + (f_next[i] / mass) * dt
                clamp_velocity(v_next, i)
                x_next[i] = x[i] + v_next[i] * dt
        handle_sphere_collision(x_next, v_next)
    for i in range(N * N):
        v[i] = v_next[i]
        x[i] = x_next[i]

def main():
    init_cloth()

    window = ti.ui.Window("Cloth with Shear, Bending & Sphere Collision", (800, 800))
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0.0, 0.5, 2.0)
    camera.lookat(0.0, 0.0, 0.0)

    current_method = 1
    paused = False

    while window.running:
        window.GUI.begin("Control Panel", 0.02, 0.02, 0.38, 0.36)

        window.GUI.text("Integration Method:")

        prefix_0 = "[*] " if current_method == 0 else "[ ] "
        prefix_1 = "[*] " if current_method == 1 else "[ ] "
        prefix_2 = "[*] " if current_method == 2 else "[ ] "

        if window.GUI.button(prefix_0 + "Explicit Euler (Explosive)"):
            current_method = 0
            init_cloth()
        if window.GUI.button(prefix_1 + "Semi-Implicit Euler (Stable)"):
            current_method = 1
            init_cloth()
        if window.GUI.button(prefix_2 + "Implicit Euler (Damped)"):
            current_method = 2
            init_cloth()

        window.GUI.text("")

        pause_label = "Resume Simulation" if paused else "Pause Simulation"
        if window.GUI.button(pause_label):
            paused = not paused

        if window.GUI.button("Reset Cloth"):
            init_cloth()

        window.GUI.text("")
        window.GUI.text("Spring Stiffness:")
        global k_s, k_shear, k_bending
        k_s = window.GUI.slider_float("Structural", k_s, 1000.0, 20000.0)
        k_shear = window.GUI.slider_float("Shear", k_shear, 100.0, 10000.0)
        k_bending = window.GUI.slider_float("Bending", k_bending, 10.0, 5000.0)

        window.GUI.end()

        if not paused:
            for _ in range(40):
                if current_method == 0:
                    step_explicit()
                elif current_method == 1:
                    step_semi_implicit()
                elif current_method == 2:
                    step_implicit_iter()

        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        scene.mesh(sphere_vertices, indices=sphere_indices, color=(0.4, 0.4, 0.4))
        scene.particles(x, radius=0.012, color=(0.95, 0.95, 0.98))
        scene.lines(x, indices=spring_indices, width=1.0, color=(0.85, 0.85, 0.9))

        canvas.scene(scene)

        window.show()

if __name__ == '__main__':
    main()
