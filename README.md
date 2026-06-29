# 质点弹簧模型实验

## 📋 一、实验概述

### 🏷️ 1.1 实验名称
质点弹簧模型（Mass-Spring Model）

### 💻 1.2 实验环境
- **操作系统**：Windows 11
- **编程语言**：Python 3.12
- **核心框架**：Taichi 1.7.4
- **构建工具**：uv 0.4.x
- **GPU 架构**：CUDA

### 📁 1.3 项目结构
```
cg_lab7/
├── src/
│   ├── test.py        # 基础版：结构弹簧 + 三种积分方法
│   └── advanced.py    # 高级版：补充剪切弹簧、弯曲弹簧和球体碰撞
├── pyproject.toml     # 项目配置
├── .gitignore         # Git 忽略配置
└── README.md          # 项目说明
```

---

## 🎯 二、实验目标

1. **掌握动态场景渲染**：使用 Taichi 框架构建 3D 场景，学习 Taichi GGUI 交互面板的编写。
2. **理解质点-弹簧模型**：掌握基于物理的弹力与阻尼力计算方法，处理数值爆炸问题。
3. **对比数值积分方法**：实现显式欧拉、半隐式欧拉、隐式欧拉三种积分求解器，观察稳定性差异。
4. **理解 GPU 编程基础**：学习 Taichi 中的 `ti.kernel` 与 `ti.func`，了解并行计算中的状态同步。

---

## 🧮 三、实验原理

### 🔗 3.1 质点-弹簧模型

质点-弹簧系统是计算机图形学中经典的变形体模拟方法。将布料离散化为网格状的质点集合，质点之间通过弹簧相连。

**胡克定律（弹力公式）**：
$$f_{a} = -k_{s} (|x_a - x_b| - l) \frac{x_a - x_b}{|x_a - x_b|}$$
其中：
- $k_s$ 为弹簧的劲度系数
- $l$ 为弹簧的原长
- $x$ 为质点位置

**阻尼力公式**：
$$f_{d} = -k_{d} v_{a}$$
引入阻尼力防止系统能量无限增加导致发散。

### 🔢 3.2 数值积分方法

根据牛顿第二定律 $F = ma$，质点的加速度 $a = F/m$。在离散时间步 $\Delta t$ 内，通过数值积分更新速度 $v$ 和位置 $x$。

| 方法 | 位置更新 | 速度更新 | 稳定性 |
|------|----------|----------|--------|
| 显式欧拉 | $x_{t+1} = x_t + v_t \Delta t$ | $v_{t+1} = v_t + a_t \Delta t$ | 不稳定，易发散 |
| 半隐式欧拉 | $x_{t+1} = x_t + v_{t+1} \Delta t$ | $v_{t+1} = v_t + a_t \Delta t$ | 较稳定，常用 |
| 隐式欧拉 | $x_{t+1} = x_t + v_{t+1} \Delta t$ | $v_{t+1} = v_t + a_{t+1} \Delta t$ | 最稳定，有阻尼效果 |

### ⚡ 3.3 弹簧类型

本实验实现了三种弹簧类型：

| 弹簧类型 | 连接方式 | 作用 | 刚度系数 |
|----------|----------|------|----------|
| 结构弹簧（Structural） | 水平/垂直相邻质点 | 维持布料基本形状 | $k_s = 10000$ |
| 剪切弹簧（Shear） | 对角相邻质点 | 防止剪切变形 | $k_{shear} = 5000$ |
| 弯曲弹簧（Bending） | 隔一个质点的水平/垂直连接 | 控制弯曲程度 | $k_{bending} = 1000$ |

---

## 🔧 四、核心实现

### 📐 4.1 数据结构设计

```python
# 质点状态
x = ti.Vector.field(3, dtype=float, shape=N * N)      # 位置
v = ti.Vector.field(3, dtype=float, shape=N * N)      # 速度
f = ti.Vector.field(3, dtype=float, shape=N * N)      # 受力
is_fixed = ti.field(dtype=int, shape=N * N)           # 是否固定

# 弹簧结构
spring_pairs = ti.Vector.field(2, dtype=int, shape=max_springs)   # 弹簧连接的两个质点
spring_lengths = ti.field(dtype=float, shape=max_springs)         # 弹簧原长
spring_stiffness = ti.field(dtype=float, shape=max_springs)       # 弹簧刚度系数
num_springs = ti.field(dtype=int, shape=())                        # 弹簧数量
```

### 🚀 4.2 初始化模块

按照实验要求，将初始化拆分为多个 `@ti.kernel` 以保证 GPU 状态同步：

**位置初始化**（`src/test.py:29-38`）：
```python
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
```

- 创建 20×20 的网格，质点间距为 0.05
- 布料初始位置：x ∈ [-0.5, 0.5], y = 0.8, z ∈ [-0.5, 0.5]
- 固定左上角和右上角两个质点（j=0 且 i=0 或 i=N-1）

**弹簧初始化**（`src/test.py:41-53`）：
```python
@ti.kernel
def init_springs():
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        if i < N - 1:
            idx_right = (i + 1) * N + j
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_right])
            spring_lengths[c] = (x[idx] - x[idx_right]).norm()
        if j < N - 1:
            idx_down = i * N + (j + 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_down])
            spring_lengths[c] = (x[idx] - x[idx_down]).norm()
```

- 使用 `ti.atomic_add` 确保多线程环境下弹簧计数的正确性
- 为每个质点创建向右和向下的结构弹簧

### ⚙️ 4.3 力学计算模块

**受力计算**（`src/test.py:67-82`）：
```python
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
            f_spring = -k_s * (dist - spring_lengths[i]) * d_normalized
            ti.atomic_add(force[idx_a], f_spring)
            ti.atomic_add(force[idx_b], -f_spring)
```

- 使用 `ti.func` 声明，编译时强制内联，减少 GPU 函数调用开销
- 每个质点受到重力和阻尼力
- 弹簧力使用 `ti.atomic_add` 累加，避免多线程写入冲突

**速度钳制**（`src/test.py:84-88`）：
```python
@ti.func
def clamp_velocity(vel: ti.template(), idx: int):
    vel_norm = vel[idx].norm()
    if vel_norm > max_velocity:
        vel[idx] = vel[idx] / vel_norm * max_velocity
```

- 限制最大速度为 50.0，防止数值爆炸
- 在显式欧拉方法中尤为重要

### 🔄 4.4 积分求解器实现

**显式欧拉**（`src/test.py:91-97`）：
```python
@ti.kernel
def step_explicit():
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            x[i] += v[i] * dt
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)
```

- 先更新位置，再更新速度
- 使用当前时刻的状态计算下一时刻
- 不稳定，时间步长较大时容易发散

**半隐式欧拉**（`src/test.py:100-106`）：
```python
@ti.kernel
def step_semi_implicit():
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)
            x[i] += v[i] * dt
```

- 先更新速度，再用新速度更新位置
- 比显式欧拉稳定，是实际应用中最常用的方法

**隐式欧拉**（`src/test.py:109-122`）：
```python
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
    for i in range(N * N):
        v[i] = v_next[i]
        x[i] = x_next[i]
```

- 使用定点迭代法近似求解
- 迭代 3 次，每次用当前预测的未来状态重新计算力
- 最稳定，但有阻尼效果，模拟结果显得较僵硬

### 🎨 4.5 渲染与交互模块

**GGUI 控制面板**（`src/test.py:137-165`）：
```python
window.GUI.begin("Control Panel", 0.02, 0.02, 0.38, 0.36)
window.GUI.text("Integration Method:")

if window.GUI.button("[*] Explicit Euler (Explosive)"):
    current_method = 0
    init_cloth()
if window.GUI.button("[*] Semi-Implicit Euler (Stable)"):
    current_method = 1
    init_cloth()
if window.GUI.button("[*] Implicit Euler (Damped)"):
    current_method = 2
    init_cloth()

pause_label = "Resume Simulation" if paused else "Pause Simulation"
if window.GUI.button(pause_label):
    paused = not paused
if window.GUI.button("Reset Cloth"):
    init_cloth()
window.GUI.end()
```

- 实时切换三种积分方法
- 暂停/恢复模拟
- 重置布料状态

**3D 渲染**（`src/test.py:176-184`）：
```python
camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
scene.set_camera(camera)
scene.ambient_light((0.5, 0.5, 0.5))
scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

scene.particles(x, radius=0.015, color=(0.95, 0.95, 0.98))
scene.lines(x, indices=spring_indices, width=1.5, color=(0.85, 0.85, 0.9))
canvas.scene(scene)
window.show()
```

- 银白色质点和灰色弹簧线框
- 支持鼠标右键旋转视角

---

### 4.6 效果




## 💡 五、选做内容实现

### 🔀 5.1 弹簧模型完善

高级版（`src/advanced.py`）在结构弹簧基础上补充了：

**剪切弹簧**（`src/advanced.py:106-117`）：
```python
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
```

**弯曲弹簧**（`src/advanced.py:118-129`）：
```python
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
```

### ⚽ 5.2 球体碰撞检测

**碰撞处理函数**（`src/advanced.py:168-179`）：
```python
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
```

- 球体中心：$(0.0, -0.2, 0.0)$，半径：$0.35$
- 当质点进入球体内部时，将其推到球面上
- 消除法向速度分量，防止质点穿透球体

**球体渲染**（`src/advanced.py:40-67`）：
```python
def generate_sphere():
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
```

使用经纬度法生成球体网格，20×20 分辨率。

---

## 📊 六、实验结果与分析

### 🆚 6.1 积分方法对比

| 方法 | 稳定性 | 视觉效果 | 适用场景 |
|------|--------|----------|----------|
| **显式欧拉** | 差（易发散） | 数值爆炸，质点飞出屏幕 | 仅用于演示不稳定性 |
| **半隐式欧拉** | 中（稳定） | 自然流畅，摆动幅度适中 | **推荐使用** |
| **隐式欧拉** | 好（最稳定） | 沉重僵硬，摆动迅速衰减 | 需要高稳定性场景 |

### 📈 6.2 参数影响分析

**阻尼系数 $k_d$ 的影响**：
- $k_d = 1.0$：布料摆动幅度较大，衰减较慢，视觉效果自然
- $k_d = 5.0$：布料摆动迅速衰减，很快达到静止状态

**弹簧刚度的影响**：
- 结构弹簧刚度增加：布料拉伸变形减小，更接近刚体
- 剪切弹簧刚度增加：布料扭曲变形减小
- 弯曲弹簧刚度增加：布料更硬挺，不易弯曲

### 🔗 6.3 弹簧模型对比

| 模型 | 弹簧数量 | 布料形态 |
|------|----------|----------|
| 仅结构弹簧 | ~760 | 柔软，易拉伸变形 |
| +剪切弹簧 | ~1520 | 抗剪切，形态更稳定 |
| +弯曲弹簧 | ~2280 | 硬挺，保持形状能力强 |

### 💥 6.4 碰撞效果

布料与球体碰撞时，质点会被弹开，形成自然的凹陷效果。碰撞响应及时，没有明显的穿透现象。

---

## 📝 七、运行说明

### 🛠️ 7.1 环境配置

```bash
# 安装依赖
uv install

# 运行基础版（结构弹簧）
uv run python src/test.py

# 运行高级版（剪切+弯曲弹簧+球体碰撞）
uv run python src/advanced.py
```

### 🖱️ 7.2 交互操作

- **鼠标右键拖动**：旋转视角
- **控制面板**：
  - 切换积分方法（显式/半隐式/隐式欧拉）
  - 暂停/恢复模拟
  - 重置布料
  - 调节弹簧刚度（高级版）

---

## 🌟 八、总结

本次实验成功实现了基于 Taichi 的质点-弹簧布料模拟系统，主要完成以下工作：

1. 实现了完整的质点-弹簧模型，包括结构弹簧、剪切弹簧和弯曲弹簧
2. 编写了三种数值积分求解器，对比了它们的稳定性差异
3. 实现了布料与球体的空间碰撞检测和响应
4. 使用 Taichi GGUI 构建了交互式控制面板
5. 掌握了 GPU 并行编程的基本概念和优化方法

实验结果表明，半隐式欧拉方法在稳定性和效率之间取得了最佳平衡，是实际应用中的首选方法。添加剪切弹簧和弯曲弹簧可以显著改善布料的形态稳定性，使模拟结果更加真实。
