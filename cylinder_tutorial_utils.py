"""CFDBench tutorial utilities shared by student and teacher notebooks."""

from __future__ import annotations


from typing import Any, Callable, Mapping, TYPE_CHECKING


from matplotlib.axes import Axes


from matplotlib.figure import Figure


import copy


import random


import re


import time


from pathlib import Path


import numpy as np


import matplotlib.pyplot as plt


import matplotlib.tri as mtri


from matplotlib.collections import LineCollection


from matplotlib.animation import FuncAnimation


import torch


from torch import nn


from torch_geometric.data import Data


from torch_geometric.utils import to_undirected, to_torch_csr_tensor


from scipy.spatial import cKDTree


from IPython.display import display, HTML


from tqdm.auto import trange


from matplotlib.patches import Circle

if TYPE_CHECKING:
    import pyvista as pv


CFD_SPEED_LIMITS = (0.0, 8.0)


def rel_l2(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> float:
    err = (pred - target).square().sum(-1)
    ref = target.square().sum(-1)

    if weight is not None:
        err, ref = err * weight, ref * weight

    return (err.sum() / ref.sum().clamp_min(1e-12)).sqrt().item()


def train_loop(
    model: nn.Module,
    loss_fn: Callable[[nn.Module], torch.Tensor],
    steps: int,
    val_fn: Callable[[nn.Module], float] | None = None,
    lr: float = 2e-3,
    clip: float | None = 1.0,
) -> np.ndarray:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history, best_state, best_val = [], None, float("inf")
    start = time.perf_counter()

    for step in trange(steps):
        model.train()

        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model)
        loss.backward()

        if clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        if step == 0 or (step + 1) % max(1, steps // 5) == 0:
            model.eval()

            with torch.no_grad():
                val = float(val_fn(model)) if val_fn is not None else float("nan")

            history.append((step + 1, loss.item(), val))

            if val_fn is not None and val < best_val:
                best_val = val
                best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()

    print(
        f"{type(model).__name__}: {steps} steps, {time.perf_counter()-start:.1f}s, "
        f"last train loss={history[-1][1]:.4g}",
        f"best val={best_val:.4g}" if val_fn is not None else "",
    )

    return np.array(history)


def plot_history(histories: Mapping[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 2.5), layout="constrained")

    for name, h in histories.items():
        ax.semilogy(h[:, 0], h[:, 1], label=name + " train")

        if np.isfinite(h[:, 2]).any():
            ax.semilogy(h[:, 0], h[:, 2], "--", label=name + " validation")

    ax.set(xlabel="Optimizer steps", ylabel="Mean squared error")
    ax.legend(fontsize=8)
    plt.show()


class CylinderDataset:
    def __init__(
        self,
        directory: str | Path,
        pattern: str = "data*.txt",
        cylinder: tuple[float, float, float] | None = (
            0.0,
            0.0,
            0.01,
        ),  # (cx, cy, radius); adjust for your case
    ) -> None:
        # Natural ordering: ...0009, ...0010, ...0011.
        def step_number(path: str | Path) -> int:
            return int(re.findall(r"\d+", path.stem)[-1])

        self.files = sorted(
            Path(directory).glob(pattern),
            key=step_number,
        )
        if not self.files:
            raise FileNotFoundError(f"No files matching {pattern!r} in {directory}")

        self.node_ids, positions, _, _ = self._read(self.files[0])
        self.positions = positions
        self.pos = torch.tensor(positions, dtype=torch.float32)

        # Visualization triangulation; not the original CFD connectivity.
        self.triangulation = mtri.Triangulation(positions[:, 0], positions[:, 1])

        if cylinder is not None:
            cx, cy, radius = cylinder
            centers = positions[self.triangulation.triangles].mean(axis=1)

            # Approximate the circular hole using the sampled boundary.
            solid = (centers[:, 0] - cx) ** 2 + (centers[:, 1] - cy) ** 2 < radius**2
            self.triangulation.set_mask(solid)

        triangles = self.triangulation.get_masked_triangles()
        self.face = torch.tensor(triangles.T.copy(), dtype=torch.long)

        edges = torch.cat(
            [
                self.face[[0, 1]],
                self.face[[1, 2]],
                self.face[[2, 0]],
            ],
            dim=1,
        )
        self.edge_index = to_undirected(edges, num_nodes=len(positions))

        src, dst = self.edge_index
        displacement = self.pos[dst] - self.pos[src]
        self.edge_attr = torch.cat(
            [
                displacement,
                displacement.norm(dim=-1, keepdim=True),
            ],
            dim=-1,
        )

    @staticmethod
    def _read(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with open(path) as file:
            columns = file.readline().split()
            values = np.loadtxt(file, ndmin=2)

        index = {name: i for i, name in enumerate(columns)}

        def column(name: str) -> np.ndarray:
            return values[:, index[name]]

        # Align snapshots by node ID rather than assuming row order.
        order = np.argsort(column("nodenumber"))
        node_ids = column("nodenumber")[order].astype(np.int64)

        if len(np.unique(node_ids)) != len(node_ids):
            raise ValueError(f"Duplicate node IDs in {path}")

        positions = np.column_stack(
            [
                column("x-coordinate"),
                column("y-coordinate"),
            ]
        )[order]

        velocity = np.column_stack(
            [
                column("x-velocity"),
                column("y-velocity"),
            ]
        )[
            order
        ].astype(np.float32)

        pressure = column("absolute-pressure")[order].astype(np.float32)

        if not all(np.isfinite(a).all() for a in (positions, velocity, pressure)):
            raise ValueError(f"Non-finite values in {path}")

        return node_ids, positions, velocity, pressure

    def __len__(self) -> int:
        return len(self.files)

    def read_fields(self, step: int) -> tuple[np.ndarray, np.ndarray]:
        if not 0 <= step < len(self):
            raise IndexError("Frame index out of range")
        ids, positions, velocity, pressure = self._read(self.files[step])

        if not np.array_equal(ids, self.node_ids):
            raise ValueError(f"Node IDs changed at step {step}")
        if not np.allclose(positions, self.positions, atol=1e-9, rtol=1e-6):
            raise ValueError(f"Mesh coordinates changed at step {step}")

        return velocity, pressure

    def __getitem__(self, step: int) -> Data:
        if not 1 <= step < len(self):
            raise IndexError("Transition index must satisfy 1 <= step < len(dataset)")
        velocity_pre, _ = self.read_fields(step - 1)
        velocity, _ = self.read_fields(step)

        return Data(
            pos=self.pos.clone(),
            face=self.face.clone(),
            edge_index=self.edge_index.clone(),
            edge_attr=self.edge_attr.clone(),
            x=torch.from_numpy(velocity_pre),
            y=torch.from_numpy(velocity),
            filename=self.files[step].name,
        )


def plot_data(
    data: Data, field: str = "y", component: str = "speed", figsize: tuple[float, float] = (13, 4)
) -> tuple[Figure, np.ndarray]:
    """
    Plot connectivity and solution from one PyG Data object.

    field:     "x" for input state, "y" for target state.
    component: "speed", "u", or "v".
    """
    pos = data.pos.detach().cpu().numpy()
    velocity = getattr(data, field).detach().cpu().numpy()

    if component == "speed":
        values = np.linalg.norm(velocity[:, :2], axis=1)
        label = "Velocity magnitude"
    elif component in ("u", "v"):
        values = velocity[:, 0 if component == "u" else 1]
        label = f"{component}-velocity"
    else:
        raise ValueError("component must be 'speed', 'u', or 'v'.")

    # Actual graph connectivity, displaying each undirected edge once.
    edges = data.edge_index.detach().cpu().numpy().T
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    edges = edges[edges[:, 0] != edges[:, 1]]

    fig, axes = plt.subplots(1, 2, figsize=figsize, layout="constrained")

    axes[0].add_collection(
        LineCollection(
            pos[edges],
            colors="0.25",
            linewidths=0.3,
            alpha=0.7,
        )
    )
    axes[0].scatter(*pos[:, :2].T, s=1, color="black")
    axes[0].set_title(f"Connectivity · {len(pos):,} nodes, {len(edges):,} edges")

    color_limits = (
        dict(vmin=CFD_SPEED_LIMITS[0], vmax=CFD_SPEED_LIMITS[1]) if component == "speed" else {}
    )
    # Use supplied faces to preserve the obstacle hole.
    face = getattr(data, "face", None)
    if face is not None and face.numel() > 0:
        tri = mtri.Triangulation(
            pos[:, 0],
            pos[:, 1],
            triangles=face.detach().cpu().numpy().T,
        )
        artist = axes[1].tripcolor(tri, values, shading="gouraud", cmap="viridis", **color_limits)
    else:
        artist = axes[1].scatter(
            pos[:, 0],
            pos[:, 1],
            c=values,
            s=6,
            cmap="viridis",
            **color_limits,
        )

    axes[1].set_title(f"{'Input' if field == 'x' else 'Target'} · {label}")
    fig.colorbar(artist, ax=axes[1], label=label)

    for ax in axes:
        ax.set(
            aspect="equal",
            xlabel="x",
            ylabel="y",
            xlim=(pos[:, 0].min(), pos[:, 0].max()),
            ylim=(pos[:, 1].min(), pos[:, 1].max()),
        )

    plt.show()
    return fig, axes


def animate_cylinder(
    dataset: CylinderDataset,
    start_step: int = 0,
    num_steps: int = 100,
    fps: float = 15,
    field: str = "speed",  # "speed", "u", "v", or "pressure"
    show_mesh: bool = False,
) -> FuncAnimation:
    if start_step < 0 or num_steps < 1:
        raise ValueError("start_step must be >= 0 and num_steps >= 1.")

    stop_step = start_step + num_steps
    if stop_step > len(dataset):
        raise ValueError(
            f"Requested steps {start_step}–{stop_step - 1}, "
            f"but the dataset contains {len(dataset)} steps."
        )

    if field not in {"speed", "u", "v", "pressure"}:
        raise ValueError("field must be 'speed', 'u', 'v', or 'pressure'.")
    if fps <= 0:
        raise ValueError("fps must be positive.")

    steps = list(range(start_step, stop_step))

    # Load only the selected scalar fields, once.
    values = []
    for step in steps:
        velocity, pressure = dataset.read_fields(step)
        if field == "speed":
            scalar = np.linalg.norm(velocity, axis=1)
        elif field == "u":
            scalar = velocity[:, 0]
        elif field == "v":
            scalar = velocity[:, 1]
        else:
            scalar = pressure
        values.append(scalar)

    # Fixed color scale across the selected sequence.
    vmin = 0.0 if field == "speed" else min(float(a.min()) for a in values)
    vmax = CFD_SPEED_LIMITS[1] if field == "speed" else max(float(a.max()) for a in values)
    if vmax <= vmin:
        vmax = vmin + max(abs(vmin) * 1e-6, 1e-8)

    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    artist = ax.tripcolor(
        dataset.triangulation,
        values[0],
        shading="gouraud",  # Continuous interpolation of the displayed scalar
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )

    if show_mesh:
        ax.triplot(
            dataset.triangulation,
            color="black",
            linewidth=0.2,
            alpha=0.25,
        )

    ax.set(aspect="equal", xlabel="x", ylabel="y")
    title = ax.set_title("")
    fig.colorbar(artist, ax=ax, label=field)

    def update(k: int) -> list[Any]:
        step = steps[k]
        artist.set_array(values[k])
        title.set_text(f"{field} · step {step} · {dataset.files[step].name}")
        return artist, title

    update(0)
    movie = FuncAnimation(
        fig,
        update,
        frames=len(steps),
        interval=1000 / fps,
        blit=False,
    )
    plt.close(fig)

    with plt.rc_context({"animation.embed_limit": 100}):
        display(HTML(movie.to_jshtml()))

    return movie


def face_edges(face: torch.Tensor, n_nodes: int) -> torch.Tensor:
    edges = torch.cat([face[[0, 1]], face[[1, 2]], face[[2, 0]]], dim=1)
    return to_undirected(edges, num_nodes=n_nodes)


def radius_edges(pos: torch.Tensor, radius: float) -> torch.Tensor:
    if radius <= 0:
        raise ValueError("Radius must be positive")
    pairs = cKDTree(pos.cpu().numpy()).query_pairs(radius, output_type="ndarray")
    directed = np.concatenate([pairs, pairs[:, ::-1]], axis=0)
    return torch.tensor(directed.T.copy(), dtype=torch.long)


def merge_large_elements(
    points: np.ndarray, faces: np.ndarray, max_triangles: int = 8
) -> tuple[pv.PolyData, np.ndarray, list[list[int]]]:
    """Agglomerate triangle cells without removing or stranding any vertex."""
    import pyvista as pv
    from collections import defaultdict

    p = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if max_triangles < 2:
        raise ValueError("max_triangles must be >= 2")

    def edge_set(poly: list[int]) -> set[tuple[int, int]]:
        return {tuple(sorted((int(a), int(b)))) for a, b in zip(poly, np.roll(poly, -1))}

    boundaries = {i: edge_set(t) for i, t in enumerate(faces)}
    vertices = {i: set(map(int, t)) for i, t in enumerate(faces)}
    sizes = {i: 1 for i in range(len(faces))}
    original_edges = set().union(*boundaries.values())

    def ordered_loop(edges: set[tuple[int, int]], required_vertices: set[int]) -> list[int] | None:
        neighbors = defaultdict(list)
        for a, b in edges:
            neighbors[a].append(b)
            neighbors[b].append(a)
        # Reject unions enclosing an original vertex or forming holes/pinches.
        if set(neighbors) != required_vertices or any(len(v) != 2 for v in neighbors.values()):
            return None
        start = min(neighbors)
        loop = [start]
        previous = None
        current = start
        while True:
            nxt = next(v for v in neighbors[current] if v != previous)
            if nxt == start:
                break
            if nxt in loop:
                return None
            loop.append(nxt)
            previous, current = current, nxt
        if len(loop) != len(neighbors):
            return None
        xy = p[loop]
        area = np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - xy[:, 1] * np.roll(xy[:, 0], -1)) / 2
        if area == 0:
            return None
        return loop if area > 0 else loop[::-1]

    while True:
        owners = defaultdict(list)
        for i, edges in boundaries.items():
            for edge in edges:
                owners[edge].append(i)
        pairs = {tuple(sorted(ids)) for ids in owners.values() if len(ids) == 2}
        candidates = []
        for a, b in pairs:
            if sizes[a] + sizes[b] > max_triangles:
                continue
            edges = boundaries[a] ^ boundaries[b]
            verts = vertices[a] | vertices[b]
            loop = ordered_loop(edges, verts)
            if loop is None:
                continue
            xy = p[loop]
            delta = np.roll(xy, -1, axis=0) - xy
            area = (
                abs(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - xy[:, 1] * np.roll(xy[:, 0], -1))) / 2
            )
            compactness = np.linalg.norm(delta, axis=1).sum() ** 2 / area
            candidates.append((compactness, a, b, edges, verts))
        if not candidates:
            break
        used = set()
        for _, a, b, edges, verts in sorted(candidates, key=lambda x: (x[0], x[1], x[2])):
            if a in used or b in used:
                continue
            used.update((a, b))
            boundaries[a] = edges
            vertices[a] = verts
            sizes[a] += sizes[b]
            del boundaries[b], vertices[b], sizes[b]
    polygons = [ordered_loop(edges, vertices[i]) for i, edges in boundaries.items()]
    cells = np.concatenate([np.asarray([len(poly), *poly], dtype=np.int64) for poly in polygons])
    original_area = (
        np.abs(
            (p[faces[:, 1], 0] - p[faces[:, 0], 0]) * (p[faces[:, 2], 1] - p[faces[:, 0], 1])
            - (p[faces[:, 1], 1] - p[faces[:, 0], 1]) * (p[faces[:, 2], 0] - p[faces[:, 0], 0])
        ).sum()
        / 2
    )
    polygon_area = sum(
        abs(np.sum(p[q, 0] * np.roll(p[q, 1], -1) - p[q, 1] * np.roll(p[q, 0], -1))) / 2
        for q in polygons
    )
    assert np.isclose(original_area, polygon_area, rtol=1e-8), "Domain area changed"
    mesh = pv.PolyData(np.column_stack([p, np.zeros(len(p))]), cells)
    edges = np.asarray(sorted(set().union(*boundaries.values())), dtype=np.int64)
    assert np.array_equal(np.unique(edges), np.unique(faces)), "Active vertex set changed"
    assert sum(sizes.values()) == len(faces)
    print(
        f"{len(faces):,} triangles → {len(polygons):,} polygon elements; "
        f"mean {len(faces)/len(polygons):.2f}, max {max(sizes.values())} triangles per element"
    )
    print(
        f"Removed {len(original_edges)-len(edges):,} internal edges; all {len(p):,} vertices retained."
    )
    return mesh, edges, polygons


def random_edge_collapse(
    points: np.ndarray, faces: np.ndarray, percent: float = 10.0, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, pv.PolyData]:
    """Attempt floor(percent/100 * initial undirected edges) valid contractions.

    Randomize original edge candidates and endpoint direction. Keep one endpoint
    at its original coordinate, redirect the other endpoint's incident triangles,
    and delete the two triangles on the collapsed interior edge.
    """
    import pyvista as pv

    p = np.asarray(points, dtype=np.float64)
    t = np.asarray(faces, dtype=np.int64).copy()
    if not 0 <= percent <= 100:
        raise ValueError("percent must be in [0,100]")

    def edge_counts(tri: np.ndarray) -> dict[tuple[int, int], int]:
        e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
        return np.unique(e, axis=0, return_counts=True)

    original_edges, counts = edge_counts(t)
    original_boundary = original_edges[counts == 1]
    fixed = set(original_boundary.ravel())
    incident = [set() for _ in range(len(p))]
    for i, tri in enumerate(t):
        for v in tri:
            incident[v].add(i)
    live = np.ones(len(t), dtype=bool)
    removed = np.zeros(len(p), dtype=bool)

    def neighbors(v: int) -> set[int]:
        return {int(w) for f in incident[v] for w in t[f] if w != v}

    def signed_area2(tri: np.ndarray) -> np.ndarray:
        a = p[tri[:, 1]] - p[tri[:, 0]]
        b = p[tri[:, 2]] - p[tri[:, 0]]
        return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]

    rng = np.random.default_rng(seed)
    requested = int(np.floor(percent / 100 * len(original_edges)))
    done = 0
    for j in rng.permutation(len(original_edges)):
        if done >= requested:
            break
        a, b = map(int, original_edges[j])
        if removed[a] or removed[b] or a in fixed or b in fixed:
            continue
        shared = incident[a] & incident[b]
        if len(shared) != 2:
            continue
        opposite = {int(v) for f in shared for v in t[f] if v not in (a, b)}
        if neighbors(a) & neighbors(b) != opposite:
            continue  # manifold link condition
        if rng.integers(2):
            a, b = b, a
        # Try both endpoint directions, rejecting inverted or degenerate triangles.
        accepted = False
        for survivor, victim in [(a, b), (b, a)]:
            changed = sorted(incident[victim] - shared)
            old = t[changed].copy()
            new = old.copy()
            new[new == victim] = survivor
            before = signed_area2(old)
            after = signed_area2(new)
            if np.any(before * after <= 0) or np.any(np.abs(after) < 1e-8 * np.abs(before)):
                continue
            for f in list(incident[victim]):
                for v in t[f]:
                    incident[v].discard(f)
                if f in shared:
                    live[f] = False
                else:
                    t[f][t[f] == victim] = survivor
                    for v in t[f]:
                        incident[v].add(f)
            removed[victim] = True
            done += 1
            accepted = True
            break
    keep = np.flatnonzero(~removed)
    inverse = np.full(len(p), -1, dtype=np.int64)
    inverse[keep] = np.arange(len(keep))
    triangles = inverse[t[live]]
    new_edges, new_counts = edge_counts(t[live])
    assert np.array_equal(original_boundary, new_edges[new_counts == 1]), "Boundary changed"
    assert len(np.unique(np.sort(triangles, axis=1), axis=0)) == len(triangles)
    original_area = np.abs(signed_area2(np.asarray(faces))).sum()
    assert np.isclose(original_area, np.abs(signed_area2(t[live])).sum(), rtol=1e-8), "Area changed"
    assert len(keep) == len(p) - done
    mesh = pv.PolyData(
        np.column_stack([p[keep], np.zeros(len(keep))]),
        np.column_stack([np.full(len(triangles), 3), triangles]).ravel(),
    )
    print(
        f"Successful edge collapses: {done:,}/{requested:,} requested "
        f"({100*done/len(original_edges):.1f}% of initial edges)"
    )
    print(f"Vertices: {len(p):,} → {len(keep):,}; triangles: {len(faces):,} → {len(triangles):,}")
    if done < requested:
        print("Stopped after exhausting original edge candidates compatible with the constraints.")
    return keep, triangles, mesh


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def on_device(graph: Data, device: torch.device) -> Data:
    g = graph.clone().to(device)
    g.adj_t = to_torch_csr_tensor(g.edge_index.flip(0), size=g.num_nodes)
    return g


def graph_variants(
    sample: Data,
    radius: float,
    max_triangles: int,
    collapse_percent: float,
    seed: int,
) -> tuple[dict[str, tuple[dict[str, Data], torch.Tensor, torch.Tensor]], dict[str, pv.PolyData]]:
    merged_mesh, edges, polygons = merge_large_elements(
        sample.pos.numpy(), sample.face.T.numpy(), max_triangles
    )
    merged_edges = torch.tensor(np.concatenate([edges, edges[:, ::-1]]).T.copy(), dtype=torch.long)
    keep, triangles, collapsed_mesh = random_edge_collapse(
        sample.pos.numpy(), sample.face.T.numpy(), collapse_percent, seed
    )
    keep = torch.as_tensor(keep, dtype=torch.long)
    face = torch.tensor(triangles.T.copy(), dtype=torch.long)

    def pair(ids: torch.Tensor, edges: torch.Tensor, face: torch.Tensor) -> dict[str, Data]:
        p, x, y = sample.pos[ids], sample.x[ids], sample.y[ids]
        return {
            "mesh": Data(
                pos=p.clone(), x=x.clone(), y=y.clone(), edge_index=edges.clone(), face=face.clone()
            ),
            "pointcloud": Data(
                pos=p.clone(),
                x=x.clone(),
                y=y.clone(),
                edge_index=radius_edges(p.clone(), radius),
                face=face.clone(),
            ),
        }

    full = torch.arange(sample.num_nodes)
    experiments = {
        "Larger elements": (pair(full, merged_edges, sample.face), full, sample.face),
        "Edge collapse": (pair(keep, face_edges(face, len(keep)), face), keep, face),
    }
    return experiments, {"Larger elements": merged_mesh, "Edge collapse": collapsed_mesh}


@torch.no_grad()
def predict_graph(model: nn.Module, graph: Data, device: torch.device) -> torch.Tensor:
    model.eval()
    return model(on_device(graph, device)).cpu()


@torch.no_grad()
def query_field(
    model: nn.Module,
    pos: torch.Tensor,
    device: torch.device,
    chunk: int = 4096,
) -> torch.Tensor:
    model.eval()
    return torch.cat([model(p.to(device)).cpu() for p in pos.split(chunk)])


def _tri(pos: torch.Tensor, face: torch.Tensor | None) -> mtri.Triangulation:
    pos = pos.detach().cpu().numpy()
    return mtri.Triangulation(pos[:, 0], pos[:, 1], face.detach().cpu().numpy().T)


def _style(ax: Axes) -> None:
    ax.set_aspect("equal")
    ax.set_axis_off()


def plot_predictions(sample: Data, predictions: Mapping[str, torch.Tensor | np.ndarray]) -> Figure:
    names = list(predictions)
    fig, axes = plt.subplots(
        2, len(names) + 1, figsize=(5 * (len(names) + 1), 6), squeeze=False, layout="constrained"
    )
    tri = _tri(sample.pos, sample.face)
    fields = [sample.y, *predictions.values()]
    vmax = CFD_SPEED_LIMITS[1]
    errors = [(v - sample.y).norm(dim=-1) for v in predictions.values()]
    emax = max([float(e.max()) for e in errors] + [1e-12])
    for j, (name, v) in enumerate(zip(["CFD", *names], fields)):
        a = axes[0, j].tripcolor(
            tri,
            v.norm(dim=-1).cpu(),
            shading="gouraud",
            vmin=0,
            vmax=max(vmax, 1e-12),
            cmap="viridis",
        )
        axes[0, j].set_title(name + ("" if j == 0 else f" · L2 {rel_l2(v,sample.y):.3f}"))
        if j:
            b = axes[1, j].tripcolor(
                tri, errors[j - 1].cpu(), shading="gouraud", vmin=0, vmax=emax, cmap="magma"
            )
            axes[1, j].set_title("Error against CFD")
    axes[1, 0].set_axis_off()
    for ax in axes.ravel():
        _style(ax)
    fig.colorbar(a, ax=axes[0].tolist(), label="Velocity magnitude", shrink=0.8)
    fig.colorbar(b, ax=axes[1, 1:].tolist(), label="Vector error magnitude", shrink=0.8)
    plt.show()
    return fig


def plot_connectivities(
    original: Mapping[str, Data],
    experiments: Mapping[str, tuple[dict[str, Data], torch.Tensor, torch.Tensor]],
    center: tuple[float, float] = (0.03, 0),
    half_width: float = 0.025,
) -> Figure:
    groups = {"Original": original, **{label: g for label, (g, _, _) in experiments.items()}}
    fig, axes = plt.subplots(
        2,
        len(groups),
        figsize=(5 * len(groups), 8),
        squeeze=False,
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    lo = np.asarray(center) - half_width
    hi = np.asarray(center) + half_width
    for col, (label, graphs) in enumerate(groups.items()):
        for row, name in enumerate(["mesh", "pointcloud"]):
            g = graphs[name]
            xy = g.pos.cpu().numpy()
            edges = np.unique(np.sort(g.edge_index.cpu().numpy().T, axis=1), axis=0)
            seg = xy[edges]
            mask = np.all(seg.max(1) >= lo, 1) & np.all(seg.min(1) <= hi, 1)
            points = np.all((xy >= lo) & (xy <= hi), 1)
            ax = axes[row, col]
            ax.add_collection(
                LineCollection(seg[mask], linewidths=0.4, colors="#2563eb", alpha=0.6)
            )
            ax.scatter(*xy[points].T, s=3, c="black")
            ax.set(
                title=f"{label} · {name}\n{len(xy):,} vertices",
                aspect="equal",
                xlim=(lo[0], hi[0]),
                ylim=(lo[1], hi[1]),
            )
    plt.show()
    return fig


def plot_drift(
    sample: Data,
    experiments: Mapping[str, tuple[dict[str, Data], torch.Tensor, torch.Tensor]],
    baseline: Mapping[str, torch.Tensor],
    changed: Mapping[str, Mapping[str, torch.Tensor]],
    quantile: float = 0.75,
) -> Figure:
    """Same CFD/drift layout for any number of networks, including a neural field."""
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0,1]")
    names = list(baseline)
    fig, axes = plt.subplots(
        len(experiments),
        1 + len(names),
        figsize=(5 * (1 + len(names)), 3.5 * len(experiments)),
        squeeze=False,
        layout="constrained",
    )
    vmax = CFD_SPEED_LIMITS[1]
    for row, (label, (graphs, ids, face)) in enumerate(experiments.items()):
        target = sample.y[ids]
        tri = _tri(graphs["mesh"].pos, face)
        drifts = {name: (changed[label][name] - baseline[name][ids]).norm(dim=-1) for name in names}
        limit = max(
            float(torch.quantile(torch.cat(list(drifts.values())).float(), quantile)), 1e-12
        )
        a = axes[row, 0].tripcolor(
            tri, target.norm(dim=-1), cmap="viridis", shading="gouraud", vmin=0, vmax=vmax
        )
        axes[row, 0].set_title(f"{label} · CFD")
        for col, name in enumerate(names, 1):
            v = drifts[name]
            relative = float(v.norm() / target.norm().clamp_min(1e-12))
            b = axes[row, col].tripcolor(
                tri, v, cmap="magma", shading="gouraud", vmin=0, vmax=limit
            )
            axes[row, col].set_title(f"{name} · drift {relative:.3g}")
        for ax in axes[row]:
            _style(ax)
        fig.colorbar(a, ax=axes[row, 0], label="CFD speed", shrink=0.8)
        fig.colorbar(
            b, ax=axes[row, 1:].tolist(), label="Prediction drift", extend="max", shrink=0.8
        )
    plt.show()
    return fig


def load_sequence(
    dataset: CylinderDataset,
    start: int,
    num_frames: int,
    n_points: int = 1024,
    seed: int = 42,
    stride: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if stride < 1 or num_frames < 2:
        raise ValueError("stride >= 1 and num_frames >= 2 required")
    if start < 0 or start + (num_frames - 1) * stride >= len(dataset):
        raise ValueError(
            "Requested sequence exceeds available frames; reduce start, stride, or horizon"
        )
    ids = np.sort(
        np.random.default_rng(seed).choice(
            dataset.pos.shape[0], min(n_points, len(dataset.pos)), replace=False
        )
    )
    values = np.stack(
        [dataset.read_fields(i)[0][ids] for i in range(start, start + num_frames * stride, stride)]
    )
    return dataset.pos[torch.as_tensor(ids)].clone(), torch.from_numpy(values)


def rollout_metrics(
    reference: torch.Tensor | np.ndarray,
    reseeded: torch.Tensor | np.ndarray,
    autonomous: torch.Tensor | np.ndarray,
    initial: torch.Tensor | np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Single metric implementation shared by plots and animation."""

    def array(x: torch.Tensor | np.ndarray) -> np.ndarray:
        return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

    ref, reset, pure = [array(x).astype(np.float64) for x in (reference, reseeded, autonomous)]
    if ref.shape != reset.shape or ref.shape != pure.shape or ref.ndim != 3:
        raise ValueError("Rollout arrays must have identical [time, vertex, channel] shapes")
    denominator = np.maximum(np.sum(ref**2, axis=(1, 2)), 1e-12)

    def error(pred: np.ndarray) -> np.ndarray:
        return np.sqrt(np.sum((pred - ref) ** 2, axis=(1, 2)) / denominator)

    curves = {"GT reseeded each step": error(reset), "Pure neural rollout": error(pure)}
    if initial is not None:
        curves["Frozen initial state"] = error(array(initial))
    return curves


def _rollout_frame_ids(frame_ids: np.ndarray | list[int] | None, horizon: int) -> np.ndarray:
    frames = np.arange(1, horizon + 1) if frame_ids is None else np.asarray(frame_ids)
    if frames.shape != (horizon,) or not np.all(np.isfinite(frames)):
        raise ValueError("Exactly one finite target frame ID per rollout step is required")
    if not np.all(frames == np.floor(frames)) or np.any(np.diff(frames) <= 0):
        raise ValueError("Target frame IDs must be strictly increasing integers")
    return frames.astype(np.int64)


def plot_rollout_errors(
    reference: torch.Tensor | np.ndarray,
    reseeded: torch.Tensor | np.ndarray,
    autonomous: torch.Tensor | np.ndarray,
    initial: torch.Tensor | np.ndarray | None,
    frame_ids: np.ndarray | list[int] | None = None,
    title: str | None = None,
) -> dict[str, np.ndarray]:
    horizon = len(reference)
    frames = _rollout_frame_ids(frame_ids, horizon)
    curves = rollout_metrics(reference, reseeded, autonomous, initial)
    fig, ax = plt.subplots(figsize=(8, 3.5), layout="constrained")
    for name, values in curves.items():
        ax.plot(np.arange(1, horizon + 1), values, label=name, marker=".")
    heading = title or "Rollout errors"
    ax.set(
        title=f"{heading} · target CFD frames {frames[0]}–{frames[-1]}",
        xlabel="Steps from initial CFD frame",
        ylabel="Relative L2 velocity error",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    ax.set_yscale("log")
    plt.show()
    return curves


def animate_rollouts(
    pos: torch.Tensor,
    reference: torch.Tensor | np.ndarray,
    reseeded: torch.Tensor | np.ndarray,
    autonomous: torch.Tensor | np.ndarray,
    frame_ids: np.ndarray | list[int] | None = None,
    cylinder: tuple[float, float, float] | None = None,
    fps: float = 12,
    component: str = "speed",
    every: int = 1,
    show_errors: bool = True,
    face: torch.Tensor | None = None,
    expected_curves: Mapping[str, np.ndarray] | None = None,
) -> FuncAnimation:
    """Synchronized field/error animation; arrays [time, vertices, 2] in physical units.

    When face is provided, use that exact mesh connectivity and its existing hole.
    Otherwise build a display triangulation with an optional cylinder mask.
    """

    def array(x: torch.Tensor | np.ndarray) -> np.ndarray:
        return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

    xy = array(pos)
    ref, reset, pure = map(array, (reference, reseeded, autonomous))
    if (
        ref.shape != reset.shape
        or ref.shape != pure.shape
        or ref.ndim != 3
        or ref.shape[1:] != (len(xy), 2)
    ):
        raise ValueError("Expected matching [time, vertices, 2] arrays")
    if fps <= 0 or every < 1:
        raise ValueError("fps and every must be positive")
    if not all(np.isfinite(v).all() for v in [ref, reset, pure]):
        raise ValueError("Rollout contains nonfinite values: inspect stability before rendering")
    if face is not None:
        triangles = array(face)
        if triangles.ndim != 2:
            raise ValueError("face must be [3, triangles] or [triangles, 3]")
        if triangles.shape[0] == 3:
            triangles = triangles.T
        if triangles.shape[1] != 3:
            raise ValueError("Expected triangular faces")
        tri = mtri.Triangulation(xy[:, 0], xy[:, 1], triangles=triangles.astype(np.int64))
    else:
        tri = mtri.Triangulation(xy[:, 0], xy[:, 1])
        if cylinder is not None:
            center = np.asarray(cylinder[:2])
            radius = cylinder[2]
            verts = xy[tri.triangles]
            mask = np.zeros(len(verts), dtype=bool)
            # Distance from center to triangle: edge distances plus point-in-triangle.
            crosses = []
            for j in range(3):
                a = verts[:, j]
                b = verts[:, (j + 1) % 3]
                edge = b - a
                t = np.clip(
                    np.sum((center - a) * edge, axis=1)
                    / np.maximum(np.sum(edge * edge, axis=1), 1e-30),
                    0,
                    1,
                )
                closest = a + t[:, None] * edge
                mask |= np.sum((closest - center) ** 2, axis=1) < (radius * (1 - 1e-6)) ** 2
                crosses.append(
                    edge[:, 0] * (center[1] - a[:, 1]) - edge[:, 1] * (center[0] - a[:, 0])
                )
            crosses = np.stack(crosses, axis=1)
            mask |= np.all(crosses >= 0, axis=1) | np.all(crosses <= 0, axis=1)
            tri.set_mask(mask)
    if component not in ("speed", "u", "v"):
        raise ValueError("component must be speed, u, or v")
    scalar = lambda v: (
        np.linalg.norm(v, axis=-1) if component == "speed" else v[..., 0 if component == "u" else 1]
    )
    fields = [scalar(v) for v in [ref, reset, pure]]
    errors = [np.linalg.norm(v - ref, axis=-1) for v in [reset, pure]]
    vmin = 0 if component == "speed" else min(float(v.min()) for v in fields)
    vmax = (
        CFD_SPEED_LIMITS[1]
        if component == "speed"
        else max(max(float(v.max()) for v in fields), vmin + 1e-12)
    )
    emax = max(max(float(v.max()) for v in errors), 1e-12)
    frames = _rollout_frame_ids(frame_ids, len(ref))
    curves = rollout_metrics(ref, reset, pure)
    if expected_curves is not None:
        for name, values in curves.items():
            if name not in expected_curves or not np.allclose(
                values, expected_curves[name], rtol=1e-10, atol=1e-12
            ):
                raise ValueError(f"Animation errors differ from plot values: {name}")
    fig, axes = plt.subplots(
        2 if show_errors else 1,
        3,
        figsize=(14, 7 if show_errors else 3.7),
        squeeze=False,
        layout="constrained",
    )
    artists = []
    for j, (v, title) in enumerate(
        zip(fields, ["CFD", "GT reseeded each step", "Pure neural rollout"])
    ):
        artists.append(
            axes[0, j].tripcolor(tri, v[0], shading="gouraud", cmap="viridis", vmin=vmin, vmax=vmax)
        )
        axes[0, j].set_title(title)
    fig.colorbar(artists[0], ax=axes[0].tolist(), label=component, shrink=0.8)
    error_artists = []
    if show_errors:
        axes[1, 0].set_axis_off()
        for j, v in enumerate(errors, 1):
            error_artists.append(
                axes[1, j].tripcolor(tri, v[0], shading="gouraud", cmap="magma", vmin=0, vmax=emax)
            )
        fig.colorbar(
            error_artists[0], ax=axes[1, 1:].tolist(), label="Vector error against CFD", shrink=0.8
        )
    for ax in axes.ravel():
        _style(ax)
    title = fig.suptitle("")

    def update(k: int) -> list[Any]:
        for artist, v in zip(artists, fields):
            artist.set_array(v[k])
        for j, (artist, v, pred) in enumerate(zip(error_artists, errors, [reset, pure]), 1):
            artist.set_array(v[k])
            relative = curves[["GT reseeded each step", "Pure neural rollout"][j - 1]][k]
            axes[1, j].set_title(f"Relative L2: {relative:.3f}")
        title.set_text(f"CFD frame {frames[k]} · rollout step {k+1}")
        return artists + error_artists + [title]

    update(0)
    animation = FuncAnimation(
        fig, update, frames=list(range(0, len(ref), every)), interval=1000 / fps, blit=False
    )
    animation.target_frame_ids = frames.copy()
    animation.displayed_rollout_steps = np.arange(1, len(ref) + 1, every)
    animation.error_curves = curves
    plt.close(fig)
    with plt.rc_context({"animation.embed_limit": 150}):
        display(HTML(animation.to_jshtml()))
    return animation


from contextlib import nullcontext


import torch.nn.functional as F

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:
    # PyTorch 2.0–2.2 exposes SDPA controls through torch.backends.cuda.
    SDPBackend = sdpa_kernel = None


ATTENTION_BACKENDS_SEEN = set()


def flash_sdpa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, key_padding_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Q/K/V: [B,H,N,D]; modern or legacy SDPA controls (PyTorch >=2.0)."""
    if not hasattr(F, "scaled_dot_product_attention"):
        raise RuntimeError("Attention exercises require PyTorch >=2.0.")
    original_dtype = q.dtype
    if q.is_cuda:
        q, k, v = (t.to(torch.float16) for t in (q, k, v))
    allowed = None
    if key_padding_mask is not None and bool(key_padding_mask.any()):
        if bool(key_padding_mask.all(dim=-1).any()):
            raise ValueError("Each case needs at least one valid key.")
        allowed = (~key_padding_mask)[:, None, None, :]
    if sdpa_kernel is None:
        # Older versions lack both nn.attention and modern capability probes.
        # Enable supported kernels and let SDPA select; do not claim a specific backend.
        legacy_kernel = getattr(torch.backends.cuda, "sdp_kernel", None)
        context = (
            legacy_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True)
            if q.is_cuda and legacy_kernel is not None
            else nullcontext()
        )
        label = (
            "legacy SDPA automatic selection (Flash/efficient/math)"
            if q.is_cuda
            else f"SDPA fallback ({q.device.type}; legacy API)"
        )
    else:
        try:
            params = torch.backends.cuda.SDPAParams(q, k, v, allowed, 0.0, False, False)
        except TypeError:
            # Some releases predate the enable_gqa argument.
            params = torch.backends.cuda.SDPAParams(q, k, v, allowed, 0.0, False)
        if q.is_cuda and torch.backends.cuda.can_use_flash_attention(params):
            backend, label = SDPBackend.FLASH_ATTENTION, "FlashAttention (CUDA float16)"
        elif q.is_cuda and torch.backends.cuda.can_use_efficient_attention(params):
            backend, label = SDPBackend.EFFICIENT_ATTENTION, "efficient SDPA fallback (CUDA)"
        else:
            backend, label = SDPBackend.MATH, f"math SDPA fallback ({q.device.type})"
        context = sdpa_kernel(backend)
    if label not in ATTENTION_BACKENDS_SEEN:
        print("Attention backend:", label)
        ATTENTION_BACKENDS_SEEN.add(label)
    with context:
        answer = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed, dropout_p=0.0, is_causal=False
        )
    return answer.to(original_dtype)


def _array(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def display_records(records: list[dict[str, Any]]) -> None:
    """Small comparison table, without a pandas dependency."""
    import html

    keys = list(records[0]) if records else []

    def format_value(value: Any) -> str:
        if isinstance(value, float):
            return "N/A" if not np.isfinite(value) else f"{value:.5g}"
        return str(value)

    header = "".join(f"<th>{html.escape(key)}</th>" for key in keys)
    rows = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(format_value(row[key]))}</td>" for key in keys)
        + "</tr>"
        for row in records
    )
    display(HTML(f"<table><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table>"))


def plot_anchor_locations(
    sample: Data, source_pos: torch.Tensor, anchor_pos: torch.Tensor, radius: float | None = None
) -> Figure:
    """Show conditioning anchors; optional circles are for radius-pooling examples."""
    from matplotlib.patches import Circle

    source, anchors = _array(source_pos), _array(anchor_pos)
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    ax.triplot(_tri(sample.pos, sample.face), color="#bfc7d0", linewidth=0.25)
    ax.scatter(source[:, 0], source[:, 1], s=3, c="#607d8b", alpha=0.5, label="All points")
    if radius is not None:
        for point in anchors:
            ax.add_patch(
                Circle(point, radius, fill=False, color="#c62828", linewidth=0.7, alpha=0.35)
            )
    ax.scatter(
        anchors[:, 0],
        anchors[:, 1],
        s=38,
        c="#c62828",
        edgecolor="white",
        linewidth=0.5,
        label=f"{len(anchors)} selected anchors",
        zorder=3,
    )
    ax.set(
        aspect="equal",
        xlabel="x",
        ylabel="y",
    )
    ax.legend(loc="upper right")
    plt.show()
    return fig


def plot_graph_connectivities(
    graphs: Mapping[str, Data],
    radius: float,
    node_id: int | None = None,
    half_width: float | None = None,
) -> Figure:
    pos = graphs["mesh"].pos.detach().cpu().numpy()[:, :2]

    if node_id is None:
        node_id = np.argmin(((pos - pos.mean(axis=0)) ** 2).sum(axis=1))

    center = pos[node_id]
    edge_sets = {
        name: np.unique(
            np.sort(graph.edge_index.detach().cpu().numpy().T, axis=1),
            axis=0,
        )
        for name, graph in graphs.items()
    }

    # Include the mesh neighbors even when the radius is very small.
    if half_width is None:
        edges = edge_sets["mesh"]
        incident = edges[(edges == node_id).any(axis=1)]
        neighbors = np.unique(incident)
        extent = np.abs(pos[neighbors] - center).max() if len(neighbors) else radius
        half_width = max(3 * radius, 1.2 * extent)

    lo, hi = center - half_width, center + half_width
    visible_nodes = ((pos >= lo) & (pos <= hi)).all(axis=1)

    fig, axes = plt.subplots(
        1,
        len(graphs),
        figsize=(12, 5),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )

    for ax, (name, edges) in zip(axes[0], edge_sets.items()):
        # Opaque disk behind the connectivity.
        ax.add_patch(
            Circle(
                center,
                radius,
                facecolor="#fff0b3",
                edgecolor="#d69e00",
                alpha=1,
                linewidth=1.5,
                zorder=0,
            )
        )

        segments = pos[edges]
        visible_edges = (segments.max(axis=1) >= lo).all(axis=1) & (segments.min(axis=1) <= hi).all(
            axis=1
        )
        ax.add_collection(
            LineCollection(
                segments[visible_edges],
                colors="#94a3b8",
                linewidths=0.7,
                zorder=1,
            )
        )

        incident = (edges == node_id).any(axis=1)
        ax.add_collection(
            LineCollection(
                segments[incident],
                colors="#2563eb",
                linewidths=2,
                zorder=2,
            )
        )

        ax.scatter(
            *pos[visible_nodes].T,
            s=14,
            color="#334155",
            zorder=3,
        )
        ax.scatter(
            *center,
            s=65,
            color="#dc2626",
            edgecolor="white",
            zorder=4,
        )

        ax.set(
            title=f"{name} · selected vertex has {incident.sum()} neighbors",
            xlim=(lo[0], hi[0]),
            ylim=(lo[1], hi[1]),
            xlabel="x",
            ylabel="y",
            aspect="equal",
        )

    fig.suptitle(f"Vertex {node_id} · yellow disk: radius {radius:g}")
    plt.show()
    return fig


@torch.no_grad()
def query_anchor_field(
    model: nn.Module,
    anchor_pos: torch.Tensor,
    anchor_state: torch.Tensor,
    query_pos: torch.Tensor,
    query_state: torch.Tensor,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Append queries to fixed anchors; return [B,Q,2] query predictions.

    Chunking is equivalent only for an anchor-conditioned model. Passing chunks
    to a full-self-attention model is an intentional query-dependence diagnostic.
    Current-state inputs at every coordinate must stay fixed across comparisons.
    """
    if anchor_state.ndim == 2:
        anchor_state = anchor_state.unsqueeze(0)
    if query_state.ndim == 2:
        query_state = query_state.unsqueeze(0)
    if len(anchor_pos) == 0 or anchor_state.shape[1] != len(anchor_pos):
        raise ValueError("Need nonempty anchors with matching current states")
    if query_state.shape[:2] != (anchor_state.shape[0], len(query_pos)):
        raise ValueError("Query state must have shape [B,Q,channels]")
    if chunk_size is not None and chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if len(query_pos) == 0:
        return query_state.clone()
    m = len(anchor_pos)
    local_idx = torch.arange(m, device=anchor_pos.device)
    chunk = len(query_pos) if chunk_size is None else chunk_size
    outputs = []
    for start in range(0, len(query_pos), chunk):
        pos = torch.cat([anchor_pos, query_pos[start : start + chunk]])
        state = torch.cat([anchor_state, query_state[:, start : start + chunk]], dim=1)
        outputs.append(model(pos, state, local_idx)[:, m:])
    return torch.cat(outputs, dim=1)


@torch.no_grad()
def predict_anchor_snapshot(
    model: nn.Module,
    pos: torch.Tensor,
    state: torch.Tensor,
    anchor_idx: torch.Tensor,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Original vertex order, no duplicated anchors; return [B,N,2].

    Chunking preserves estimates for anchor attention only. Query inputs and
    conditioning stay fixed. Full-self calls must decode all tokens together.
    """
    if state.ndim == 2:
        state = state.unsqueeze(0)
    if chunk_size is None:
        return model(pos, state, anchor_idx)
    if chunk_size < 1 or len(anchor_idx) == 0:
        raise ValueError("Need a positive chunk size and nonempty anchors")
    if len(anchor_idx.unique()) != len(anchor_idx):
        raise ValueError("Anchor indices must be unique")
    mask = torch.ones(len(pos), dtype=torch.bool, device=pos.device)
    mask[anchor_idx] = False
    queries = torch.arange(len(pos), device=pos.device)[mask]
    result = torch.empty_like(state)
    anchor_pos, anchor_state = pos[anchor_idx], state[:, anchor_idx]
    result[:, anchor_idx] = model(
        anchor_pos, anchor_state, torch.arange(len(anchor_idx), device=pos.device)
    )
    result[:, queries] = query_anchor_field(
        model, anchor_pos, anchor_state, pos[queries], state[:, queries], chunk_size
    )
    return result


def train_profiled_epochs(
    model: nn.Module,
    loss_fn: Callable[[nn.Module, int], torch.Tensor],
    transitions: list[int],
    epochs: int,
    lr: float,
    label: str,
) -> list[dict[str, Any]]:
    """Profile complete epochs; CUDA memory is measured, CPU memory is unavailable."""
    device = next(model.parameters()).device
    gpu = device.type == "cuda"
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    records = []
    for epoch in trange(1, epochs + 1, desc=label):
        model.train()
        model.zero_grad(set_to_none=True)
        if gpu:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        losses = []
        status = "ok"
        try:
            for t in transitions:
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model, t)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(loss.detach())
            # One synchronization at epoch end; do not time plotting or validation.
            if gpu:
                torch.cuda.synchronize(device)
        except torch.cuda.OutOfMemoryError:
            status = "OOM (incomplete epoch)"
        seconds = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated(device) / 2**20 if gpu else float("nan")
        mean_loss = torch.stack(losses).mean().item() if losses else float("nan")
        records.append(
            {
                "model": label,
                "epoch": epoch,
                "seconds": seconds,
                "peak_gpu_MiB": peak,
                "mse": mean_loss,
                "updates": len(losses),
                "status": status,
            }
        )
        if status != "ok":
            print(label, status, "after", len(losses), "updates; full mesh was not subsampled.")
            break
    model.zero_grad(set_to_none=True)
    model.eval()
    return records


def plot_epoch_costs(records: list[dict[str, Any]]) -> Figure:
    """Complete-epoch wall time, total peak CUDA allocation, and supervised loss."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5), layout="constrained")
    for label in dict.fromkeys(row["model"] for row in records):
        rows = [r for r in records if r["model"] == label and r["status"] == "ok"]
        for ax, key in zip(axes, ["seconds", "peak_gpu_MiB", "mse"]):
            ax.plot([r["epoch"] for r in rows], [r[key] for r in rows], "o-", label=label)
    for ax, ylabel in zip(
        axes, ["Seconds per epoch", "Peak allocated GPU memory [MiB]", "Training MSE"]
    ):
        ax.set(xlabel="Epoch", ylabel=ylabel)
        ax.grid(alpha=0.2)
    axes[0].legend()
    if not any(np.isfinite(r["peak_gpu_MiB"]) for r in records):
        axes[1].text(
            0.5, 0.5, "GPU memory unavailable on CPU", ha="center", transform=axes[1].transAxes
        )
    failures = [r["model"] for r in records if r["status"] != "ok"]
    if failures:
        fig.suptitle("Incomplete full-mesh run: " + ", ".join(failures))
    plt.show()
    return fig


def _retained_tri(
    sample: Data, ids: torch.Tensor | None
) -> tuple[mtri.Triangulation | None, np.ndarray, np.ndarray]:
    """Use original cells whose vertices all remain; never invent new connectivity."""
    pos, face = _array(sample.pos), _array(sample.face).T
    selected = np.arange(len(pos)) if ids is None else _array(ids)
    lookup = np.full(len(pos), -1, dtype=np.int64)
    lookup[selected] = np.arange(len(selected))
    local_faces = lookup[face]
    local_faces = local_faces[(local_faces >= 0).all(axis=1)]
    points = pos[selected]
    tri = mtri.Triangulation(points[:, 0], points[:, 1], local_faces) if len(local_faces) else None
    return tri, points, selected


def _map(
    ax: Axes,
    tri: mtri.Triangulation | None,
    points: np.ndarray,
    values: np.ndarray,
    vmax: float,
    cmap: str,
) -> Any:
    if tri is None:
        artist = ax.scatter(points[:, 0], points[:, 1], c=values, s=4, vmin=0, vmax=vmax, cmap=cmap)
    else:
        artist = ax.tripcolor(tri, values, shading="gouraud", vmin=0, vmax=vmax, cmap=cmap)
    _style(ax)
    return artist


def plot_vertex_maps(
    sample: Data,
    maps: Mapping[str, torch.Tensor],
    ids: torch.Tensor | None = None,
    title: str = "Prediction drift",
    quantile: float = 0.75,
) -> Figure | None:
    """Compare scalar maps at the same vertices with one shared quantile scale."""
    if not maps:
        return None
    tri, points, _ = _retained_tri(sample, ids)
    values = {name: _array(value) for name, value in maps.items()}
    vmax = max(float(np.quantile(np.concatenate(list(values.values())), quantile)), 1e-12)
    fig, axes = plt.subplots(
        1, len(maps), figsize=(5 * len(maps), 4), squeeze=False, layout="constrained"
    )
    for ax, (name, value) in zip(axes[0], values.items()):
        artist = _map(ax, tri, points, value, vmax, "magma")
        ax.set_title(name + f" · max {value.max():.3g}")
    fig.colorbar(artist, ax=axes.ravel().tolist(), label="Vector prediction drift", extend="max")
    fig.suptitle(title)
    plt.show()
    return fig


def plot_attention_maps(
    sample: Data,
    predictions: Mapping[str, torch.Tensor],
    baseline: torch.Tensor | None = None,
    ids: torch.Tensor | None = None,
    title: str = "Attention predictions",
    quantile: float = 0.75,
) -> Figure:
    """CFD/prediction speed, vector CFD error, and optional prediction drift.

    All arrays refer to the same retained vertex order. Each example has its own
    error/drift colorbar; speed always uses 0–8. Numerically tiny drift is reported
    through its actual maximum rather than interpreted as model dependence.
    """
    tri, points, selected = _retained_tri(sample, ids)
    truth = _array(sample.y)[selected]
    values = {name: _array(value) for name, value in predictions.items()}
    errors = {name: np.linalg.norm(value - truth, axis=-1) for name, value in values.items()}
    drift = (
        {}
        if baseline is None
        else {
            name: np.linalg.norm(value - _array(baseline), axis=-1)
            for name, value in values.items()
        }
    )
    nrows, ncols = (3 if drift else 2), len(values) + 1
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False, layout="constrained"
    )
    for ax in axes.ravel():
        _style(ax)
    speed = _map(axes[0, 0], tri, points, np.linalg.norm(truth, axis=-1), 8.0, "viridis")
    axes[0, 0].set_title("CFD next frame")
    for row, collection, ylabel in [
        (1, errors, "Vector error against CFD"),
        (2, drift, "Vector prediction drift"),
    ]:
        if not collection:
            continue
        vmax = max(float(np.quantile(np.concatenate(list(collection.values())), quantile)), 1e-12)
        for col, (name, value) in enumerate(collection.items(), 1):
            artist = _map(axes[row, col], tri, points, value, vmax, "magma")
            axes[row, col].set_title(ylabel + f" · max {value.max():.3g}")
        axes[row, 0].set_visible(False)
        fig.colorbar(artist, ax=axes[row, 1:].tolist(), label=ylabel, extend="max")
    for col, (name, value) in enumerate(values.items(), 1):
        _map(axes[0, col], tri, points, np.linalg.norm(value, axis=-1), 8.0, "viridis")
        l2 = np.linalg.norm(value - truth) / max(np.linalg.norm(truth), 1e-12)
        axes[0, col].set_title(name + f" · CFD L2 {l2:.3g}")
    fig.colorbar(speed, ax=axes[0].tolist(), label="Velocity magnitude", extend="max")
    fig.suptitle(title)
    plt.show()
    return fig
