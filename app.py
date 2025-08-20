import streamlit as st
import numpy as np
import math, time, random
from collections import deque, defaultdict
import plotly.graph_objects as go

# --------------------------
# Utility & data structures
# --------------------------

RNG = np.random.default_rng(42)

def dist(a, b):
    return float(np.hypot(a[0]-b[0], a[1]-b[1]))

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

class Packet:
    def __init__(self, t, ptype, src, dst=None, group=None, payload=None, ttl=64, color=None):
        self.time = t
        self.ptype = ptype  # 'DATA','RREQ','RREP','JOIN','JREP'
        self.src = src
        self.dst = dst
        self.group = group  # for multicast
        self.payload = payload or {}
        self.ttl = ttl
        self.color = color

class Node:
    def __init__(self, nid, pos, v=(0,0), energy=100.0, is_anchor=False, clock_offset=0.0):
        self.id = nid
        self.pos = np.array(pos, dtype=float)
        self.v = np.array(v, dtype=float)
        self.energy = energy
        self.is_anchor = is_anchor
        self.clock_offset = clock_offset

        # Routing state
        self.seqno = 0  # used by AODV/DSDV
        self.rtable = {}  # DSDV/AODV: dst -> (next_hop, hops, seqno, valid)
        self.precursors = defaultdict(set)  # AODV
        self.seq_seen = set()  # AODV seen RREQ (src, rreq_id)

        # Multicast ODMRP-like
        self.mesh_membership = set()  # groups where node is forwarder

        # Clustering / role
        self.is_cluster_head = False
        self.cluster_parent = None

        # DV-Hop localization
        self.dvhop_hops = {}      # anchor_id -> hop_count
        self.dvhop_dist_est = {}  # anchor_id -> estimated distance
        self.estimated_pos = None

        # Transport demo
        self.loss_rate_est = 0.0
        self.rtt_est = 1.0

    def tick_clock(self, dt):
        # simulate simple drift (already captured by offset held constant per run)
        pass

# --------------------------
# Simulation core
# --------------------------

class Sim:
    def __init__(self, n=25, area=500, comm_range=120, vehicular=False, seed=42):
        self.area = area
        self.comm_range = comm_range
        self.time = 0.0
        self.dt = 0.25
        self.vehicular = vehicular
        self.rand = np.random.default_rng(seed)

        self.nodes = []
        self.links = set()
        self.msg_queue = deque()
        self.sent_packets = 0
        self.delivered_packets = 0
        self.dropped_packets = 0
        self.avg_delay = 0.0
        self._delay_acc = 0.0
        self._delay_n = 0

        self.groups = {0: set()}  # multicast group 0 by default
        self.rreq_id = 0

        # transport toggles
        self.congestion_awareness = False

        self._init_nodes(n)

    def _init_nodes(self, n):
        self.nodes.clear()
        # anchors for DV-Hop
        anchor_ids = set(self.rand.choice(n, size=max(3, n//10), replace=False))
        for i in range(n):
            x = float(self.rand.uniform(0, self.area))
            y = float(self.rand.uniform(0, self.area))
            if self.vehicular:
                # Snap to lanes: horizontal/vertical lines every 100 units
                lanes = [100, 200, 300, 400]
                if self.rand.uniform() < 0.5:
                    y = float(self.rand.choice(lanes))
                else:
                    x = float(self.rand.choice(lanes))
            speed = self.rand.uniform(10, 30) if self.vehicular else self.rand.uniform(5, 15)
            ang = self.rand.uniform(0, 2*math.pi)
            vx, vy = speed*math.cos(ang), speed*math.sin(ang)
            is_anchor = (i in anchor_ids)
            clock_offset = self.rand.uniform(-0.2, 0.2)  # +/- 200 ms
            self.nodes.append(Node(i, (x, y), (vx, vy), energy=100.0, is_anchor=is_anchor, clock_offset=clock_offset))

    def rebuild_links(self):
        self.links.clear()
        n = len(self.nodes)
        for i in range(n):
            for j in range(i+1, n):
                if dist(self.nodes[i].pos, self.nodes[j].pos) <= self.comm_range:
                    self.links.add((i, j))
        # neighbor list cache
        self.neighbors = {i: set() for i in range(n)}
        for i, j in self.links:
            self.neighbors[i].add(j)
            self.neighbors[j].add(i)

    def step_mobility(self):
        for nd in self.nodes:
            nd.pos += nd.v * self.dt
            # bounce at borders (or wrap if vehicular)
            if self.vehicular:
                # keep on lanes but wrap around borders
                nd.pos[0] = (nd.pos[0] + self.area) % self.area
                nd.pos[1] = (nd.pos[1] + self.area) % self.area
            else:
                for k in [0,1]:
                    if nd.pos[k] < 0 or nd.pos[k] > self.area:
                        nd.v[k] *= -1
                        nd.pos[k] = clamp(nd.pos[k], 0, self.area)

        self.time += self.dt

    # --------------------------
    # Energy model & send cost
    # --------------------------
    def tx_cost(self, d):
        # simple free-space: E ~ d^2; scaled
        return 0.0005 * (d**2) + 0.02  # include small electronics cost

    def charge_for_link(self, u, v):
        d = dist(self.nodes[u].pos, self.nodes[v].pos)
        c = self.tx_cost(d)
        self.nodes[u].energy = max(0.0, self.nodes[u].energy - c)
        return c

    # --------------------------
    # DSDV (proactive) updates
    # --------------------------
    def dsdv_periodic(self):
        # Each node broadcasts its table to neighbors; simplified Bellman-Ford style
        for u in range(len(self.nodes)):
            table = {dst: (nh, hops, seq, valid) for dst, (nh, hops, seq, valid) in self.nodes[u].rtable.items() if valid}
            table[u] = (u, 0, self.nodes[u].seqno, True)
            for v in self.neighbors.get(u, []):
                # cost to v is 1 hop
                for dst, (nh, hops, seq, valid) in table.items():
                    cand = (u if dst == u else u, hops+1, seq, True)
                    if (dst not in self.nodes[v].rtable) or (seq > self.nodes[v].rtable[dst][2]) or (seq == self.nodes[v].rtable[dst][2] and hops+1 < self.nodes[v].rtable[dst][1]):
                        self.nodes[v].rtable[dst] = (u, hops+1, seq, True)

    # --------------------------
    # AODV (reactive) discovery
    # --------------------------
    def aodv_route(self, s, d):
        # BFS-level simplified AODV with RREQ/RREP bookkeeping
        self.rreq_id += 1
        rid = self.rreq_id
        q = deque([s])
        visited = {s}
        parents = {s: None}
        while q:
            u = q.popleft()
            if u == d:
                break
            for v in self.neighbors.get(u, []):
                if v not in visited and self.nodes[u].energy > 0 and self.nodes[v].energy > 0:
                    visited.add(v)
                    parents[v] = u
                    q.append(v)
        if d not in parents:
            return None
        # build path
        path = []
        cur = d
        while cur is not None:
            path.append(cur)
            cur = parents[cur]
        path.reverse()
        # Install next-hops along path (forward and reverse)
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            self.nodes[u].rtable[d] = (v, len(path)-i-1, self.nodes[u].seqno, True)
            self.nodes[v].rtable[s] = (u, i+1, self.nodes[v].seqno, True)
        return path

    # --------------------------
    # Energy-aware path variant
    # --------------------------
    def energy_aware_path(self, s, d):
        # Dijkstra-like with cost = (distance^2)/(min_residual_energy_along_path+eps)
        n = len(self.nodes)
        Q = set(range(n))
        dist_cost = {i: float('inf') for i in range(n)}
        best_prev = {i: None for i in range(n)}
        min_residual = {i: 0.0 for i in range(n)}
        dist_cost[s] = 0.0
        min_residual[s] = self.nodes[s].energy
        while Q:
            u = min(Q, key=lambda x: dist_cost[x])
            Q.remove(u)
            if dist_cost[u] == float('inf'): break
            if u == d: break
            for v in self.neighbors.get(u, []):
                if self.nodes[u].energy <= 0 or self.nodes[v].energy <= 0: 
                    continue
                hop_d = dist(self.nodes[u].pos, self.nodes[v].pos)
                hop_cost = (hop_d**2) / (min(self.nodes[u].energy, self.nodes[v].energy)+1e-6)
                alt = dist_cost[u] + hop_cost
                if alt < dist_cost[v]:
                    dist_cost[v] = alt
                    best_prev[v] = u
                    min_residual[v] = min(min_residual[u] if min_residual[u]>0 else self.nodes[u].energy, self.nodes[v].energy)
        if dist_cost[d] == float('inf'):
            return None
        path = []
        cur = d
        while cur is not None:
            path.append(cur)
            cur = best_prev[cur]
        path.reverse()
        return path

    # --------------------------
    # ODMRP-like multicast
    # --------------------------
    def odmrp_mesh_build(self, group_id, src):
        # Flood JOIN; receivers reply JREP forming a mesh (simplified)
        members = list(self.groups.get(group_id, []))
        if src not in members:
            members.append(src)
        # Build shortest paths from src to each member (using current neighbor graph)
        mesh = set()
        for m in members:
            path = self.aodv_route(src, m)
            if path:
                for i in range(len(path)-1):
                    mesh.add((path[i], path[i+1]))
                    mesh.add((path[i+1], path[i]))
        # Mark nodes on mesh as forwarders for this group
        for nd in self.nodes:
            # check if node participates in any mesh edge
            is_member = any((nd.id==u or nd.id==v) for (u,v) in mesh)
            if is_member:
                nd.mesh_membership.add(group_id)
            else:
                if group_id in nd.mesh_membership:
                    nd.mesh_membership.remove(group_id)
        return mesh

    # --------------------------
    # Transport "TCP-ish" demo
    # --------------------------
    def tx_delay(self, hops):
        base = 0.02 * hops  # 20ms per hop
        if self.congestion_awareness:
            # add queueing as function of degree
            deg = np.mean([len(self.neighbors[i]) for i in range(len(self.nodes))]) + 1e-6
            base *= (1.0 + 0.05*deg)
        return base

    # --------------------------
    # Send unicast or multicast
    # --------------------------
    def send_unicast(self, s, d, mode='AODV'):
        if s == d: 
            return True, [s], 0.0
        if mode == 'AODV':
            path = self.aodv_route(s, d)
        elif mode == 'DSDV':
            # follow table if exists else fall back to AODV
            path = self.get_path_from_table(s, d)
            if path is None:
                path = self.aodv_route(s, d)
        elif mode == 'ENERGY':
            path = self.energy_aware_path(s, d)
        else:
            path = self.aodv_route(s, d)
        if not path:
            self.dropped_packets += 1
            return False, None, 0.0
        # charge energy and simulate delivery
        delay = self.tx_delay(len(path)-1)
        ok = True
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            if (u not in self.neighbors) or (v not in self.neighbors[u]):
                ok = False
                break
            self.charge_for_link(u, v)
        self.sent_packets += 1
        if ok:
            self.delivered_packets += 1
            self._delay_acc += delay
            self._delay_n += 1
        else:
            self.dropped_packets += 1
        return ok, path, delay

    def send_multicast(self, s, group_id):
        # forward along current mesh (build/refresh first)
        mesh = self.odmrp_mesh_build(group_id, s)
        members = list(self.groups.get(group_id, []))
        reached = set()
        if not mesh:
            self.dropped_packets += 1
            return False, set(), 0.0
        # BFS along mesh edges
        adj = defaultdict(set)
        for u, v in mesh:
            adj[u].add(v)
        q = deque([s])
        seen = {s}
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    self.charge_for_link(u, v)
                    q.append(v)
                    if v in members:
                        reached.add(v)
        self.sent_packets += 1
        delay = self.tx_delay(max(1, len(seen)-1))
        if reached >= set(members):
            self.delivered_packets += 1
            self._delay_acc += delay
            self._delay_n += 1
            return True, reached, delay
        else:
            self.dropped_packets += 1
            return False, reached, delay

    def get_path_from_table(self, s, d):
        # Reconstruct via next-hops if table is complete
        path = [s]
        cur = s
        visited = set([s])
        while cur != d:
            if d not in self.nodes[cur].rtable: 
                return None
            nh, _, _, valid = self.nodes[cur].rtable[d]
            if not valid: 
                return None
            if nh in visited: 
                return None
            path.append(nh)
            visited.add(nh)
            cur = nh
            if len(path) > len(self.nodes): 
                return None
        return path

    # --------------------------
    # Clustering (LEACH-ish)
    # --------------------------
    def elect_cluster_heads(self, p=0.1, mobile_code=None):
        # Reset roles
        for nd in self.nodes:
            nd.is_cluster_head = False
            nd.cluster_parent = None

        for nd in self.nodes:
            choose = False
            if mobile_code:
                try:
                    choose = bool(mobile_code(nd=nd, neighbors=list(self.neighbors.get(nd.id, [])), energy=nd.energy))
                except Exception:
                    choose = False
            else:
                # Probabilistic, bias by residual energy
                bias = clamp(nd.energy/100.0, 0.1, 1.0)
                choose = (self.rand.uniform() < p*bias)

            nd.is_cluster_head = choose

        # attach members to nearest head
        heads = [nd for nd in self.nodes if nd.is_cluster_head]
        for nd in self.nodes:
            if nd.is_cluster_head: 
                continue
            if not heads:
                continue
            best = min(heads, key=lambda h: dist(h.pos, nd.pos))
            nd.cluster_parent = best.id

    # --------------------------
    # Time sync (pairwise)
    # --------------------------
    def sync_offsets(self):
        # choose a root and set others by averaging pairwise messages along 1-hop neighbors
        root = 0
        self.nodes[root].clock_offset = 0.0
        for nd in self.nodes:
            if nd.id == root: 
                continue
            neigh = list(self.neighbors.get(nd.id, []))
            if not neigh: 
                continue
            # pretend we exchanged timestamps with neighbors; estimate as mean neighbor offset
            est = np.mean([self.nodes[v].clock_offset + 0.005 for v in neigh]) if neigh else nd.clock_offset
            nd.clock_offset = float(0.5*nd.clock_offset + 0.5*est)

    # --------------------------
    # DV-Hop Localization
    # --------------------------
    def dvhop_round(self):
        # Initialize hop-counts from anchors
        anchors = [nd.id for nd in self.nodes if nd.is_anchor]
        # 1) hop count flooding
        changed = True
        for nd in self.nodes:
            nd.dvhop_hops = {a: (0 if nd.id==a else (1 if a in self.neighbors.get(nd.id, []) else np.inf)) for a in anchors}
        while changed:
            changed = False
            for u in range(len(self.nodes)):
                for v in self.neighbors.get(u, []):
                    for a in anchors:
                        alt = self.nodes[v].dvhop_hops[a] + 1
                        if alt < self.nodes[u].dvhop_hops[a]:
                            self.nodes[u].dvhop_hops[a] = alt
                            changed = True
        # 2) compute average hop distance using inter-anchor true distances
        if len(anchors) >= 2:
            dsum, hsum = 0.0, 0.0
            for i in range(len(anchors)):
                for j in range(i+1, len(anchors)):
                    ai, aj = anchors[i], anchors[j]
                    dsum += dist(self.nodes[ai].pos, self.nodes[aj].pos)
                    hsum += self.nodes[ai].dvhop_hops[aj]
            avg_hop = dsum / max(hsum, 1e-6)
        else:
            avg_hop = 1.0

        # 3) nodes estimate distance to anchors
        for nd in self.nodes:
            nd.dvhop_dist_est = {}
            for a in anchors:
                hops = self.nodes[nd.id].dvhop_hops[a]
                nd.dvhop_dist_est[a] = float(hops * avg_hop)

        # 4) trilateration (least squares with up to 3 anchors)
        for nd in self.nodes:
            if nd.is_anchor:
                nd.estimated_pos = nd.pos.copy()
                continue
            if len(anchors) < 3:
                nd.estimated_pos = None
                continue
            sel = anchors[:3]
            A = []
            b = []
            x1, y1 = self.nodes[sel[0]].pos
            r1 = nd.dvhop_dist_est[sel[0]]
            for a in sel[1:]:
                xa, ya = self.nodes[a].pos
                ra = nd.dvhop_dist_est[a]
                A.append([2*(xa - x1), 2*(ya - y1)])
                b.append(r1**2 - ra**2 - x1**2 + xa**2 - y1**2 + ya**2)
            try:
                A = np.array(A, dtype=float)
                b = np.array(b, dtype=float)
                xy, *_ = np.linalg.lstsq(A, b, rcond=None)
                nd.estimated_pos = xy
            except Exception:
                nd.estimated_pos = None

    # --------------------------
    # One simulation tick
    # --------------------------
    def tick(self, force_dsdv=False, do_sync=False, do_dvhop=False):
        self.step_mobility()
        self.rebuild_links()
        if force_dsdv:
            for nd in self.nodes:
                nd.seqno += 1
            self.dsdv_periodic()
        if do_sync:
            self.sync_offsets()
        if do_dvhop:
            self.dvhop_round()
        if self._delay_n > 0:
            self.avg_delay = self._delay_acc / self._delay_n

# --------------------------
# Streamlit UI
# --------------------------

st.set_page_config(page_title="MANET-Studio", layout="wide")

st.title("MANET-Studio: Advanced Ad Hoc, Sensor & Vehicular Network Simulator")
st.caption("AODV • DSDV • Energy-aware routing • ODMRP multicast • Clustering with mobile code • Time sync • DV-Hop localization • Vehicular lanes")

# Sidebar controls
with st.sidebar:
    st.header("Setup")
    N = st.number_input("Nodes", 10, 150, 35, step=5)
    AREA = st.number_input("Area (units)", 200, 1200, 600, step=50)
    CR = st.slider("Comm range", 50, 300, 130, step=5)
    vehicular = st.checkbox("Vehicular lanes mobility", value=False)
    protocol = st.selectbox("Unicast Routing", ["AODV", "DSDV", "ENERGY"])
    use_congestion = st.checkbox("Transport congestion awareness", value=False)
    auto_step = st.checkbox("Run simulation", value=False)
    speed = st.slider("Speed (steps/sec)", 1, 20, 8)
    dsdv_every = st.slider("DSDV update every X ticks", 1, 20, 5)
    do_sync = st.checkbox("Enable time sync", value=False)
    do_dvhop = st.checkbox("Enable DV-Hop localization", value=False)

    st.divider()
    st.subheader("Clustering & Mobile Code")
    p_heads = st.slider("Cluster-head probability p", 1, 50, 12) / 100.0
    mobile_code_text = st.text_area(
        "Optional: mobile code for CH election\n"
        "Define: def mobile_code(nd, neighbors, energy): return True/False",
        value="def mobile_code(nd, neighbors, energy):\n    # Favor high-energy nodes with degree >= 3\n    return (energy > 60) and (len(neighbors) >= 3)\n",
        height=140
    )
    apply_cluster = st.button("Elect cluster heads")

    st.divider()
    st.subheader("Traffic")
    src = st.number_input("Unicast src", 0, max(0, N-1), 0)
    dst = st.number_input("Unicast dst", 0, max(0, N-1), 1)
    send_u = st.button("Send unicast")
    st.caption("Multicast group 0 below")
    add_member = st.number_input("Add member (node id) to group 0", 0, max(0, N-1), 2)
    add_btn = st.button("Add member")
    send_m = st.button("Send multicast from src")

# Session state init
if "sim" not in st.session_state:
    st.session_state.sim = Sim(n=N, area=AREA, comm_range=CR, vehicular=vehicular)
    st.session_state.tick_count = 0
    st.session_state.last_dsdv = 0

sim: Sim = st.session_state.sim

# Rebuild if params changed
def maybe_reset():
    changed = (len(sim.nodes) != N) or (sim.area != AREA) or (sim.comm_range != CR) or (sim.vehicular != vehicular)
    if changed:
        st.session_state.sim = Sim(n=N, area=AREA, comm_range=CR, vehicular=vehicular)
        st.session_state.tick_count = 0
        st.session_state.last_dsdv = 0

maybe_reset()
sim = st.session_state.sim
sim.congestion_awareness = use_congestion

# Apply clustering (with optional mobile code)
if apply_cluster:
    mobile_code_fn = None
    if mobile_code_text.strip():
        try:
            loc = {}
            exec(mobile_code_text, {}, loc)
            mobile_code_fn = loc.get("mobile_code", None)
        except Exception as e:
            st.warning(f"Mobile code error: {e}")
            mobile_code_fn = None
    sim.elect_cluster_heads(p=p_heads, mobile_code=mobile_code_fn)

# Traffic actions
if add_btn:
    sim.groups.setdefault(0, set()).add(int(add_member))

sent_info = None
if send_u:
    ok, path, delay = sim.send_unicast(int(src), int(dst), mode=protocol)
    if ok:
        sent_info = ("Unicast delivered", path, delay)
    else:
        sent_info = ("Unicast failed", path, delay)

if send_m:
    ok, reached, delay = sim.send_multicast(int(src), 0)
    if ok:
        sent_info = ("Multicast delivered", sorted(list(reached)), delay)
    else:
        sent_info = ("Multicast partial/failed", sorted(list(reached)), delay)

# Periodic DSDV & sync/DV-Hop
if auto_step:
    # multiple ticks per refresh to smooth animation at higher speeds
    steps = max(1, int(speed//2))
    for _ in range(steps):
        st.session_state.tick_count += 1
        sim.tick(force_dsdv=(st.session_state.tick_count - st.session_state.last_dsdv >= dsdv_every),
                 do_sync=do_sync, do_dvhop=do_dvhop)
        if st.session_state.tick_count - st.session_state.last_dsdv >= dsdv_every:
            st.session_state.last_dsdv = st.session_state.tick_count
else:
    # single step advance on each rerun
    sim.tick(force_dsdv=False, do_sync=do_sync, do_dvhop=do_dvhop)

# ------------- Visualization -------------

# Build node scatter (true vs estimated pos)
xs = [nd.pos[0] for nd in sim.nodes]
ys = [nd.pos[1] for nd in sim.nodes]
energies = [nd.energy for nd in sim.nodes]
labels = [f"Node {nd.id}"
          + (", Anchor" if nd.is_anchor else "")
          + (", CH" if nd.is_cluster_head else "")
          + (f", parent:{nd.cluster_parent}" if nd.cluster_parent is not None else "")
          for nd in sim.nodes]

fig = go.Figure()
# Links
for (u, v) in sim.links:
    fig.add_trace(go.Scatter(
        x=[sim.nodes[u].pos[0], sim.nodes[v].pos[0]],
        y=[sim.nodes[u].pos[1], sim.nodes[v].pos[1]],
        mode="lines",
        line=dict(width=1),
        hoverinfo="none",
        showlegend=False
    ))

# Nodes (size by energy)
sizes = [8 + 10*(e/100.0) for e in energies]
marker_colors = ["red" if nd.is_cluster_head else ("orange" if nd.is_anchor else "blue") for nd in sim.nodes]

fig.add_trace(go.Scatter(
    x=xs, y=ys, mode="markers+text",
    text=[nd.id for nd in sim.nodes],
    textposition="top center",
    marker=dict(size=sizes, color=marker_colors, line=dict(width=1, color="black")),
    hovertext=labels, hoverinfo="text", name="nodes"
))

# Estimated positions (DV-Hop)
if any(nd.estimated_pos is not None and not nd.is_anchor for nd in sim.nodes):
    ex = []
    ey = []
    for nd in sim.nodes:
        if nd.estimated_pos is not None and not nd.is_anchor:
            ex.append(nd.estimated_pos[0])
            ey.append(nd.estimated_pos[1])
    fig.add_trace(go.Scatter(
        x=ex, y=ey, mode="markers",
        marker=dict(symbol="x", size=8),
        name="DV-Hop Estimate"
    ))

fig.update_layout(
    width=800, height=700,
    xaxis=dict(range=[0, sim.area]),
    yaxis=dict(range=[0, sim.area]),
    title=f"Topology @ t={sim.time:.2f}s | Links={len(sim.links)}",
    margin=dict(l=10, r=10, t=35, b=10)
)

# Metrics panel
col1, col2 = st.columns([1.2, 1])
with col1:
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
with col2:
    st.subheader("Stats")
    st.metric("Sent", sim.sent_packets)
    st.metric("Delivered", sim.delivered_packets)
    st.metric("Dropped", sim.dropped_packets)
    st.metric("Avg delay (s)", round(sim.avg_delay, 4) if sim.avg_delay>0 else 0.0)
    energy_left = round(np.mean([nd.energy for nd in sim.nodes]), 2)
    st.metric("Mean energy", energy_left)
    st.write(f"Group 0 members: {sorted(list(sim.groups.get(0, set())))}")
    if sent_info:
        st.success(f"{sent_info[0]} | Path/Reached: {sent_info[1]} | Delay ~ {sent_info[2]:.3f}s")

    if do_sync:
        off = [round(nd.clock_offset, 4) for nd in sim.nodes]
        st.write("Clock offsets (s):", off[:min(15, len(off))], "…")

    if do_dvhop:
        # quick localization error if estimates exist
        errs = []
        for nd in sim.nodes:
            if nd.estimated_pos is not None:
                errs.append(dist(nd.pos, nd.estimated_pos))
        if errs:
            st.write(f"DV-Hop mean error: {np.mean(errs):.2f}")

st.caption("Tip: toggle protocols, add multicast members, and paste custom mobile code to re-elect cluster heads live. Try ENERGY routing when nodes start dying; routes will shift to preserve minimum residual energy.")
