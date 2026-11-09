# Parking Lot Occupancy and Vehicle Trajectory Prediction System

# Import the necessary libraries
import ast
import csv
import hashlib
import json
import os
import shutil
import subprocess
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks
from ultralytics import YOLO
import config

# Paths (file names loaded from config.py, resolved relative to this script's folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path = lambda name: os.path.join(BASE_DIR, name)

# Colors (BGR) (loaded from config.py)
COLOR_OCCUPIED = config.COLOR_OCCUPIED
COLOR_EMPTY = config.COLOR_EMPTY

# Fonts
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


# Function to pick the fastest available inference device
def pick_device():
    if config.DEVICE:
        return config.DEVICE
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Function to read a single frame from the video
def read_frame(video_path, index):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {index} of {video_path}")
    return frame


# Function to detect objects and flag which detections are parked-vehicle classes
def detect(model, frame, conf, device):
    r = model.predict(frame, imgsz=config.IMG_SIZE, conf=conf, max_det=config.MAX_DET,
                      device=device, verbose=False)[0]
    boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
    cls = r.boxes.cls.cpu().numpy().astype(int)
    is_vehicle = np.array([r.names[c] in config.VEHICLE_CLASSES for c in cls], dtype=bool)
    return boxes, is_vehicle


# Function to convert xyxy boxes into 4-corner polygons
def box_corners(boxes):
    if len(boxes) == 0:
        return np.zeros((0, 4, 2))
    return np.stack([boxes[:, [0, 1]], boxes[:, [2, 1]], boxes[:, [2, 3]], boxes[:, [0, 3]]], 1)


# Function to apply a 2x3 affine matrix to (..., 2) points
def apply_affine(pts, M):
    return np.asarray(pts, np.float64) @ M[:, :2].T + M[:, 2]


# Function to apply a 3x3 homography to (..., 2) points
def apply_homography(pts, H):
    shape = np.shape(pts)
    p = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, H).reshape(shape)


# Function to rasterize boxes into a binary mask (slightly shrunk to keep nearby lines)
def boxes_to_mask(shape, boxes, shrink=3):
    m = np.zeros(shape, np.uint8)
    for x1, y1, x2, y2 in np.round(boxes).astype(int):
        m[max(0, y1 + shrink):max(0, y2 - shrink + 1), max(0, x1 + shrink):max(0, x2 - shrink + 1)] = 1
    return m


# Function to build painted line masks (top-hat on max(B,G,R), vehicles masked out)
def line_masks(img, boxes):
    v = img.max(axis=2)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (config.TOPHAT_KERNEL, config.TOPHAT_KERNEL))
    raw = (cv2.morphologyEx(v, cv2.MORPH_TOPHAT, k) > config.LINE_THRESH).astype(np.uint8)
    clean = raw & (1 - boxes_to_mask(raw.shape, boxes))
    hor = cv2.morphologyEx(clean, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (config.HOR_KERNEL, 1)))
    ver = cv2.morphologyEx(clean, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, config.VER_KERNEL)))
    return raw, clean, hor, ver


# Function to estimate the lot rotation so that slot dividers become horizontal
def estimate_skew(clean):
    small = cv2.resize(clean * 255, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA).astype(np.float32)
    h, w = small.shape

    # Coarse search: angle that gives the sharpest row projection
    def score(a):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        return float(np.sum(cv2.warpAffine(small, M, (w, h)).sum(1) ** 2))

    best = max(np.arange(-config.MAX_SKEW_DEG, config.MAX_SKEW_DEG + 1e-6, 1.0), key=score)
    angle = float(max(np.arange(best - 1.0, best + 1.0 + 1e-6, 0.1), key=score))

    # Fine refinement: fit lines to divider segments until the residual slope is ~0
    H, W = clean.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    for _ in range(3):
        M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
        hor = cv2.morphologyEx(cv2.warpAffine(clean, M, (W, H), flags=cv2.INTER_NEAREST), cv2.MORPH_OPEN, kernel)
        n, lab, st, _ = cv2.connectedComponentsWithStats(hor)
        slopes = []
        for i in range(1, n):
            x, y, bw, bh, _ = st[i]
            if bw >= 40 and bh <= 7:
                ys, xs = np.where(lab[y:y + bh, x:x + bw] == i)
                slopes.append(np.polyfit(xs, ys, 1)[0])
        if len(slopes) < 10:
            break
        residual = float(np.degrees(np.arctan(np.median(slopes))))
        angle += residual
        if abs(residual) < 0.01:
            break
    return angle


# Function to build a rotation matrix that rotates the image without cropping
def rotation_matrix(w, h, angle):
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(round(h * sin + w * cos)), int(round(h * cos + w * sin))
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return M, (nw, nh)


# Function to find parking lanes as plateaus of the horizontal line histogram
def find_runs(hor):
    hx = np.convolve(hor.sum(0).astype(float), np.ones(5) / 5, "same")
    nz = hx[hx > 0]
    if nz.size == 0:
        return [], hx
    on = hx > 0.15 * np.percentile(nz, 95)
    runs, i = [], 0
    while i < len(hx):
        if on[i]:
            j = i
            while j < len(hx) and on[j]:
                j += 1
            runs.append((i, j - 1))
            i = j
        else:
            i += 1
    return runs, hx


# Function to get the divider profile of a lane (fraction of lane width covered by lines per row)
def lane_profile(hor, a, b):
    prof = hor[:, a:b + 1].mean(1).astype(float)
    return np.maximum.reduce([prof, np.roll(prof, 1), np.roll(prof, -1)])


# Function to find divider peaks with sub-pixel (centroid) accuracy
def subpixel_peaks(prof, height, distance):
    peaks, _ = find_peaks(prof, height=height, distance=max(1, int(distance)))
    out = []
    for y in peaks:
        lo, hi = max(0, y - 2), min(len(prof), y + 3)
        w = prof[lo:hi]
        out.append(float((w * np.arange(lo, hi)).sum() / w.sum()))
    return np.array(out)


# Function to estimate the slot pitch (distance between dividers) from all lanes
def estimate_pitch(hor, runs):
    diffs = []
    for a, b in runs:
        d = np.diff(subpixel_peaks(lane_profile(hor, a, b), config.PEAK_HEIGHT, 8))
        diffs += list(d[(d > 8) & (d < 200)])
    if not diffs:
        raise RuntimeError("No parking divider lines were found.")
    hist, edges = np.histogram(diffs, bins=np.arange(8, 201, 1.0))
    mode = edges[np.argmax(np.convolve(hist, [1, 2, 3, 2, 1], "same"))] + 0.5
    d = np.array(diffs)
    return float(np.median(d[np.abs(d - mode) < 0.2 * mode]))


# Function to refine lane edges and split double rows at the painted centre line
def split_lanes(runs, hx, ver, pitch):
    lanes = []
    for a, b in runs:
        if b - a < 1.2 * pitch:
            continue
        idx = np.where(hx[a:b + 1] > 0.35 * np.median(hx[a:b + 1]))[0]
        a2, b2 = a + int(idx[0]), a + int(idx[-1])
        w = b2 - a2
        if w > 3.3 * pitch:
            lo, hi = a2 + w // 3, b2 - w // 3
            col = ver[:, lo:hi].sum(0)
            c = lo + int(np.argmax(col)) if col.max() > 0 else (a2 + b2) // 2
            lanes += [(a2, c), (c, b2)]
        else:
            lanes.append((a2, b2))
    return lanes


# Class to decide whether a candidate slot is a vehicle, empty asphalt or invalid (island / hatch)
class SlotValidator:
    def __init__(self, img, raw, vehicle_boxes, valid, max_car_width):
        self.H = img.shape[0]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        # Islands: large bright, low-saturation concrete areas
        self.island = cv2.morphologyEx(((hsv[..., 2] > 160) & (hsv[..., 1] < 100)).astype(np.uint8),
                                       cv2.MORPH_OPEN, k7)
        # Hatched zones: diagonal stripes in yellow paint
        k1 = np.eye(9, dtype=np.uint8)
        diag = cv2.morphologyEx(raw, cv2.MORPH_OPEN, k1) | cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.fliplr(k1).copy())
        yellow = ((hsv[..., 1] > 45) & (hsv[..., 0] > 8) & (hsv[..., 0] < 35)).astype(np.uint8)
        self.hatch = cv2.dilate(diag, np.ones((7, 7), np.uint8)) & yellow
        # Painted no-parking zones: bright yellow paint (hatched aisles, loading bands)
        paint = (yellow & (hsv[..., 1] > 55) & (hsv[..., 2] > 90)).astype(np.uint8)   # concrete: S < 50
        n, lab, st, _ = cv2.connectedComponentsWithStats(paint)
        big = (st[:, cv2.CC_STAT_AREA] >= 150) | (st[:, cv2.CC_STAT_WIDTH] >= 25)
        big[0] = False
        self.paint = big[lab].astype(np.float32)
        self.vehicle_mask = boxes_to_mask(raw.shape, vehicle_boxes)
        # Vehicles parked inside a slot (vehicles parked across a painted band are wider)
        car = vehicle_boxes[(vehicle_boxes[:, 2] - vehicle_boxes[:, 0]) <= max_car_width] \
            if len(vehicle_boxes) else vehicle_boxes
        self.car_mask = boxes_to_mask(raw.shape, car, shrink=-2)
        self.max_car_width = max_car_width
        self.vehicles = vehicle_boxes
        self.valid = valid
        self._rows = {}

    # Carry a painted band hidden under a wide vehicle over from the other row of the lane
    def mark_crossing_vehicles(self, lanes):
        for x1, y1, x2, y2 in self.vehicles:
            if x2 - x1 <= self.max_car_width:
                continue
            cols = [(a, b) for a, b in lanes if min(x2, b) - max(x1, a) > 0.5 * (b - a)]
            if any(self.painted(a, b, y1, y2) >= 0.3 for a, b in cols):
                for a, b in cols:
                    self.painted_rows(a, b)[int(y1):int(np.ceil(y2)) + 1] = True

    # Boolean per-row flag of a lane: row is covered by yellow paint (parked cars ignored)
    def painted_rows(self, a, b):
        if (a, b) not in self._rows:
            free = 1.0 - self.car_mask[:, a + 4:b - 3]
            n = free.sum(1)
            frac = np.where(n >= 8, (self.paint[:, a + 4:b - 3] * free).sum(1) / np.maximum(n, 1), 0.0)
            rows = (np.convolve(frac, np.ones(3) / 3, "same") > config.PAINT_ROW_FRAC).astype(np.uint8)
            rows = cv2.morphologyEx(rows[:, None], cv2.MORPH_CLOSE, np.ones((7, 1), np.uint8))
            rows = cv2.morphologyEx(rows, cv2.MORPH_OPEN, np.ones((7, 1), np.uint8))   # drop thin lines
            self._rows[(a, b)] = rows[:, 0].astype(bool)
        return self._rows[(a, b)]

    # Fraction of the rows of [ya, yb] that are painted
    def painted(self, a, b, ya, yb):
        rows = self.painted_rows(a, b)[max(0, int(round(ya))):max(0, int(round(yb)))]
        return float(rows.mean()) if rows.size else 0.0

    # Fraction of the lane width where bright concrete fills most of the rows of [ya, yb]
    def island_span(self, a, b, ya, yb):
        m = self.island[int(ya) + 3:int(yb) - 2, a + 3:b - 2]
        return float((m.mean(0) > 0.5).mean()) if m.size else 0.0

    # Fraction of the lane area [ya, yb] covered by vehicles
    def covered(self, a, b, ya, yb):
        m = self.vehicle_mask[max(0, int(ya)):max(0, int(yb)), a:b]
        return float(m.mean()) if m.size else 0.0

    def vehicle_cover(self, a, b, ya, yb):
        if len(self.vehicles) == 0:
            return 0.0
        v = self.vehicles
        ix = np.clip(np.minimum(v[:, 2], b) - np.maximum(v[:, 0], a), 0, None)
        iy = np.clip(np.minimum(v[:, 3], yb) - np.maximum(v[:, 1], ya), 0, None)
        return float((ix * iy).max() / ((b - a) * (yb - ya)))

    # Part of [ya, yb] that lies inside the video frame over the whole lane width
    def visible(self, a, b, ya, yb):
        y0, y1 = max(0, int(np.floor(ya))), min(self.H, int(np.ceil(yb)))
        rows = np.flatnonzero(self.valid[y0:y1, a:b + 1].min(1)) if y1 > y0 else np.zeros(0, int)
        if rows.size == 0:
            return ya, ya
        return max(ya, y0 + rows[0]), min(yb, y0 + rows[-1] + 1)

    def kind(self, a, b, ya, yb, inset=3, paint=True, hatch=True, island=True):
        full = yb - ya
        ya, yb = self.visible(a, b, ya, yb)
        if yb - ya < config.EDGE_MIN_VISIBLE * full:
            return "bad"
        r = (slice(int(ya) + inset, int(yb) - inset + 1), slice(a + 3, b - 2))
        if self.valid[r].mean() < 0.95:
            return "bad"
        if paint and self.painted(a, b, ya, yb) > config.PAINT_SLOT_FRAC:
            return "bad"
        free = (1 - self.vehicle_mask[r]).astype(bool)
        cover = self.vehicle_cover(a, b, ya, yb)
        if free.sum() >= 0.25 * free.size:
            if hatch and self.hatch[r][free].mean() > config.HATCH_FRAC:
                return "bad"
            if island and cover < config.CAR_COVER and self.island[r][free].mean() > config.ISLAND_FRAC:
                return "bad"
        return "vehicle" if cover >= config.CAR_COVER else "asphalt"


# Function to extract the slots of one lane: divider grid + hidden dividers + row-end extension
def lane_slots(hor, lane, pitch, val):
    a, b = lane
    prof = lane_profile(hor, a, b)
    ys = subpixel_peaks(prof, config.PEAK_HEIGHT, 0.6 * pitch)
    if len(ys) < 2:
        return []
    d = np.diff(ys)
    good = np.abs(d - pitch) < 0.2 * pitch
    lp = float(np.median(d[good])) if good.any() else pitch   # lane specific pitch

    # Chain consecutive dividers into rows, filling dividers hidden by vehicles
    groups, cur = [], [(ys[0], True)]
    for y0, y1 in zip(ys, ys[1:]):
        k = int(round((y1 - y0) / lp))
        # Long gaps are allowed only when large vehicles (trucks, trailers) hide the dividers
        max_k = config.MAX_MISSING_COVERED if val.covered(a, b, y0, y1) >= config.COVERED_FRAC else config.MAX_MISSING
        if 1 <= k <= max_k and abs((y1 - y0) - k * lp) <= config.GAP_TOL * lp:
            cur += [(y0 + (y1 - y0) * j / k, j == k) for j in range(1, k + 1)]
        else:
            groups.append(cur)
            cur = [(y1, True)]
    groups.append(cur)

    slots, H = [], len(prof)
    for g in groups:
        if len(g) < config.MIN_GROUP:
            continue
        strength = [prof[max(0, int(round(y)) - 1):int(round(y)) + 2].max() for y, real in g if real]
        if np.median(strength) < config.MIN_LINE_STRENGTH:
            continue
        bounds = [y for y, _ in g]

        # Extend row ends with the grid (slots bounded by an island instead of a painted line)
        for direction in (-1, 1):
            for _ in range(config.EXTRAPOLATE):
                edge = bounds[0] if direction < 0 else bounds[-1]
                ya, yb = sorted([edge, edge + direction * lp])
                # Next to a painted zone the slots are irregular -> handled below
                if val.painted(a, b, *sorted([edge, edge + direction * 1.5 * lp])) > 0:
                    break
                kind = val.kind(a, b, ya, yb)
                if kind == "bad":
                    break
                if kind == "asphalt":
                    ny = edge + direction * lp
                    nyi = int(round(ny))
                    weak_line = 0 <= nyi < H and prof[max(0, nyi - 2):nyi + 3].max() > 0.02
                    na, nb = sorted([ny, ny + direction * 0.8 * lp])
                    if not (weak_line or val.kind(a, b, na, nb) == "bad"):
                        break
                if direction < 0:
                    bounds.insert(0, edge - lp)
                else:
                    bounds.append(edge + lp)

        pairs = list(zip(bounds, bounds[1:]))
        ok = [val.kind(a, b, ya, yb) != "bad" for ya, yb in pairs]
        # A bright undetected vehicle between two valid slots does not span the whole lane like an island
        for i in range(1, len(pairs) - 1):
            if not ok[i] and ok[i - 1] and ok[i + 1] and val.island_span(a, b, *pairs[i]) < 0.6:
                ok[i] = val.kind(a, b, *pairs[i], island=False) != "bad"
        slots += [p for p, good in zip(pairs, ok) if good]

    # Irregular slots between painted zones (e.g. accessible spaces separated by hatched aisles)
    grid, slots = slots, []
    painted = val.painted_rows(a, b).astype(np.int8)
    edges = np.flatnonzero(np.diff(painted)) + 0.5
    runs = [(s, e) for s, e in zip(edges[painted[0]::2], edges[painted[0] + 1::2])]
    cand = np.sort(np.concatenate([ys[prof[np.round(ys).astype(int)] >= 0.3], edges]))
    for ya, yb in zip(cand, cand[1:]):
        paint_ok = False
        if yb - ya < config.IRREGULAR_MIN * lp:
            # Take the missing length from an unusually long neighbouring painted run
            target = 1.1 * lp
            for s, e in runs:
                if abs(e - ya) < 1 and e - s > lp and e - s - (target - (yb - ya)) >= 0.5 * lp:
                    ya, paint_ok = yb - target, True
                elif abs(s - yb) < 1 and e - s > lp and e - s - (target - (yb - ya)) >= 0.5 * lp:
                    yb, paint_ok = ya + target, True
            if not paint_ok or yb - ya < config.IRREGULAR_MIN * lp:
                continue
        if yb - ya > config.IRREGULAR_MAX * lp:
            continue
        near = val.painted(a, b, ya - 0.5 * lp, ya) > 0 or val.painted(a, b, yb, yb + 0.5 * lp) > 0
        if not near or any(min(yb, y1) - max(ya, y0) > 0.3 * lp for y0, y1 in slots):
            continue
        kind = val.kind(a, b, ya, yb, inset=7, paint=not paint_ok, hatch=False)
        if kind == "bad":
            continue
        # An empty space must be next to an aisle-sized painted block, not to a large painted zone
        if kind == "asphalt":
            nb = [e - s for s, e in runs if ya - 0.5 * lp <= e <= ya + 1 or yb - 1 <= s <= yb + 0.5 * lp]
            if max(nb, default=np.inf) > 1.5 * lp:
                continue
        slots.append((ya, yb))
    slots += [(y0, y1) for y0, y1 in grid if all(min(y1, yb) - max(y0, ya) <= 0.3 * lp for ya, yb in slots)]

    # Slots cut by the frame border keep only their visible part
    slots = [val.visible(a, b, ya, yb) for ya, yb in sorted(slots)]
    return [np.array([[a, ya], [b, ya], [b, yb], [a, yb]], np.float64) for ya, yb in slots]


# Function to extract all parking slots of the lot from one frame
def extract_slots(frame, boxes, is_vehicle):
    h, w = frame.shape[:2]

    # Deskew the image so that the slot dividers are horizontal
    _, clean, _, _ = line_masks(frame, boxes)
    skew = estimate_skew(clean)
    skew = 0.0 if abs(skew) < 0.05 else skew
    M, (nw, nh) = rotation_matrix(w, h, skew)
    img = cv2.warpAffine(frame, M, (nw, nh), flags=cv2.INTER_LINEAR)
    valid = cv2.warpAffine(np.ones((h, w), np.uint8), M, (nw, nh), flags=cv2.INTER_NEAREST)
    rc = apply_affine(box_corners(boxes).reshape(-1, 2), M).reshape(-1, 4, 2)
    rboxes = np.concatenate([rc.min(1), rc.max(1)], 1).astype(np.float32) if len(boxes) else boxes

    # Lanes, pitch and slots in the deskewed image
    raw, _, hor, ver = line_masks(img, rboxes)
    runs, hx = find_runs(hor)
    pitch = estimate_pitch(hor, runs)
    lanes = split_lanes(runs, hx, ver, pitch)
    lane_w = float(np.median([b - a for a, b in lanes]))
    val = SlotValidator(img, raw, rboxes[is_vehicle], valid, 1.25 * lane_w)
    val.mark_crossing_vehicles(lanes)

    # Rotate slot polygons back to the original image
    Minv = cv2.invertAffineTransform(M)
    slots = []
    for li, lane in enumerate(lanes):
        for q in lane_slots(hor, lane, pitch, val):
            slots.append({"id": len(slots) + 1, "lane": li, "polygon": apply_affine(q, Minv)})
    return {"frame_size": [w, h], "skew_deg": round(skew, 3), "pitch_px": round(pitch, 3),
            "lanes": len(lanes), "slots": slots}


# Settings that change the extracted slots
SLOT_SETTINGS = ("INPUT_VIDEO", "REF_FRAME", "MODEL_PATH", "VEHICLE_CLASSES", "IMG_SIZE", "SLOT_CONF_THRESH",
                 "MAX_DET", "TOPHAT_KERNEL", "LINE_THRESH", "HOR_KERNEL", "VER_KERNEL", "MAX_SKEW_DEG",
                 "PEAK_HEIGHT", "GAP_TOL", "MAX_MISSING", "MAX_MISSING_COVERED", "COVERED_FRAC", "MIN_GROUP",
                 "MIN_LINE_STRENGTH", "EXTRAPOLATE", "CAR_COVER", "ISLAND_FRAC", "HATCH_FRAC", "PAINT_ROW_FRAC",
                 "PAINT_SLOT_FRAC", "IRREGULAR_MIN", "IRREGULAR_MAX", "EDGE_MIN_VISIBLE")


# Functions and classes whose code changes the extracted slots
SLOT_CODE = ("line_masks", "estimate_skew", "rotation_matrix", "find_runs", "lane_profile", "subpixel_peaks",
             "estimate_pitch", "split_lanes", "SlotValidator", "lane_slots", "extract_slots", "boxes_to_mask")


# Function to fingerprint the slot extraction (its code + settings): saved slots are rebuilt when it changes
def slot_signature():
    with open(os.path.abspath(__file__), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    code = "".join(ast.dump(n) for n in tree.body if getattr(n, "name", None) in SLOT_CODE)
    settings = repr([(k, getattr(config, k, None)) for k in SLOT_SETTINGS])
    return hashlib.sha1((code + settings).encode("utf-8")).hexdigest()[:12]


# Function to tell whether the saved slots must be extracted again
def slots_outdated(file_path):
    if config.REBUILD_SLOTS or not os.path.exists(file_path):
        return True
    try:
        with open(file_path, encoding="utf-8") as f:
            return json.load(f).get("signature") != slot_signature()
    except (OSError, ValueError):
        return True


# Function to save the slots to JSON
def save_slots(result, file_path):
    data = {"video": config.INPUT_VIDEO, "ref_frame": config.REF_FRAME, "signature": slot_signature(),
            **{k: v for k, v in result.items() if k != "slots"},
            "slots": [{"id": s["id"], "lane": s["lane"], "polygon": np.round(s["polygon"], 3).tolist()}
                      for s in result["slots"]]}
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)


# Function to draw the numbered slot map
def draw_slot_map(frame, slots):
    polys = [np.round(np.asarray(s["polygon"]) * 16).astype(np.int32) for s in slots]
    overlay = frame.copy()
    cv2.fillPoly(overlay, polys, COLOR_EMPTY, cv2.LINE_AA, shift=4)
    out = cv2.addWeighted(overlay, config.ALPHA_EMPTY, frame, 1 - config.ALPHA_EMPTY, 0)
    cv2.polylines(out, polys, True, COLOR_EMPTY, 1, cv2.LINE_AA, shift=4)
    for s in slots:
        draw_center_text(out, str(s["id"]), np.asarray(s["polygon"]).mean(0))
    return out


# Function to write small centred text
def draw_center_text(img, text, center, scale=0.3):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
    cv2.putText(img, text, (int(center[0] - tw / 2), int(center[1] + th / 2)), FONT, scale,
                (255, 255, 255), 1, cv2.LINE_AA)


# Class to estimate the reference -> current frame homography
class Stabilizer:
    def __init__(self, ref):
        self.scale = config.STAB_SCALE
        self.S = np.diag([1 / self.scale, 1 / self.scale, 1.0])
        self.ref = self.prepare(ref)
        self.pts = cv2.goodFeaturesToTrack(self.ref, config.STAB_POINTS, 0.01, 8, blockSize=7)
        self.lk = dict(winSize=(21, 21), maxLevel=4,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))
        self.orb = cv2.ORB_create(3000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.kp_ref, self.des_ref = self.orb.detectAndCompute(self.ref, None)
        self.H_s = np.eye(3)   # last homography in scaled coordinates
        self.inliers = 0

    def prepare(self, img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)

    def track_lk(self, cur):
        guess = cv2.perspectiveTransform(self.pts, self.H_s).astype(np.float32)
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(self.ref, cur, self.pts, guess.copy(),
                                              flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **self.lk)
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(cur, self.ref, p1, self.pts.copy(),
                                               flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **self.lk)
        good = (st1.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(p0r - self.pts, axis=2).ravel() < 0.5)
        return self.pts[good].reshape(-1, 2), p1[good].reshape(-1, 2)

    def match_orb(self, cur):
        kp, des = self.orb.detectAndCompute(cur, None)
        if des is None or len(kp) < 50:
            return np.zeros((0, 2)), np.zeros((0, 2))
        ms = self.matcher.match(self.des_ref, des)
        return (np.float32([self.kp_ref[m.queryIdx].pt for m in ms]),
                np.float32([kp[m.trainIdx].pt for m in ms]))

    def __call__(self, frame):
        cur = self.prepare(frame)
        for method in (self.track_lk, self.match_orb):
            p0, p1 = method(cur)
            if len(p0) < 40:
                continue
            H, inl = cv2.findHomography(p0, p1, cv2.RANSAC, 1.0, maxIters=3000, confidence=0.999)
            if H is not None and inl.sum() >= 40:
                self.H_s, self.inliers = H, int(inl.sum())
                break
        return self.S @ self.H_s @ np.linalg.inv(self.S)   # full resolution homography


# Class to compute slot occupancy with temporal smoothing and hysteresis
class OccupancyTracker:
    def __init__(self, slots_ref):
        self.slots = slots_ref.astype(np.float32)
        self.area = np.array([cv2.contourArea(s) for s in self.slots])
        self.lo, self.hi = self.slots.min(1), self.slots.max(1)
        self.score = None
        self.state = np.zeros(len(self.slots), bool)
        self.occupied_frames = np.zeros(len(self.slots), int)

    # Max. (vehicle ∩ slot) / slot area for every slot
    def overlap(self, vehicles):
        ratio = np.zeros(len(self.slots))
        if len(vehicles):
            vlo, vhi = vehicles.min(1), vehicles.max(1)
            hit = ((self.lo[:, None, 0] < vhi[None, :, 0]) & (self.hi[:, None, 0] > vlo[None, :, 0]) &
                   (self.lo[:, None, 1] < vhi[None, :, 1]) & (self.hi[:, None, 1] > vlo[None, :, 1]))
            for i, j in zip(*np.nonzero(hit)):
                inter, _ = cv2.intersectConvexConvex(self.slots[i], vehicles[j])
                ratio[i] = max(ratio[i], inter / self.area[i])
        return ratio

    def update(self, vehicles, crosswise=()):
        raw = self.overlap(vehicles) >= config.OVERLAP_THRESH
        # Trucks / trailers parked across the slots cover only part of each slot they block
        if len(crosswise):
            raw |= self.overlap(crosswise) >= config.CROSSWISE_OVERLAP
        raw = raw.astype(float)
        self.score = raw if self.score is None else config.EMA_ALPHA * raw + (1 - config.EMA_ALPHA) * self.score
        self.state = np.where(self.score >= config.ON_THRESH, True,
                              np.where(self.score <= config.OFF_THRESH, False, self.state))
        self.occupied_frames += self.state
        return self.state


# Function to find how far a bright concrete island continues past the end of a parking row
def island_end(island, a, b, y, step, limit, start_gap):
    end, weak = y, 0
    for k in range(1, limit + 1):
        yy = y + step * k
        if not 0 <= yy < island.shape[0]:
            break
        if island[yy, a:b].mean() >= config.CURB_ISLAND_FRAC:
            end, weak = yy, 0
        else:
            weak += 1
            if weak > (start_gap if end == y else config.CURB_ISLAND_GAP):
                break
    return end


# Function to build the curb mask: islands, medians, loading bands and hatched blocks of the parking rows
def build_curb_mask(img, slots, pitch):
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    island = cv2.morphologyEx(((hsv[..., 2] > 160) & (hsv[..., 1] < 100)).astype(np.uint8),
                              cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

    # Parking rows: lanes closer than CURB_LANE_GAP form one row (the gap between them is a median)
    lanes = {}
    for s in slots:
        lanes.setdefault(s["lane"], []).append(np.asarray(s["polygon"]))
    spans = sorted((p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max())
                   for p in (np.concatenate(v) for v in lanes.values()))
    rows = []
    for x0, x1, y0, y1 in spans:
        if rows and x0 - rows[-1][1] < config.CURB_LANE_GAP * pitch:
            rows[-1] = [rows[-1][0], max(rows[-1][1], x1), min(rows[-1][2], y0), max(rows[-1][3], y1)]
        else:
            rows.append([x0, x1, y0, y1])

    # Whole row area plus the islands at both ends, minus the parking slots
    mask = np.zeros((h, w), np.uint8)
    limit = int(config.CURB_MAX_EXTENT * pitch)
    start_gap = int(config.CURB_ISLAND_START * pitch)
    for x0, x1, y0, y1 in rows:
        a, b = max(0, int(np.floor(x0))), min(w, int(np.ceil(x1)) + 1)
        top = island_end(island, a, b, int(np.floor(y0)), -1, limit, start_gap)
        bottom = island_end(island, a, b, int(np.ceil(y1)), 1, limit, start_gap)
        mask[max(0, top):min(h, bottom + 1), a:b] = 1
    cv2.fillPoly(mask, [np.round(np.asarray(s["polygon"])).astype(np.int32) for s in slots], 0)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((config.CURB_MIN_WIDTH, config.CURB_MIN_WIDTH), np.uint8))


# Function to compute the IoU matrix of two sets of axis-aligned boxes (x1, y1, x2, y2)
def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


# Function to solve an assignment on a cost matrix, keeping only pairs that pass the validity mask
def assign(cost, valid):
    if cost.size == 0 or not valid.any():
        return []
    rows, cols = linear_sum_assignment(np.where(valid, cost, 1e6))
    return [(r, c) for r, c in zip(rows, cols) if valid[r, c]]


# Class to track one vehicle with a constant-velocity Kalman filter on its box centre
class Track:
    F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    Hm = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
    Q = config.KF_PROCESS_NOISE * np.array([[0.25, 0, 0.5, 0], [0, 0.25, 0, 0.5],
                                            [0.5, 0, 1, 0], [0, 0.5, 0, 1]], float)
    R = np.eye(2) * config.KF_MEASURE_NOISE ** 2

    def __init__(self, tid, box, cls_name, frame_idx):
        center = (box[:2] + box[2:]) / 2
        self.id = tid
        self.x = np.array([center[0], center[1], 0.0, 0.0])
        self.P = np.diag([4.0, 4.0, 4.0, 4.0])
        self.size = box[2:] - box[:2]
        self.cls = cls_name
        self.age = 1
        self.missed = 0
        self.history = [tuple(center)]
        self.raw = [tuple(center)]   # detection centres (heading / turn-rate fit)
        self.obs = [(frame_idx, center.copy())]     # recent real observations (OCM direction)
        self.snapshot = (self.x.copy(), self.P.copy())   # filter state at the last observation
        self.fit_speed = 0.0
        self.moving = False
        self.ever_moving = False
        self.heading = None          # radians, image axes (y down)
        self.yaw_rate = 0.0          # radians / frame
        self.target_slot = None
        self.first_frame = self.last_frame = frame_idx
        self.moving_frames = 0
        self.distance_px = 0.0
        self.speeds = []

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def kf_update(self, center):
        y = np.asarray(center) - self.Hm @ self.x
        S = self.Hm @ self.P @ self.Hm.T + self.R
        K = self.P @ self.Hm.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.Hm) @ self.P

    def add_point(self, center):
        self.history.append((self.x[0], self.x[1]))
        self.history = self.history[-max(config.TRAIL_LENGTH, config.MOVE_WINDOW + 1):]
        self.raw.append(tuple(center))
        self.raw = self.raw[-config.HEADING_WINDOW:]

    def update(self, box, cls_name, frame_idx):
        center = (box[:2] + box[2:]) / 2
        gap = frame_idx - self.last_frame
        if gap > 1:
            # ORU: rewind to the last observation and re-run the filter over virtual observations
            last = self.obs[-1][1]
            self.x, self.P = self.snapshot[0].copy(), self.snapshot[1].copy()
            for i in range(1, gap):
                virtual = last + (center - last) * i / gap
                self.predict()
                self.kf_update(virtual)
                self.add_point(virtual)
            self.predict()
        self.kf_update(center)
        self.add_point(center)
        a = config.BOX_SIZE_ALPHA
        self.size = a * (box[2:] - box[:2]) + (1 - a) * self.size
        self.cls = cls_name
        self.age += 1
        self.missed = 0
        self.last_frame = frame_idx
        self.obs.append((frame_idx, center.copy()))
        self.obs = self.obs[-(config.OCM_DELTA_T + 2):]
        self.snapshot = (self.x.copy(), self.P.copy())

    @property
    def pos(self):
        return self.x[:2]

    @property
    def box(self):
        return np.concatenate([self.pos - self.size / 2, self.pos + self.size / 2])

    @property
    def last_box(self):
        c = self.obs[-1][1]
        return np.concatenate([c - self.size / 2, c + self.size / 2])

    @property
    def obs_direction(self):
        f_last, c_last = self.obs[-1]
        older = [c for f, c in self.obs if f <= f_last - config.OCM_DELTA_T]
        if not older:
            return None
        d = c_last - older[-1]
        n = np.hypot(*d)
        return d / n if n > 1.0 else None

    @property
    def speed(self):
        return float(np.hypot(self.x[2], self.x[3]))

    # Moving / parked decision with hysteresis and a displacement check
    def update_motion(self):
        h = self.history
        disp = np.hypot(h[-1][0] - h[0][0], h[-1][1] - h[0][1]) if len(h) > config.MOVE_WINDOW else 0.0
        if not self.moving:
            self.moving = (self.age >= config.MIN_TRACK_AGE and self.speed > config.MOVE_ON_SPEED
                           and disp > config.MOVE_MIN_DISPLACEMENT)
        elif self.speed < config.MOVE_OFF_SPEED:
            self.moving = False
        if not self.moving:
            self.yaw_rate, self.target_slot, self.heading = 0.0, None, None
            return
        self.ever_moving = True
        self.moving_frames += 1
        self.speeds.append(self.speed)
        if len(h) >= 2:
            self.distance_px += np.hypot(h[-1][0] - h[-2][0], h[-1][1] - h[-2][1])
        # Heading and turn rate from a quadratic fit of the last raw positions
        W = config.HEADING_WINDOW
        if len(self.raw) < W:
            return
        tt = np.arange(W, dtype=float) - (W - 1)          # 0 = current frame
        pts = np.asarray(self.raw[-W:])
        ax, bx, _ = np.polyfit(tt, pts[:, 0], 2)
        ay, by, _ = np.polyfit(tt, pts[:, 1], 2)
        vx, vy, acx, acy = bx, by, 2 * ax, 2 * ay           # velocity and acceleration at t = 0
        v2 = vx * vx + vy * vy
        if v2 < 1e-6:
            return
        max_rate = np.radians(config.MAX_YAW_RATE_DEG) / fps
        rate = float(np.clip((vx * acy - vy * acx) / v2, -max_rate, max_rate))
        if abs(rate) < np.radians(config.YAW_DEADBAND_DEG) / fps:   # ignore jitter on straight driving
            rate = 0.0
        self.yaw_rate = rate if self.heading is None else (
            config.YAW_RATE_ALPHA * rate + (1 - config.YAW_RATE_ALPHA) * self.yaw_rate)
        self.heading = float(np.arctan2(vy, vx))
        self.fit_speed = float(np.sqrt(v2))

    # Predicted future path (CTRV, turn rate decays so the path does not curl up)
    def predict_path(self, steps):
        x, y = self.pos
        v, th, w = (self.fit_speed or self.speed), self.heading, self.yaw_rate
        pts = [(x, y)]
        for _ in range(steps):
            th += w
            w *= config.YAW_DECAY
            x += v * np.cos(th)
            y += v * np.sin(th)
            pts.append((x, y))
        return np.array(pts)


# Class to match detections to vehicle tracks (OC-SORT + ByteTrack two-stage matching)
class Tracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 1

    def update(self, boxes, scores, classes, frame_idx):
        for t in self.tracks:
            t.predict()
        n = len(boxes)
        det_track = [None] * n
        high = np.nonzero(scores >= config.TRACK_HIGH_THRESH)[0]
        low = np.nonzero(scores < config.TRACK_HIGH_THRESH)[0]
        free_t = list(range(len(self.tracks)))
        free_h = list(high)

        def commit(pairs, t_idx, d_idx):
            used_t, used_d = set(), set()
            for r, c in pairs:
                t, d = self.tracks[t_idx[r]], d_idx[c]
                t.update(boxes[d], classes[d], frame_idx)
                det_track[d] = t
                used_t.add(t_idx[r])
                used_d.add(d)
            return [i for i in t_idx if i not in used_t], [d for d in d_idx if d not in used_d]

        # Stage 1: high detections, IoU + direction consistency
        if free_t and free_h:
            T = [self.tracks[i] for i in free_t]
            iou = iou_matrix(np.array([t.box for t in T]), boxes[free_h])
            cost = 1.0 - iou
            centers = (boxes[free_h, :2] + boxes[free_h, 2:]) / 2
            for r, t in enumerate(T):
                d = t.obs_direction
                if d is None:
                    continue
                v = centers - t.obs[-1][1]
                norm = np.linalg.norm(v, axis=1)
                ok = norm > 1.0
                cos = np.where(ok, (v @ d) / np.maximum(norm, 1e-9), 1.0)
                cost[r] += config.OCM_WEIGHT * np.arccos(np.clip(cos, -1, 1)) / np.pi
            free_t, free_h = commit(assign(cost, iou >= config.MATCH_IOU_HIGH), free_t, free_h)

        # Stage 2: low detections (ByteTrack), IoU only
        if free_t and len(low):
            T = [self.tracks[i] for i in free_t]
            iou = iou_matrix(np.array([t.box for t in T]), boxes[low])
            free_t, _ = commit(assign(1.0 - iou, iou >= config.MATCH_IOU_LOW), free_t, list(low))

        # Stage 3: OC-SORT recovery with the last observed boxes
        if free_t and free_h:
            T = [self.tracks[i] for i in free_t]
            iou = iou_matrix(np.array([t.last_box for t in T]), boxes[free_h])
            free_t, free_h = commit(assign(1.0 - iou, iou >= config.MATCH_IOU_HIGH), free_t, free_h)

        for i in free_t:
            self.tracks[i].missed = frame_idx - self.tracks[i].last_frame

        # Track-aware initialization: skip detections overlapping a track or a stronger new detection
        existing = np.array([t.box for t in self.tracks]) if self.tracks else np.zeros((0, 4))
        accepted = []
        for d in sorted(free_h, key=lambda d: -scores[d]):
            if len(existing) and iou_matrix(boxes[d][None], existing).max() >= config.NEW_TRACK_IOU:
                continue
            if accepted and iou_matrix(boxes[d][None], boxes[accepted]).max() >= config.NEW_TRACK_IOU:
                continue
            accepted.append(d)
            t = Track(self.next_id, boxes[d], classes[d], frame_idx)
            self.next_id += 1
            self.tracks.append(t)
            det_track[d] = t

        self.tracks = [t for t in self.tracks if t.missed <= config.MAX_MISSED]
        for t in self.tracks:
            if t.missed == 0:
                t.update_motion()
        return det_track


# Function to pick the first empty slot the predicted path drives into
def find_target_slot(path, slot_centers, empty, max_dist):
    if not empty.any():
        return None
    idx = np.nonzero(empty)[0]
    d = np.linalg.norm(slot_centers[idx][:, None, :] - path[None, 1:, :], axis=2)   # slots x steps
    near = d.min(1) <= max_dist
    if not near.any():
        return None
    first_step = np.where(near, d.argmin(1), np.inf)
    return int(idx[int(np.argmin(first_step))])


# Function to blend filled polygons with transparency
def blend_polygons(img, polys, color, alpha):
    if not polys:
        return img
    overlay = img.copy()
    cv2.fillPoly(overlay, polys, color, cv2.LINE_AA, shift=4)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


# Function to draw the slots as transparent zones: occupied = red, empty = green (borders optional)
def draw_occupancy(frame, slots_img, state, ids, boxes):
    P = [np.round(s * 16).astype(np.int32) for s in slots_img]   # 4-bit sub-pixel precision
    occupied = [p for p, s in zip(P, state) if s]
    empty = [p for p, s in zip(P, state) if not s]
    out = blend_polygons(frame, occupied, COLOR_OCCUPIED, config.ALPHA_OCCUPIED)
    out = blend_polygons(out, empty, COLOR_EMPTY, config.ALPHA_EMPTY)
    if config.SHOW_SLOT_BORDERS:
        # Hairline borders: 1 px anti-aliased lines blended with BORDER_ALPHA
        lines = out.copy()
        cv2.polylines(lines, empty, True, COLOR_EMPTY, 1, cv2.LINE_AA, shift=4)
        cv2.polylines(lines, occupied, True, COLOR_OCCUPIED, 1, cv2.LINE_AA, shift=4)
        out = cv2.addWeighted(lines, config.BORDER_ALPHA, out, 1 - config.BORDER_ALPHA, 0)
    if config.SHOW_BOXES:
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 200, 0), 1, cv2.LINE_AA)
    if config.SHOW_SLOT_IDS:
        for s, i in zip(slots_img, ids):
            draw_center_text(out, str(i), s.mean(0))
    return out


# Function to draw the statistics panel
def draw_panel(img, total, occupied):
    empty = total - occupied
    pct = 100 * occupied / max(1, total)
    x0, y0, w, h = 16, 16, 270, 150
    roi = img[y0:y0 + h, x0:x0 + w]
    img[y0:y0 + h, x0:x0 + w] = cv2.addWeighted(roi, 0.35, np.full_like(roi, 20), 0.65, 0)
    cv2.putText(img, "PARKING OCCUPANCY", (x0 + 14, y0 + 30), FONT_BOLD, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(img, (x0 + 22, y0 + 56), 6, COLOR_OCCUPIED, -1, cv2.LINE_AA)
    cv2.putText(img, f"Occupied: {occupied}", (x0 + 36, y0 + 62), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(img, (x0 + 22, y0 + 82), 6, COLOR_EMPTY, -1, cv2.LINE_AA)
    cv2.putText(img, f"Empty: {empty}", (x0 + 36, y0 + 88), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"Total: {total}   Rate: {pct:.1f}%", (x0 + 14, y0 + 114), FONT, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    bx, by, bw = x0 + 14, y0 + h - 22, w - 28
    cv2.rectangle(img, (bx, by), (bx + bw, by + 8), (70, 70, 70), -1)
    cv2.rectangle(img, (bx, by), (bx + int(bw * pct / 100), by + 8), COLOR_OCCUPIED, -1)
    return img


# Function to resample a polyline at a fixed arc-length spacing
def resample(path, spacing):
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < spacing:
        return path[:1]
    t = np.arange(0, s[-1], spacing)
    return np.stack([np.interp(t, s, path[:, 0]), np.interp(t, s, path[:, 1])], 1)


# Function to draw trail, dotted predicted path, arrowhead and target slot of each moving vehicle
def draw_motion(img, moving, H, slots_img):
    overlay = img.copy()
    for t in moving:
        color = config.TRACK_COLOR

        # Past trail (solid, thin)
        trail = apply_homography(np.array(t.history[-config.TRAIL_LENGTH:]), H)
        if len(trail) >= 2:
            cv2.polylines(overlay, [np.round(trail * 16).astype(np.int32)], False, color, 1, cv2.LINE_AA, shift=4)

        # Predicted path (dotted, dots shrink with prediction time) + arrowhead
        path = apply_homography(t.predict_path(STEPS), H)
        dots = resample(path, config.DOT_SPACING)[1:]
        for k, (x, y) in enumerate(dots):
            r = 3.2 - 1.8 * k / max(1, len(dots))
            cv2.circle(overlay, (int(round(x * 16)), int(round(y * 16))), int(r * 16), color, -1, cv2.LINE_AA, shift=4)
        if len(path) >= 2:
            tip, back = path[-1], path[-2]
            d = tip - back
            n = np.hypot(*d)
            if n > 1e-6:
                d /= n
                p = np.array([-d[1], d[0]])
                tri = np.array([tip + 6 * d, tip - 6 * d + 5 * p, tip - 6 * d - 5 * p])
                cv2.fillPoly(overlay, [np.round(tri * 16).astype(np.int32)], color, cv2.LINE_AA, shift=4)

        # Likely target slot
        if t.target_slot is not None:
            poly = np.round(slots_img[t.target_slot] * 16).astype(np.int32)
            cv2.polylines(overlay, [poly], True, color, 2, cv2.LINE_AA, shift=4)

        # Vehicle marker
        c = apply_homography(t.pos[None], H)[0]
        cv2.circle(overlay, (int(round(c[0] * 16)), int(round(c[1] * 16))), 4 * 16, color, 2, cv2.LINE_AA, shift=4)
    return cv2.addWeighted(overlay, 0.92, img, 0.08, 0)


# Class to write the output video (H.264 through FFmpeg when available, else OpenCV mp4v)
class VideoWriter:
    def __init__(self, file_path, w, h, fps):
        self.proc = self.cv = None
        if shutil.which("ffmpeg"):
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                   "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-preset", "medium",
                   "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", file_path]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            self.cv = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def write(self, frame):
        if self.proc:
            self.proc.stdin.write(frame.tobytes())
        else:
            self.cv.write(frame)

    def release(self):
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()
        else:
            self.cv.release()


# Load YOLO model and select device
device = pick_device()
model = YOLO(path(config.MODEL_PATH))
vehicle_class_ids = [i for i, n in model.names.items() if n in config.VEHICLE_CLASSES]
crosswise_class_ids = [i for i, n in model.names.items() if n in config.CROSSWISE_CLASSES]

# Reference frame for slot extraction and stabilization
video_path = path(config.INPUT_VIDEO)
ref = read_frame(video_path, config.REF_FRAME)

# Extract parking slots (or load them if they were extracted before)
if slots_outdated(path(config.SLOTS_JSON)):
    print("Extracting parking slots from the reference frame...")
    boxes, is_vehicle = detect(model, ref, config.SLOT_CONF_THRESH, device)
    result = extract_slots(ref, boxes, is_vehicle)
    save_slots(result, path(config.SLOTS_JSON))
    cv2.imwrite(path(config.SLOTS_PREVIEW), draw_slot_map(ref, result["slots"]))
    print(f"  skew {result['skew_deg']} deg | pitch {result['pitch_px']} px | "
          f"{result['lanes']} lanes | {len(result['slots'])} slots")

with open(path(config.SLOTS_JSON), encoding="utf-8") as f:
    slot_data = json.load(f)
slot_ids = [s["id"] for s in slot_data["slots"]]
slots_ref = np.array([s["polygon"] for s in slot_data["slots"]], np.float64)
print(f"{len(slots_ref)} parking slots loaded")

# Slot geometry used by the trajectory part (parking area test, target slot, metric scale)
slot_centers = slots_ref.mean(1)
slot_polys = slots_ref.astype(np.float32)
slot_lo, slot_hi = slots_ref.min(1), slots_ref.max(1)
pitch = float(slot_data.get("pitch_px") or 31.0)
m_per_px = config.SLOT_WIDTH_M / pitch


# Function to check whether a point (reference coordinates) lies inside any parking slot
def in_parking_area(p):
    cand = np.nonzero(np.all((slot_lo <= p) & (p <= slot_hi), axis=1))[0]
    return any(cv2.pointPolygonTest(slot_polys[i], (float(p[0]), float(p[1])), False) >= 0 for i in cand)


# Curb zones of the lot (no vehicle is ever detected there)
curb_mask = build_curb_mask(ref, slot_data["slots"], pitch)

# Initialize stabilization, occupancy tracking and vehicle tracking
stabilizer = Stabilizer(ref)
tracker = OccupancyTracker(slots_ref)
vehicle_tracker = Tracker()

# Open video (paths loaded from config.py)
video_capture = cv2.VideoCapture(video_path)

# Video features
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video_capture.get(cv2.CAP_PROP_FPS) or 30
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
STEPS = max(1, int(round(config.PREDICT_SECONDS * fps)))

# Create the VideoWriter object to save the video file
output_video = VideoWriter(path(config.OUTPUT_VIDEO), frame_width, frame_height, fps)

# Create the CSV files for the per-frame statistics and the vehicle tracks
csv_file = open(path(config.TIMESERIES_CSV), "w", newline="")
log = csv.writer(csv_file)
log.writerow(["frame", "time_sec", "total", "occupied", "empty", "occupancy_pct",
              "shift_x", "shift_y", "rotation_deg", "scale", "stab_inliers", "vehicles", "moving_vehicles"])
tracks_file = open(path(config.TRACKS_CSV), "w", newline="")
track_log = csv.writer(tracks_file)
track_log.writerow(["frame", "time_sec", "track_id", "class", "x_ref", "y_ref", "speed_kmh", "heading_deg",
                    "turn_rate_deg_s", "pred_x_ref", "pred_y_ref", "target_slot"])

occupancy_history = []
shown = {}
start, frame_idx = time.time(), 0
while True:
    ret, frame = video_capture.read()  # read video frame
    if not ret:
        break

    # Reference -> current frame homography
    H = stabilizer(frame)
    H_inv = np.linalg.inv(H)

    # Detect vehicles and map them into reference coordinates
    results = model.predict(frame, imgsz=config.IMG_SIZE, conf=min(config.CONF_THRESH, config.TRACK_LOW_THRESH),
                            classes=sorted(set(vehicle_class_ids) | set(crosswise_class_ids)),
                            max_det=config.MAX_DET, device=device, verbose=False)[0]
    all_boxes = results.boxes.xyxy.cpu().numpy()
    all_scores = results.boxes.conf.cpu().numpy()
    all_cls = results.boxes.cls.cpu().numpy().astype(int)
    all_corners = apply_homography(box_corners(all_boxes), H_inv).astype(np.float32)

    # Drop detections on curb zones (islands, medians, loading bands): they are never vehicles
    if config.CURB_EXCLUDE and len(all_boxes):
        cx, cy = np.round(all_corners.mean(1)).astype(int).T
        inside = (cx >= 0) & (cx < curb_mask.shape[1]) & (cy >= 0) & (cy < curb_mask.shape[0])
        on_curb = np.zeros(len(all_boxes), bool)
        on_curb[inside] = curb_mask[cy[inside], cx[inside]] > 0
        all_boxes, all_scores, all_cls, all_corners = (all_boxes[~on_curb], all_scores[~on_curb],
                                                       all_cls[~on_curb], all_corners[~on_curb])

    # Keep the occupancy detections (confidence >= CONF_THRESH)
    sel = all_scores >= config.CONF_THRESH
    boxes, cls, corners_ref = all_boxes[sel], all_cls[sel], all_corners[sel]

    # Large vehicles parked across the slots: long side perpendicular to the slots (tall boxes)
    size = corners_ref.max(1) - corners_ref.min(1) if len(boxes) else np.zeros((0, 2))
    crosswise = np.isin(cls, crosswise_class_ids) & (size[:, 1] > config.CROSSWISE_ASPECT * size[:, 0])
    is_vehicle = np.isin(cls, vehicle_class_ids)
    boxes = boxes[is_vehicle | crosswise]

    # Update slot occupancy
    state = tracker.update(corners_ref[is_vehicle], corners_ref[crosswise])
    occupied = int(state.sum())
    occupancy_history.append(occupied)

    # Track vehicles and predict the path of the moving ones outside the parking slots
    keep = np.nonzero(np.isin(all_cls, vehicle_class_ids))[0]
    boxes_ref = (np.concatenate([all_corners[keep].min(1), all_corners[keep].max(1)], 1).astype(np.float64)
                 if len(keep) else np.zeros((0, 4)))
    vehicle_tracker.update(boxes_ref, all_scores[keep], [model.names[c] for c in all_cls[keep]], frame_idx)
    moving = [t for t in vehicle_tracker.tracks if t.moving and t.missed <= 2 and t.heading is not None
              and not in_parking_area(t.pos)]
    for t in moving:
        shown[t.id] = t
        path_pred = t.predict_path(STEPS)
        t.target_slot = (find_target_slot(path_pred, slot_centers, ~state, config.TARGET_SLOT_DIST * pitch)
                         if config.PREDICT_TARGET_SLOT else None)
        track_log.writerow([frame_idx, f"{frame_idx / fps:.3f}", t.id, t.cls, f"{t.pos[0]:.2f}", f"{t.pos[1]:.2f}",
                            f"{t.speed * fps * m_per_px * 3.6:.2f}", f"{np.degrees(t.heading):.1f}",
                            f"{np.degrees(t.yaw_rate) * fps:.2f}", f"{path_pred[-1, 0]:.2f}",
                            f"{path_pred[-1, 1]:.2f}", slot_ids[t.target_slot] if t.target_slot is not None else ""])

    # Draw slots (moved into the current frame), trajectories and the statistics panel
    slots_img = apply_homography(slots_ref, H)
    annotated_frame = draw_occupancy(frame, slots_img, state, slot_ids, boxes)
    annotated_frame = draw_motion(annotated_frame, moving, H, slots_img)
    annotated_frame = draw_panel(annotated_frame, len(state), occupied)

    # Write the drawn frame to the video file to be saved
    output_video.write(annotated_frame)

    # Log per-frame statistics (camera motion measured at the image centre)
    c = apply_homography(np.array([[frame_width / 2, frame_height / 2], [frame_width / 2 + 100, frame_height / 2]]), H)
    d = c[1] - c[0]
    log.writerow([frame_idx, f"{frame_idx / fps:.3f}", len(state), occupied, len(state) - occupied,
                  f"{100 * occupied / max(1, len(state)):.2f}", f"{c[0, 0] - frame_width / 2:.3f}",
                  f"{c[0, 1] - frame_height / 2:.3f}", f"{np.degrees(np.arctan2(d[1], d[0])):.4f}",
                  f"{np.hypot(*d) / 100:.5f}", stabilizer.inliers, len(boxes), len(moving)])

    frame_idx += 1
    if frame_idx % 30 == 0:
        print(f"  {frame_idx}/{total_frames} frames | {frame_idx / (time.time() - start):.1f} FPS | "
              f"occupied {occupied} / empty {len(state) - occupied} | moving {len(moving)}")

    # Show video with parking occupancy and vehicle trajectories
    if config.SHOW_WINDOW:
        cv2.imshow("Parking Lot Occupancy and Vehicle Trajectory Prediction",
                   cv2.resize(annotated_frame, (frame_width * 2 // 3, frame_height * 2 // 3)))

        # Switch off video when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release all open windows
video_capture.release()
output_video.release()
csv_file.close()
tracks_file.close()
cv2.destroyAllWindows()

# Save the summary report
elapsed = time.time() - start
hist = np.array(occupancy_history) if occupancy_history else np.zeros(1)
movers = sorted(shown.values(), key=lambda t: t.id)
kmh = lambda v: round(float(v) * fps * m_per_px * 3.6, 1)
summary = {
    "video": config.INPUT_VIDEO,
    "resolution": f"{frame_width}x{frame_height}",
    "fps": round(fps, 2),
    "processed_frames": frame_idx,
    "analysis_duration_sec": round(elapsed, 2),
    "model": config.MODEL_PATH,
    "img_size": config.IMG_SIZE,
    "conf": config.CONF_THRESH,
    "overlap_thresh": config.OVERLAP_THRESH,
    "lot": {"skew_deg": slot_data.get("skew_deg"), "pitch_px": slot_data.get("pitch_px"),
            "lanes": slot_data.get("lanes"), "total_slots": len(slot_ids)},
    "occupancy": {
        "final_occupied": int(hist[-1]),
        "final_empty": len(slot_ids) - int(hist[-1]),
        "mean_occupied": round(float(hist.mean()), 2),
        "min_occupied": int(hist.min()),
        "max_occupied": int(hist.max()),
        "mean_occupancy_pct": round(100 * float(hist.mean()) / max(1, len(slot_ids)), 2),
    },
    "slots": {str(i): {"occupied_ratio": round(n / max(1, frame_idx), 3), "final_state":
                       "occupied" if s else "empty"}
              for i, n, s in zip(slot_ids, tracker.occupied_frames, tracker.state)},
    "trajectories": {
        "prediction_horizon_sec": config.PREDICT_SECONDS,
        "meters_per_pixel": round(m_per_px, 5),
        "vehicles_with_trajectory": len(movers),
        "vehicles": {str(t.id): {
            "class": t.cls,
            "first_frame": t.first_frame,
            "last_frame": t.last_frame,
            "moving_sec": round(t.moving_frames / fps, 2),
            "distance_m": round(t.distance_px * m_per_px, 1),
            "mean_speed_kmh": kmh(np.mean(t.speeds)) if t.speeds else 0.0,
            "max_speed_kmh": kmh(np.max(t.speeds)) if t.speeds else 0.0,
        } for t in movers},
    },
}
with open(path(config.SUMMARY_JSON), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"\nDone: {frame_idx} frames in {elapsed:.1f} s | {len(movers)} vehicles with trajectory")
print(f"Video   : {config.OUTPUT_VIDEO}")
print(f"CSV     : {config.TIMESERIES_CSV}")
print(f"Tracks  : {config.TRACKS_CSV}")
print(f"Summary : {config.SUMMARY_JSON}")
