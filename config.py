# Parking Lot Occupancy and Vehicle Trajectory Prediction System — Configuration

import os

# Model and class configuration (VisDrone-trained YOLO detector)
MODEL_PATH = "visdrone.pt"
VEHICLE_CLASSES = ("car", "van", "truck")   # VisDrone classes counted as vehicles

# Detection parameters
IMG_SIZE = 1280              # inference size (aerial cars are ~58x26 px at 1080p)
CONF_THRESH = 0.30           # detection confidence threshold (occupancy)
SLOT_CONF_THRESH = 0.25      # detection confidence threshold (slot extraction)
MAX_DET = 2000               # a crowded lot exceeds the default limit of 300
DEVICE = None                # None = auto (cuda > mps > cpu), or "cpu", "mps", "cuda:0"

# Parking slot extraction (reference frame)
REF_FRAME = 0                # frame used for slot extraction and as stabilization reference
REBUILD_SLOTS = False        # True = always extract the slots again
TOPHAT_KERNEL = 9            # top-hat kernel, wider than the painted lines (px)
LINE_THRESH = 40             # top-hat threshold for painted line pixels
HOR_KERNEL = 15              # min. length of a horizontal divider segment (px)
VER_KERNEL = 31              # min. length of a vertical centre-line segment (px)
MAX_SKEW_DEG = 15.0          # max. lot rotation searched by the deskew step (degrees)
PEAK_HEIGHT = 0.06           # min. visible fraction of a divider across the lane width
GAP_TOL = 0.22               # tolerance of a divider gap to a multiple of the pitch
MAX_MISSING = 3              # max. consecutive dividers hidden by vehicles
MAX_MISSING_COVERED = 6      # max. hidden dividers under large vehicles (trucks, trailers)
COVERED_FRAC = 0.4           # vehicle-covered gap ratio that allows MAX_MISSING_COVERED
MIN_GROUP = 3                # min. dividers in a valid slot row
MIN_LINE_STRENGTH = 0.6      # min. median visibility of real dividers in a slot row
EXTRAPOLATE = 2              # max. slots added at each row end (next to islands)
CAR_COVER = 0.4              # vehicle overlap that marks a candidate slot as occupied
ISLAND_FRAC = 0.25           # island (bright concrete) ratio that rejects a slot
HATCH_FRAC = 0.12            # hatched (yellow striped) ratio that rejects a slot
PAINT_ROW_FRAC = 0.20        # yellow paint ratio that marks a lane row as no-parking
PAINT_SLOT_FRAC = 0.4        # painted row ratio that rejects a slot
IRREGULAR_MIN = 0.85         # min. length of a slot between painted zones (x pitch)
IRREGULAR_MAX = 1.45         # max. length of a slot between painted zones (x pitch)
EDGE_MIN_VISIBLE = 0.6       # min. visible part of a slot cut by the frame border

# Occupancy decision
OVERLAP_THRESH = 0.30        # vehicle area / slot area needed to call a slot occupied
EMA_ALPHA = 0.35             # temporal smoothing factor of the occupancy score
ON_THRESH = 0.60             # smoothed score above which a slot becomes occupied
OFF_THRESH = 0.35            # smoothed score below which a slot becomes empty
CROSSWISE_CLASSES = ("truck", "bus")   # large vehicles parked across the slots (trailers = "bus")
CROSSWISE_ASPECT = 1.5       # box height / width above which a vehicle lies across the slots
CROSSWISE_OVERLAP = 0.20     # overlap with a crosswise vehicle that marks a slot occupied

# Curb zones (islands, medians, loading bands and hatched blocks in the parking rows): never a vehicle
CURB_EXCLUDE = True          # drop every detection whose centre lies on a curb zone
CURB_LANE_GAP = 1.0          # lanes closer than this (x pitch) form one row, the gap is a median
CURB_ISLAND_FRAC = 0.3       # bright concrete ratio of a row that extends an island past the row end
CURB_ISLAND_GAP = 6          # weaker rows allowed inside an island (lamp posts, cracks)
CURB_ISLAND_START = 1.0      # weaker rows allowed before an island starts (x pitch, planters)
CURB_MAX_EXTENT = 5.0        # max. island length past the row end (x pitch)
CURB_MIN_WIDTH = 15          # thinner strips are not curbs (px), keeps row edges free

# Stabilization (drone drift + perspective change)
STAB_SCALE = 0.5             # image scale used for optical flow tracking
STAB_POINTS = 2000           # number of reference corner points tracked

# Vehicle tracking (OC-SORT + ByteTrack two-stage matching)
TRACK_HIGH_THRESH = 0.25     # min. confidence to start or first-stage match a track
TRACK_LOW_THRESH = 0.10      # min. confidence for the second matching stage
MATCH_IOU_HIGH = 0.20        # min. IoU for high-confidence matches and recovery
MATCH_IOU_LOW = 0.40         # min. IoU for low-confidence matches
OCM_WEIGHT = 0.20            # weight of the direction consistency cost
OCM_DELTA_T = 3              # frames between the observations used for the direction
NEW_TRACK_IOU = 0.50         # no new track from a box overlapping an existing track this much
BOX_SIZE_ALPHA = 0.5         # smoothing factor of the tracked box size
MAX_MISSED = 12              # frames a track survives without a detection
KF_PROCESS_NOISE = 0.05      # Kalman acceleration noise (px / frame^2)
KF_MEASURE_NOISE = 1.5       # Kalman measurement noise (px)

# Moving / parked decision
MIN_TRACK_AGE = 8            # frames before a track can be called moving
MOVE_ON_SPEED = 0.50         # px / frame above which a track becomes moving (~4 km/h)
MOVE_OFF_SPEED = 0.25        # px / frame below which a moving track becomes parked
MOVE_WINDOW = 15             # frames used for the displacement check
MOVE_MIN_DISPLACEMENT = 5.0  # px a car must travel within MOVE_WINDOW to count as moving

# Trajectory prediction (vehicles moving outside the slots)
PREDICT_SECONDS = 2.0        # how far ahead the dotted path is predicted
HEADING_WINDOW = 30          # past positions fitted for heading and turn rate
YAW_RATE_ALPHA = 0.3         # smoothing factor of the turn rate
YAW_DEADBAND_DEG = 10.0      # turn rates below this count as straight driving (deg / s)
MAX_YAW_RATE_DEG = 90.0      # max. turn rate used for prediction (deg / s)
YAW_DECAY = 0.97             # per-frame decay of the turn rate along the prediction
PREDICT_TARGET_SLOT = True   # outline the empty slot a predicted path drives into
TARGET_SLOT_DIST = 0.6       # max. distance path -> slot centre (x pitch)
SLOT_WIDTH_M = 2.5           # real width of one parking slot (m), used for km/h

# Visualization
COLOR_OCCUPIED = (40, 40, 235)   # BGR red
COLOR_EMPTY = (60, 220, 60)      # BGR green
ALPHA_OCCUPIED = 0.40            # transparency of the red fill
ALPHA_EMPTY = 0.22               # transparency of the green fill
SHOW_SLOT_BORDERS = True         # draw thin slot border lines
BORDER_ALPHA = 0.5               # opacity of the 1 px border lines
SHOW_SLOT_IDS = False            # draw slot numbers
SHOW_BOXES = False               # draw vehicle detection boxes
TRACK_COLOR = (255, 80, 0)       # BGR blue (trajectories)
TRAIL_LENGTH = 45                # past positions drawn as a solid trail (frames)
DOT_SPACING = 9.0                # distance between the dots of the predicted path (px)
SHOW_WINDOW = os.environ.get("SHOW_WINDOW", "1") == "1"   # live preview (press q to quit); 0 in Docker

# Video and output files
INPUT_VIDEO = "Aerial-Parking-Lot.mp4"                                        # input video file name
OUTPUT_VIDEO = "Parking-Lot-Occupancy-and-Vehicle-Trajectory-Prediction.mp4"  # output video file name
SLOTS_JSON = "Parking-Lot-Slots.json"                                         # parking slot polygons
SLOTS_PREVIEW = "Parking-Lot-Slots.png"                                       # numbered slot map image
TIMESERIES_CSV = "Parking-Lot-Occupancy-timeseries.csv"                       # per-frame occupancy statistics
TRACKS_CSV = "Parking-Lot-Vehicle-Tracks.csv"                                 # per-frame vehicle trajectories
SUMMARY_JSON = "Parking-Lot-Occupancy-summary.json"                           # overall, per-slot and per-vehicle summary
