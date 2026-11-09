# Parking Lot Occupancy and Vehicle Trajectory Prediction System - Docker image
FROM python:3.11-slim

# System libraries required by OpenCV and video I/O (ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and configuration
COPY Parking-Lot-Occupancy-and-Vehicle-Trajectory-Prediction-System.py config.py ./

# Headless container: disable the cv2.imshow preview window
ENV SHOW_WINDOW=0

# Run the batch video processing script
CMD ["python", "Parking-Lot-Occupancy-and-Vehicle-Trajectory-Prediction-System.py"]
