# ============================================================================
# Dockerfile — Qwen-Image-Edit-2509-LoRAs-Fast-Lazy-Load on RunPod Serverless
# ============================================================================
# Add this file to the ROOT of your forked repo.
# RunPod's GitHub integration will build from it automatically.

FROM runpod/base:0.6.3-cuda12.2.0

WORKDIR /app

# Install pre-requirements first (if present in the repo)
COPY pre-requirements.txt /app/pre-requirements.txt
RUN pip install --no-cache-dir -r /app/pre-requirements.txt || true

# Install main requirements (from the repo)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Install the RunPod SDK (not in the original repo's requirements)
RUN pip install --no-cache-dir runpod>=1.7.0

# Copy the entire repo into the container
# This includes qwenimage/ (custom pipeline code), examples/, etc.
COPY . /app

# The handler is the entrypoint for RunPod Serverless
CMD ["python", "-u", "/app/handler.py"]
