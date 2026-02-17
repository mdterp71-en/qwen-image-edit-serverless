FROM runpod/base:0.6.3-cuda12.2.0

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
```

And make sure your **requirements.txt** looks like this (using PyPI versions instead of git clones for a faster, more reliable build):
```
runpod>=1.7.0
torch>=2.1.0
torchvision
diffusers>=0.32.0
accelerate>=1.0.0
peft>=0.13.0
huggingface_hub
transformers
sentencepiece
Pillow
numpy
