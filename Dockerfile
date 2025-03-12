# Use PyTorch base image (CPU-only)
FROM pytorch/pytorch:latest

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# ✅ Make sure to copy the model into the container
COPY models/cat_dog_cnn.pth /app/models/cat_dog_cnn.pth


# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for Flask
EXPOSE 8000

# Start Flask app
CMD ["python", "src/inference.py"]
