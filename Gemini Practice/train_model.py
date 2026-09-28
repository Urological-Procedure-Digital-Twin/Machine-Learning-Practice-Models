import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. Function to Load and Align Dataset
def load_and_align_dataset(csv_path, video_path):
    sensor_df = pd.read_csv(csv_path)
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    min_len = min(len(sensor_df), len(frames))
    X = sensor_df[['flow_rate_lpm', 'pressure_psi']].iloc[:min_len].values
    Y = np.array(frames[:min_len])
    
    return X, Y

# 2. Load and Prepare Tensors
X, Y = load_and_align_dataset('sensor_data.csv', 'scope_video.mp4')

# Normalize inputs and outputs
X_norm = (X - X.mean(axis=0)) / X.std(axis=0)
Y_norm = np.transpose(Y, (0, 3, 1, 2)) / 255.0

X_tensor = torch.tensor(X_norm, dtype=torch.float32)
Y_tensor = torch.tensor(Y_norm, dtype=torch.float32)

dataset = TensorDataset(X_tensor, Y_tensor)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 3. Define Generator Model
class ScopeImageGenerator(nn.Module):
    def __init__(self):
        super(ScopeImageGenerator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Linear(256, 128 * 15 * 20),
            nn.ReLU()
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.fc(x)
        out = out.view(-1, 128, 15, 20)
        out = self.deconv(out)
        return out

# 4. Train Model
model = ScopeImageGenerator()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("🚀 Starting Training Loop...\n")
for epoch in range(10):
    total_loss = 0.0
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/10] - Loss: {total_loss / len(dataloader):.6f}")

print("\n✅ Training Complete!")

# 5. Generate and Save Test Image
model.eval()
with torch.no_grad():
    sample_input = X_tensor[0].unsqueeze(0)
    predicted_frame = model(sample_input).squeeze(0).numpy()
    predicted_frame = (np.transpose(predicted_frame, (1, 2, 0)) * 255.0).astype(np.uint8)
    cv2.imwrite('predicted_scope_frame.png', predicted_frame)
    print("Saved predicted frame as 'predicted_scope_frame.png'")