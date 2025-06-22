
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Define the VAE model (must be the same as the one used for training)
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(784, 400)
        self.fc2_mu = nn.Linear(400, latent_dim)
        self.fc2_logvar = nn.Linear(400, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2_mu(h1), self.fc2_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Load the trained model
latent_dim = 20
model = VAE(latent_dim)
try:
    model.load_state_dict(torch.load('vae_mnist.pth'))
    model.eval() # Set the model to evaluation mode
except FileNotFoundError:
    st.error("Model file 'vae_mnist.pth' not found. Please ensure it's in the same directory.")
    st.stop()


# Create the Streamlit application interface
st.title("Handwritten Digit Generator")

# Add a slider for digit selection
selected_digit = st.slider("Select a digit", 0, 9, 0)

# Add a button to trigger image generation
generate_button = st.button("Generate Images")

if generate_button:
    # Generate 5 random latent vectors
    num_images = 5
    with torch.no_grad(): # Disable gradient calculation for inference
        z = torch.randn(num_images, latent_dim)

        # Decode the latent vectors into images
        generated_images = model.decode(z)

    # Reshape and prepare images for display
    st.write(f"Generating {num_images} images...")
    cols = st.columns(num_images)
    for i in range(num_images):
        # Reshape from 784 to 28x28
        image_tensor = generated_images[i].view(28, 28)

        # Convert to NumPy array
        image_np = image_tensor.numpy()

        # Display the image
        with cols[i]:
            st.image(image_np, caption=f"Generated Image {i+1}", use_column_width=True)

