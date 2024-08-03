# main.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scripts.model import build_generator

# Load the preprocessed images and attributes
images = np.load('data/celebA/preprocessed_images.npy')
attributes = np.load('data/celebA/attributes.npy')

# Build generator model
generator = build_generator(input_shape=(100,))

# Load trained weights (assuming weights are saved after training)
generator.load_weights('path_to_saved_generator_weights')

# Attribute labels in CelebA dataset
attribute_labels = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs',
    'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',
    'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
    'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
    'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
    'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
]

# User selects attributes
selected_attributes = ['Smiling', 'Eyeglasses', 'Male']

# Create a binary attribute vector for the selected attributes
attribute_vector = np.zeros((1, len(attribute_labels)))
for attribute in selected_attributes:
    if attribute in attribute_labels:
        index = attribute_labels.index(attribute)
        attribute_vector[0, index] = 1

# Generate image with selected attributes
noise = tf.random.normal([1, 100])
generated_image = generator([noise, attribute_vector], training=False)

# Save and display the generated image
plt.imshow((generated_image[0] + 1) / 2)  # Denormalize the image
plt.axis('off')
plt.show()



# Save the image to the results directory
output_dir = 'results/generated_images/'
output_path = f"{output_dir}generated_image.png"
plt.imsave(output_path, (generated_image[0] + 1) / 2)
print(f"Image saved to {output_path}")