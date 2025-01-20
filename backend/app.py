from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from gradio_client import Client
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Hugging Face client setup
os.environ['HF_TOKEN'] = ''
client = Client("black-forest-labs/FLUX.1-schnell")

# Directory to save generated images
GENERATED_IMAGES_DIR = 'generated_images'


@app.route('/generate-image', methods=['POST'])
def generate_image():
    try:
        # Parse JSON input
        data = request.json
        prompt = data.get('prompt')

        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        # Default parameters for image generation
        seed = 0
        randomize_seed = True
        width = 1024
        height = 1024
        num_inference_steps = 4

        # Generate the image using the API
        result = client.predict(
            prompt=prompt,
            seed=seed,
            randomize_seed=randomize_seed,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            api_name="/infer"
        )

        # Extract the file path
        output_filepath = result[0]

        # Get current date in the format: YYYY-MM-DD_HH-MM-SS
        current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f'generated_image_{current_date}_{prompt[:10].replace(" ", "_")}.png'
        destination_path = os.path.join(GENERATED_IMAGES_DIR, filename)

        # Ensure the directory exists
        os.makedirs(GENERATED_IMAGES_DIR, exist_ok=True)

        # Save the image to the desired location
        if os.path.exists(output_filepath):
            os.rename(output_filepath, destination_path)
            return jsonify({"message": "Image generated successfully", "file_path": filename}), 200
        else:
            return jsonify({"error": "Failed to retrieve the generated image"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/generated_images/<path:filename>', methods=['GET'])
def serve_image(filename):
    try:
        return send_from_directory(GENERATED_IMAGES_DIR, filename)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
