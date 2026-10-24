"""
Flask application for Hand Sign Digit Recognition.
Handles web routing and model inference requests.
"""
from flask import Flask, render_template, request
from model import preprocess_img, predict_result

# Instantiating flask app
app = Flask(__name__)

@app.route("/")
def main():
    """Render the main index page."""
    return render_template("index.html")

@app.route('/prediction', methods=['POST'])
def predict_image_file():
    """Handle image upload, process it, and return the model's prediction."""
    try:
        if request.method == 'POST':
            img = preprocess_img(request.files['file'].stream)
            pred = predict_result(img)
            return render_template("result.html", predictions=str(pred))
    except Exception as e: # Fixed bare except block
        error = f"File cannot be processed. {str(e)}"
        return render_template("result.html", err=error)
    
    return render_template("index.html")

# Driver code
if __name__ == "__main__":
    app.run(port=9000, debug=True)