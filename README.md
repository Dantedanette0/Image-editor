# Image Processing App

This interactive desktop application allows you to apply various image processing filters and transformations to your photos using a simple graphical user interface built with PySimpleGUI. It supports features such as blurring, contrast and saturation adjustment, color temperature tuning, histogram visualization, painted look effects, undo/redo functionality, resizing, and saving of images and settings.

## Features

* **Open Any Image**: Browse and load images in common formats (PNG, JPEG, etc.).
* **Before/After View**: Compare original and processed images side by side.
* **Blur Filters**: Apply Averaging Blur and Gaussian Blur with adjustable intensity.
* **Color Adjustments**:

  * Contrast control
  * Saturation control
  * Color temperature (warmth) adjustment
* **Histogram Visualization**: Toggle a real-time overlay histogram of pixel intensity distributions.
* **Painted Look Effect**: Apply a stylized, stroke-based effect that highlights edges.
* **Basic Filters**:

  * Grayscale conversion
  * Histogram equalization (value channel)
* **Undo / Redo**: Step backward and forward through multiple edits.
* **Resize**: Resize images with optional aspect ratio preservation, using PySimpleGUI dialogs.
* **Save**: Export the processed image as a PNG file.
* **Settings Persistence**: Save and load filter settings (`contrast`, `saturation`, `temperature`) via YAML files.

## Prerequisites

* Python 3.7 or higher
* [PySimpleGUI](https://pysimplegui.readthedocs.io/)
* [OpenCV (cv2)](https://opencv.org/)
* [Pillow (PIL)](https://python-pillow.org/)
* [NumPy](https://numpy.org/)
* [Matplotlib](https://matplotlib.org/)
* [PyYAML](https://pyyaml.org/)

You can install all dependencies with:

```bash
pip install -r requirements.txt
```

### Example `requirements.txt`

```
PySimpleGUI
opencv-python
Pillow
numpy
matplotlib
PyYAML
```

## Installation

1. Clone the repository:

   ```bash
   ```

git clone [https://github.com/your-username/image-processing-app.git](https://github.com/your-username/image-processing-app.git)
cd image-processing-app

````
2. (Optional) Create and activate a virtual environment:
   ```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate   # Windows
````

3. Install the required packages:

   ```bash
   ```

pip install -r requirements.txt

````

## Usage

Run the main script to launch the GUI:

```bash
python app.py
````

1. **Open Image**: Upon startup, select an image file.
2. **Apply Filters**:

   * Adjust sliders for blur, contrast, saturation, and temperature, then click **Apply Filters**.
   * Click **Grayscale**, **Histogram Equalize**, or **Apply Painted Look** to apply those effects.
3. **View Histogram**: Click **Show/Hide Histogram** to toggle the histogram overlay.
4. **Undo/Redo**: Use **Undo** and **Redo** to navigate your edit history.
5. **Resize**: Click **Resize** to specify new dimensions and choose between Bilinear or Nearest Neighbor interpolation.
6. **Save Image**: Click **Save** to export the processed image.
7. **Save/Load Settings**: Use **Save Settings** and **Load Settings** to persist filter configurations in YAML format.
8. **Exit**: Click **Exit** or close the window to quit the application.

## Code Structure

* `app.py`: Main application script that builds the GUI and handles events.
* **Helper Functions**:

  * Image conversion between NumPy arrays and byte data for PySimpleGUI.
  * Histogram construction and visualization using Matplotlib.
  * Custom implementations of bilinear and nearest-neighbor resizing (for demonstration).
  * Stroke-based painting effect and edge detection via OpenCV.
  * Color space adjustments (HSV-based saturation, temperature manipulation, contrast enhancement with Pillow).

## Contributing

Contributions are welcome! Please open issues or submit pull requests with improvements, bug fixes, or new features.

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m "Add some feature"`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

*Created with ❤️ using PySimpleGUI and OpenCV.*
