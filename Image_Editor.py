import math
import numpy as np
import PySimpleGUI as sg
from PIL import Image, ImageEnhance
from io import BytesIO
import cv2
import yaml
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')  # Set the matplotlib backend to TkAgg for compatibility with Tkinter

def get_image_path():
    # Allow user to choose any image
    layout = [[sg.Text("Choose an image file")], [sg.Input(), sg.FileBrowse()], [sg.OK(), sg.Cancel()]]
    window = sg.Window('Open Image File', layout)
    event, values = window.read()
    window.close()

    return values[0] if event == 'OK' and values[0] else None

def np_im_to_data(im):
    # Put image into a numpy array so we can manipulate it
    array = np.array(im, dtype=np.uint8)
    im = Image.fromarray(array)

    with BytesIO() as output:
        im.save(output, format='PNG')
        data = output.getvalue()

    return data

def draw_histogram(before, after):
    # Draw the said histogram on a fig ax plt
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    construct_image_histogram(before, ax, 'Before')
    construct_image_histogram(after, ax, 'After')

    plt.title('Histogram Overlay')
    plt.tight_layout()

    return fig

def draw_hist(canvas, figure, hist_canvas):
    # Remove the histogram before drawing a new one
    clear_histogram(hist_canvas)
    tkcanvas = FigureCanvasTkAgg(figure, canvas)
    tkcanvas.draw()
    tkcanvas_widget = tkcanvas.get_tk_widget()
    tkcanvas_widget.pack(side='top', fill='both', expand=1)

    return tkcanvas

def clear_histogram(hist_canvas):
    if hist_canvas is not None:
        widget = hist_canvas.get_tk_widget()
        widget.destroy()

def construct_image_histogram(np_image, ax, title):
    # How histogram will look like
    L = 256
    bins = np.arange(L + 1)
    hist, _ = np.histogram(np_image, bins)

    ax.bar(np.arange(len(hist)), hist, alpha=0.5, label=title)
    ax.legend()

#God damn this function
def draw_strokes(image, stroke_length=15, stroke_width=2):
    strokes = np.copy(image)
    angle = np.deg2rad(45)
    for i in range(0, image.shape[0], stroke_length):
        for j in range(0, image.shape[1], stroke_length):
            end_x = int(j + stroke_length * np.cos(angle))
            end_y = int(i + stroke_length * np.sin(angle))
            if end_x < image.shape[1] and end_y < image.shape[0]:
                cv2.line(strokes, (j, i), (end_x, end_y), image[i, j].tolist(), stroke_width)
    return strokes

def detect_edges(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute gradients in x and y directions
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude and direction
    grad_magnitude = cv2.magnitude(grad_x, grad_y)
    grad_magnitude = cv2.convertScaleAbs(grad_magnitude)
    
    # Threshold gradient magnitude to get edges
    _, edge_mask = cv2.threshold(grad_magnitude, threshold, 255, cv2.THRESH_BINARY)
    
    return edge_mask

def clip_strokes(strokes, edge_mask):
    return cv2.bitwise_and(strokes, strokes, mask=edge_mask)

def apply_painted_effect(image):
    strokes = draw_strokes(image)
    edge_mask = detect_edges(image)
    final_image = clip_strokes(strokes, edge_mask)
    return final_image

def bi_linear_interp_resize(original_img, new_height, new_width):
    # Get dimensions of original image
    original_height, original_width, c = original_img.shape

    # Create an array of the desired shape
    resized_image = np.zeros((new_height, new_width, c))

    # Calculate horizontal and vertical scaling factor
    width_scale_factor = original_width / new_width if new_width != 0 else 0
    height_scale_factor = original_height / new_height if new_height != 0 else 0

    for i in range(new_height):
        # Map the x coordinates back to the original image
        x = i * height_scale_factor
        # Calculate the coordinate values for 4 surrounding pixels (first 2)
        x_floor = math.floor(x)
        x_ceil = min(original_height - 1, math.ceil(x))

        for j in range(new_width):
            # Map the y coordinates back to the original image
            y = j * width_scale_factor
            # Calculate the coordinate values for 4 surrounding pixels (remaining 2)
            y_floor = math.floor(y)
            y_ceil = min(original_width - 1, math.ceil(y))

            #we check if the point is on an original point x or y or maps to a point in the original image
            if (x_ceil == x_floor) and (y_ceil == y_floor):
                interpolated_value = original_img[int(x), int(y), :]

            elif x_ceil == x_floor:
                q1 = original_img[int(x), int(y_floor), :]
                q2 = original_img[int(x), int(y_ceil), :]
                interpolated_value = q1 * (y_ceil - y) + q2 * (y - y_floor)

            elif y_ceil == y_floor:
                q1 = original_img[int(x_floor), int(y), :]
                q2 = original_img[int(x_ceil), int(y), :]
                interpolated_value = (q1 * (x_ceil - x)) + (q2 * (x - x_floor))

            else:
                pixel_value_1 = original_img[x_floor, y_floor, :]
                pixel_value_2 = original_img[x_ceil, y_floor, :]
                pixel_value_3 = original_img[x_floor, y_ceil, :]
                pixel_value_4 = original_img[x_ceil, y_ceil, :]

                q1 = pixel_value_1 * (x_ceil - x) + pixel_value_2 * (x - x_floor)
                q2 = pixel_value_3 * (x_ceil - x) + pixel_value_4 * (x - x_floor)
                interpolated_value = q1 * (y_ceil - y) + q2 * (y - y_floor)

            resized_image[i, j, :] = interpolated_value

    return resized_image.astype(np.uint8)

def nn_resize(original_img, new_height, new_width):
    # Get dimensions of original image
    original_height, original_width, c = original_img.shape

    # Create an array of the desired shape
    resized_image = np.zeros((new_height, new_width, c), dtype=np.uint8)

    # Calculate horizontal and vertical scaling factor
    width_scale_factor = original_width / new_width if new_width != 0 else 0
    height_scale_factor = original_height / new_height if new_height != 0 else 0

    for i in range(new_height):
        for j in range(new_width):
            # Map the coordinates back to the original image
            x = int(i * height_scale_factor)
            y = int(j * width_scale_factor)
            resized_image[i, j, :] = original_img[x, y, :]

    return resized_image

def resize_image(image, max_width, max_height, maintain_aspect_ratio=True):
    original_height, original_width = image.shape[:2]

    # Maintein aspect ratio
    if maintain_aspect_ratio:
        aspect_ratio = original_width / original_height
        if max_width / max_height > aspect_ratio:
            new_height = max_height
            new_width = int(max_height * aspect_ratio)
        else:
            new_width = max_width
            new_height = int(max_width / aspect_ratio)
    else:
        new_width, new_height = max_width, max_height

    # Im using this only to load the image because my resize is VERY SLOW
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return resized_image

def update_window_with_images(window, before_data, after_data):
    # Update function
    window['-BEFORE-'].update(data=before_data)  # Update 'before' image
    window['-AFTER-'].update(data=after_data)   # Update 'after' image

def save_image(image):
    save_path = sg.popup_get_file('Save Image', save_as=True, no_window=True, default_extension='.png')
    if save_path:
        im = Image.fromarray(image)
        im.save(save_path)

def save_settings(settings):
    save_path = sg.popup_get_file('Save Settings', save_as=True, no_window=True, default_extension='.yaml')
    if save_path:
        with open(save_path, 'w') as file:
            yaml.dump(settings, file)

def load_settings():
    load_path = sg.popup_get_file('Load Settings', no_window=True, default_extension='.yaml')
    if load_path:
        with open(load_path, 'r') as file:
            return yaml.safe_load(file)
    return None

def resize_dialog():
    #pop out window for resize image
    layout = [
        [sg.Text('Enter new width:'), sg.InputText(key='-WIDTH-', size=(10, 1))],

        [sg.Text('Enter new height:'), sg.InputText(key='-HEIGHT-', size=(10, 1))],

        [sg.Checkbox('Maintain Aspect Ratio', key='-ASPECT-RATIO-')],

        [sg.Text('Select Resizing Method:')],

        [sg.Radio('Bilinear', 'RESIZE_METHOD', default=True, key='-BILINEAR-'),
         sg.Radio('Nearest Neighbor', 'RESIZE_METHOD', key='-NEAREST-')],

        [sg.OK(), sg.Cancel()]
    ]

    window = sg.Window('Resize Image', layout)
    event, values = window.read()
    window.close()
    if event == 'OK':
        width = int(values['-WIDTH-']) if values['-WIDTH-'] else None
        height = int(values['-HEIGHT-']) if values['-HEIGHT-'] else None
        if width and height:
            maintain_aspect_ratio = values['-ASPECT-RATIO-']
            method = 'bilinear' if values['-BILINEAR-'] else 'nearest'
            return width, height, method, maintain_aspect_ratio
        
    return None

def adjust_saturation(image, saturation_factor):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv_image[:, :, 1] = cv2.multiply(hsv_image[:, :, 1], np.array([saturation_factor]))
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

def adjust_color_temperature(image, temp_value):
    # Increase or decrease the red and blue channels to adjust temperature
    b, g, r = cv2.split(image)
    if temp_value > 0:
        r = cv2.add(r, temp_value)
        b = cv2.subtract(b, temp_value)
    else:
        r = cv2.subtract(r, -temp_value)
        b = cv2.add(b, -temp_value)
    r = np.clip(r, 0, 255)
    b = np.clip(b, 0, 255)
    return cv2.merge((b, g, r))

def apply_image_effects(image, values):
    # Gather values from the sliders
    saturation_factor = values['-SATURATION-']
    contrast_factor = values['-CONTRAST-']
    temp_value = values['-TEMP-']

    # Apply the effects
    image = adjust_saturation(image, saturation_factor)
    image = apply_contrast(image, contrast_factor)  # Assuming this function exists and works correctly
    image = adjust_color_temperature(image, temp_value)
    return image

def apply_contrast(image, factor):
    # Contrast the image
    enhancer = ImageEnhance.Contrast(Image.fromarray(image))
    enhanced_im = enhancer.enhance(factor)
    return np.array(enhanced_im)

def main():
    # First part of the program (choosing the path)
    image_path = get_image_path()
    if not image_path:
        sg.popup("No image selected or file path is invalid. Exiting application.")
        return

    # Reading the image from the specified path
    original_image = cv2.imread(image_path)
    if original_image is None:
        sg.popup("Failed to load image. Please check the file path.", title="Error")
        return
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Initialize the resized image and image history
    resized_image = resize_image(original_image, 640, 480, maintain_aspect_ratio=True)
    before_image_data = np_im_to_data(Image.fromarray(resized_image))
    after_image = resized_image.copy()
    image_history = [np_im_to_data(resized_image)]  # Starting with the initial image in history
    change_count = 0  # Counter for how many times the image has been changed

    # UI layout
    layout = [
        [sg.Text(f'Filename: {image_path.split("/")[-1]}', size=(40, 1)), sg.Text(f'Size: {original_image.shape[1]}x{original_image.shape[0]}')],

        [sg.Image(data=before_image_data, key='-BEFORE-'),
         sg.Image(data=before_image_data, key='-AFTER-')],

        
        [sg.Text('Averaging Blur'), sg.Slider(range=(0, 15), default_value=0, resolution=1, orientation='horizontal', key='-AVG-BLUR-', tooltip='Averaging Blur')],
        [sg.Text('Gaussian Blur'), sg.Slider(range=(0, 15), default_value=0, resolution=1, orientation='horizontal', key='-GAUSS-BLUR-', tooltip='Gaussian Blur')],

        [sg.Text('Contrast'), sg.Slider(range=(0.5, 1.5), default_value=1.0, resolution=0.1, orientation='horizontal', key='-CONTRAST-', tooltip='Contrast')],
        [sg.Text('Saturation'), sg.Slider(range=(0.5, 2.0), default_value=1.0, resolution=0.1, orientation='horizontal', key='-SATURATION-', tooltip='Saturation')],
        [sg.Text('Color Temperature'), sg.Slider(range=(-50, 50), default_value=0, resolution=1, orientation='horizontal', key='-TEMP-', tooltip='Color Temperature')],

        [sg.Canvas(key='-HIST-', visible=False)],

        [sg.Text("Image Changes: 0", key='-COUNTER-')],

        [sg.Button('Grayscale'), sg.Button('Histogram Equalize'), sg.Button('Apply Filters'),sg.Button('Apply Painted Look'), sg.Button('Show/Hide Histogram'), sg.Button('Undo'), sg.Button('Redo'),
        sg.Button('Resize'), sg.Button('Save'), sg.Button('Save Settings'), sg.Button('Load Settings'), sg.Button('Exit')]
    ]

    window = sg.Window('Image Processing App', layout, finalize=True)
    update_window_with_images(window, before_image_data, before_image_data)

    fig = draw_histogram(resized_image, resized_image)
    hist_canvas = draw_hist(window['-HIST-'].TKCanvas, fig, None)

    
    while True:

        event, values = window.read()
        # Exit
        if event == sg.WINDOW_CLOSED or event == 'Exit':
            break

        # Show/Hide Histogram
        if event == 'Show/Hide Histogram':
            current_visibility = window['-HIST-'].visible
            window['-HIST-'].update(visible=not current_visibility)

        # Undo
        elif event == 'Undo':
            if change_count > 0:
                change_count -= 1
                after_image_data = image_history[change_count]
                update_window_with_images(window, before_image_data, after_image_data)
                window['-COUNTER-'].update(f"Image Changes: {change_count}")
                after_image = np.array(Image.open(BytesIO(after_image_data)))

                fig = draw_histogram(resized_image, after_image)
                hist_canvas = draw_hist(window['-HIST-'].TKCanvas, fig, hist_canvas)

        # Redo
        elif event == 'Redo':
            if change_count < len(image_history) - 1:
                change_count += 1
                after_image_data = image_history[change_count]
                update_window_with_images(window, before_image_data, after_image_data)
                window['-COUNTER-'].update(f"Image Changes: {change_count}")
                after_image = np.array(Image.open(BytesIO(after_image_data)))

                fig = draw_histogram(resized_image, after_image)
                hist_canvas = draw_hist(window['-HIST-'].TKCanvas, fig, hist_canvas)

        # Resize
        elif event == 'Resize':
            resize_params = resize_dialog()
            if resize_params:
                width, height, method, maintain_aspect_ratio = resize_params
                resized_image = resize_image(after_image, width, height, maintain_aspect_ratio)

                after_image_data = np_im_to_data(resized_image)
                image_history.append(after_image_data)
                change_count += 1
                update_window_with_images(window, before_image_data, after_image_data)
                window['-COUNTER-'].update(f"Image Changes: {change_count}")
                fig = draw_histogram(resized_image, resized_image)
                hist_canvas = draw_hist(window['-HIST-'].TKCanvas, fig, hist_canvas)

        # Save
        elif event == 'Save':
            save_image(after_image)

        # Save Settings
        elif event == 'Save Settings':
            settings = {
                'saturation': values['-SATURATION-'],
                'contrast': values['-CONTRAST-'],
                'temperature': values['-TEMP-']
            }
            save_settings(settings)

        # Load Settings
        elif event == 'Load Settings':
            loaded_settings = load_settings()
            if loaded_settings:
                window['-SATURATION-'].update(value=loaded_settings.get('saturation', 1.0))
                window['-CONTRAST-'].update(value=loaded_settings.get('contrast', 1.0))
                window['-TEMP-'].update(value=loaded_settings.get('temperature', 0))

        # Apply Painted Look
        elif event == 'Apply Painted Look':
            # Apply the painted effect using the existing function
            after_image = np.array(Image.open(BytesIO(image_history[change_count])))
            painted_image = apply_painted_effect(after_image)

            # Convert the updated image back to image data format for display and history tracking
            after_image_data = np_im_to_data(painted_image)

            # Append the new image data to the history or overwrite future history if we've undone changes
            if change_count == len(image_history) - 1:
                image_history.append(after_image_data)
            else:
                image_history[change_count + 1] = after_image_data
                image_history = image_history[:change_count + 2]  # Trim future history if new changes are made

            # Update history counter and window with the new image
            change_count += 1
            update_window_with_images(window, before_image_data, after_image_data)
            window['-COUNTER-'].update(f"Image Changes: {change_count}")

        # Apply other filters (Grayscale, Histogram Equalize, etc.)
        elif event in ('Grayscale', 'Histogram Equalize', 'Apply Filters'):
            # Always apply the filter to the latest version in history
            after_image = np.array(Image.open(BytesIO(image_history[change_count])))

            if event == 'Grayscale':
                after_image = cv2.cvtColor(after_image, cv2.COLOR_RGB2GRAY)
                after_image = cv2.cvtColor(after_image, cv2.COLOR_GRAY2RGB)  # Convert back to 3 channels for display purposes
            elif event == 'Histogram Equalize':
                hsv_image = cv2.cvtColor(after_image, cv2.COLOR_RGB2HSV)
                hsv_image[..., 2] = cv2.equalizeHist(hsv_image[..., 2])  # Histogram equalize the value channel
                after_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
            elif event == 'Apply Filters':
                # Apply the filters from sliders
                avg_blur_size = int(values['-AVG-BLUR-']) * 2 + 1  # Convert to odd numbers for kernel size
                gauss_blur_size = int(values['-GAUSS-BLUR-']) * 2 + 1  # Same for Gaussian
                contrast_value = values['-CONTRAST-']

                after_image = apply_contrast(after_image, contrast_value)
                if avg_blur_size > 1:
                    after_image = cv2.blur(after_image, (avg_blur_size, avg_blur_size))
                if gauss_blur_size > 1:
                    after_image = cv2.GaussianBlur(after_image, (gauss_blur_size, gauss_blur_size), gauss_blur_size / 3)

            # Convert the updated image back to image data format for display and history tracking
            after_image_data = np_im_to_data(after_image)

            # Append the new image data to the history or overwrite future history if we've undone changes
            if change_count == len(image_history) - 1:
                image_history.append(after_image_data)
            else:
                image_history[change_count + 1] = after_image_data
                image_history = image_history[:change_count + 2]  # Trim future history if new changes are made

            # Keeping track of the image history
            change_count += 1
            update_window_with_images(window, before_image_data, after_image_data)
            window['-COUNTER-'].update(f"Image Changes: {change_count}")

            # Update histogram
            fig = draw_histogram(resized_image, after_image)
            hist_canvas = draw_hist(window['-HIST-'].TKCanvas, fig, hist_canvas)

    window.close()


if __name__ == '__main__':
    main()
