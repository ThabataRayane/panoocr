# PanoOCR

PanoOCR is a Python library for performing Optical Character Recognition (OCR) on equirectangular panorama images. It automatically handles the conversion between flat and spherical coordinates, making it ideal for OCR tasks involving 360° panoramic content.

# Demo

This is a demo using the built-in [preview tool](#Interactive-Preview-Tool) with the test results in `/assets` folder.

https://github.com/user-attachments/assets/57507c48-ec88-4d4a-bf68-067eefc9d42f

## Features

- Support for multiple OCR engines:
  - macOCR (macOS native OCR)
  - PaddleOCR (with optional V4 server model)
  - EasyOCR
  - Florence2
  - TrOCR
- Automatic perspective generation from equirectangular panoramas
- Spherical coordinate conversion
- Duplication detection and removal across perspectives
- Multi-language support (depending on OCR engine)
- Interactive preview tool for visualization

## Installation

1. Clone the repository:

   ```bash
   git clone [repository-url]
   cd panoocr
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Install ocr-engine-specific requirements based on your needs:

   ```bash
   # For PaddleOCR
   pip install -r requirements-paddle.txt

   # For EasyOCR
   pip install -r requirements-easyocr.txt

   # For macOCR (macOS native ocr, aka. Apple Vision Framework)
   pip install -r requirements-mac.txt

   # For Florence2
   pip install -r requirements-huggingface.txt
   ```

## Usage

### Basic Usage

The basic usage is showcased in [`run_panoocr.py`](run_panoocr.py)

To run the script, simply execute:

```bash
python run_panoocr.py --ocr-engine [ocr-engine-name] --image-path [path-to-panorama-image]
```

The result will be saved in the same folder of the original image, with the same filename but with a different extension: `.[ocr-engine-name].json`

## Core Components

The core building blocks of `panoocr` are:

- `OCREngine`: Represents an OCR engine.
- `SphereOCRDuplicationDetectionEngine`: Represents a duplication detection engine.
- `PanoramaImage`: Represents an equirectangular panorama image.
- `PerspectiveMetadata`: Represents a perspective view of the panorama.

For the two engines, `OCREngine` and `SphereOCRDuplicationDetectionEngine`, you can bring your own implementations via inheritance. Their APIs are defined in [`engine.py`](panoocr/ocr/engine.py) and [`duplication_detection.py`](panoocr/ocr/duplication_detection.py) respectively.

By default, `run_panoocr.py` uses `po.DEFAULT_IMAGE_PERSPECTIVES` which is 16 perspectives with the following settings:

- pixel_width: 2048
- pixel_height: 2048
- 45° horizontal field of view
- 0° yaw offset
- 0° pitch offset
- 22.5° yaw interval

You can also specify different perspective settings by directly constructing `PerspectiveMetadata` objects as such:

```python
perspective = po.PerspectiveMetadata(
  pixel_width=1024,
  pixel_height=512,
  horizontal_fov=45,
  vertical_fov=45,
  yaw_offset=0,
  pitch_offset=0,
)
```

## Interactive Preview Tool

I also built a web-based interactive preview tool that allows you to visualize the OCR results on the panorama image. It's located in `preview/index.html`. To run it, run a http server in the `preview` folder:

```bash
cd preview && python -m http.server
```

Then, open your browser and navigate to `http://localhost:8000`, you should see the preview tool.

Simply drag and drop the JSON result file and your panorama image to the interface, and you should see the OCR results overlaid on the panorama image.
