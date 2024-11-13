from src import panoocr as po
import json
import os
import sys
import argparse
import time

########################################################
# MARK: ARGUMENTS
########################################################

parser = argparse.ArgumentParser(description="Run panoocr for a given image")

# Add OCR_ENGINE argument
parser.add_argument(
    "--ocr-engine",
    choices=["macocr", "florence2", "paddleocr", "easyocr", "trocr"],
    default="macocr",
    help="Choose OCR engine",
)


# Add IMAGE_PATH argument
parser.add_argument(
    "--image-path",
    type=str,
    help="Path to the equirectangular panorama image",
    default="assets/test-pano.jpg",
)

args = parser.parse_args()

OCR_ENGINE_NAME = args.ocr_engine
IMAGE_PATH = args.image_path


########################################################
# MARK: INITIALIZATION
########################################################

OCR_ENGINE = None

if OCR_ENGINE_NAME == "macocr":
    OCR_ENGINE = po.create_ocr_engine(
        po.OCREngineType.MACOCR,
        {
            "language_preference": [
                po.MacOCRLanguageCode.ENGLISH_US,
            ],
            "recognition_level": po.MacOCRRecognitionLevel.ACCURATE,
        },
    )
elif OCR_ENGINE_NAME == "florence2":
    OCR_ENGINE = po.create_ocr_engine(
        po.OCREngineType.FLORENCE,
        {},
    )
elif OCR_ENGINE_NAME == "paddle":
    OCR_ENGINE = po.create_ocr_engine(
        engine_type=po.OCREngineType.PADDLEOCR,
        config={
            "language_preference": po.PaddleOCRLanguageCode.ENGLISH,
            "recognize_upside_down": False,
            "use_v4_server": True,
        },
    )
elif OCR_ENGINE_NAME == "easyocr":
    OCR_ENGINE = po.create_ocr_engine(
        engine_type=po.OCREngineType.EASYOCR,
        config={},
    )
elif OCR_ENGINE_NAME == "trocr":
    OCR_ENGINE = po.create_ocr_engine(
        engine_type=po.OCREngineType.TROCR,
        config={},
    )
else:
    raise ValueError("Invalid OCR engine")

DUPLICATION_DETECTION_ENGINE = po.SphereOCRDuplicationDetectionEngine()

PERSPECTIVES = po.DEFAULT_IMAGE_PERSPECTIVES


########################################################
# MARK: MAIN
########################################################


def main():
    panorama_image = po.PanoramaImage("Test", IMAGE_PATH)

    all_sphere_ocr_results_for_each_perspective = []
    perspective_count = len(PERSPECTIVES)

    print(f"Total Perspectives: {perspective_count}")

    for i, perspective in enumerate(PERSPECTIVES):
        print(f"Generating Perspective #{i}")

        # Get the image
        perspective_image = panorama_image.generate_perspective_image(perspective)
        perspective_pil_image = perspective_image.get_perspective_image()

        print(f"OCR Perspective #{i}")
        flat_ocr_results = OCR_ENGINE.recognize(perspective_pil_image)

        print(f"Converting OCR Results to Sphere Coordinates")
        sphere_ocr_results = [
            flat_ocr_result.to_sphere(
                horizontal_fov=perspective.horizontal_fov,
                vertical_fov=perspective.vertical_fov,
                yaw_offset=perspective.yaw_offset,
                pitch_offset=perspective.pitch_offset,
            )
            for flat_ocr_result in flat_ocr_results
        ]

        all_sphere_ocr_results_for_each_perspective.append(sphere_ocr_results)

    print("Removing Duplications")
    for i in range(0, perspective_count):
        print(f"Removing Duplications for Perspective #{i}")
        first_perspective_index = i
        second_perspective_index = 0 if i == perspective_count - 1 else i + 1

        (
            new_ocr_results_first_frame,
            new_ocr_results_second_frame,
        ) = DUPLICATION_DETECTION_ENGINE.remove_duplication_for_two_lists(
            all_sphere_ocr_results_for_each_perspective[first_perspective_index],
            all_sphere_ocr_results_for_each_perspective[second_perspective_index],
        )

        all_sphere_ocr_results_for_each_perspective[first_perspective_index] = (
            new_ocr_results_first_frame
        )
        all_sphere_ocr_results_for_each_perspective[second_perspective_index] = (
            new_ocr_results_second_frame
        )

    all_sphere_ocr_results_no_duplication = [
        result.to_dict()
        for perspective_results in all_sphere_ocr_results_for_each_perspective
        for result in perspective_results
    ]

    # Save the result at the same folder of the original image, but with a different extension
    save_path = f"{os.path.splitext(IMAGE_PATH)[0]}.{OCR_ENGINE_NAME}.json"
    with open(save_path, "w") as f:
        json.dump(all_sphere_ocr_results_no_duplication, f)


if __name__ == "__main__":
    main()
