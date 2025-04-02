from enum import Enum
from typing import List, Dict, Any
from ..engine import OCREngine
from ..models import FlatOCRResult, BoundingBox
from dataclasses import dataclass
from PIL import Image
import numpy as np


class PaddleOCRLanguageCode(Enum):
    ENGLISH = "en"
    CHINESE = "ch"
    FRENCH = "french"
    GERMAN = "german"
    KOREAN = "korean"
    JAPANESE = "japan"


DEFAULT_LANGUAGE_PREFERENCE = PaddleOCRLanguageCode.ENGLISH
DEFAULT_RECOGNIZE_UPSIDE_DOWN = False

PP_OCR_V4_SERVER = {
    "detection_model": "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar",
    "detection_yml": "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml",
    "recognition_model": "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar",
    "recognition_yml": "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml",
    "cls_model": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_slim_infer.tar",
}


class PaddleOCREngine(OCREngine):
    language_preference: str
    recognize_upside_down: bool
    use_v4_server: bool

    def __init__(self, config: Dict[str, Any] = {}) -> None:
        language_perference = config.get("language_preference", DEFAULT_LANGUAGE_PREFERENCE)
        try:
            self.language_preference = language_perference.value
        except KeyError:
            raise ValueError("Unsupported language code")

        recognize_upside_down = config.get("recognize_upside_down", DEFAULT_RECOGNIZE_UPSIDE_DOWN)
        if isinstance(recognize_upside_down, bool):
            self.recognize_upside_down = recognize_upside_down
        else:
            raise ValueError("recognize_upside_down must be a boolean")

        use_v4_server = config.get("use_v4_server", False)
        if isinstance(use_v4_server, bool):
            self.use_v4_server = use_v4_server
        else:
            raise ValueError("use_v4_server must be a boolean")

        from paddleocr import PaddleOCR

        if not self.use_v4_server:
            self.ocr = PaddleOCR(
                use_angle_cls=self.recognize_upside_down,
                lang=self.language_preference,
                use_gpu=True,
            )
        else:
            self.__download_v4_server_models()
            self.ocr = PaddleOCR(
                use_angle_cls=self.recognize_upside_down,
                det_model_dir="./models/PP-OCRv4/chinese/ch_PP-OCRv3_det_infer",
                det_algorithm="DB",
                rec_model_dir="./models/PP-OCRv4/chinese/ch_PP-OCRv3_rec_infer",
                rec_algorithm="CRNN",
                cls_model_dir="./models/PP-OCRv4/chinese/ch_ppocr_mobile_v2.0_cls_slim_infer",
                use_gpu=True,
            )

    def __download_v4_server_models(self):
        import os
        import requests
        import tarfile

        def download_file(url: str, dest_path: str):
            print(f"⬇️  Downloading from: {url}")
            r = requests.get(url, allow_redirects=True, stream=True)
            content_type = r.headers.get("Content-Type", "")
            content_length = int(r.headers.get("Content-Length", "0"))

            if dest_path.endswith(".yml"):
                if "html" in content_type:
                    raise RuntimeError(
                        f"❌ Invalid content received from {url} "
                        f"(Content-Type: {content_type}). Possible broken link or HTML page."
                    )
            else:
                if content_length < 1_000_000 or "html" in content_type:
                    raise RuntimeError(
                        f"❌ Invalid content received from {url} "
                        f"(Content-Type: {content_type}, Size: {content_length} bytes). "
                        "Possible broken link or temporary server issue."
                    )

            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"✔ Downloaded to: {dest_path}")

        def safe_extract_tar(filepath: str, extract_to: str):
            try:
                with tarfile.open(filepath) as tar:
                    tar.extractall(extract_to)
                print(f"✔ Successfully extracted: {filepath}")
            except tarfile.ReadError:
                print(f"✖ Extraction failed: {filepath}. Removing corrupted file...")
                os.remove(filepath)
                raise RuntimeError(f"Corrupted tar file: {filepath}")

        base_path = "./models/PP-OCRv4/chinese"
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        files_to_download = {
            "ch_PP-OCRv3_det_infer.tar": PP_OCR_V4_SERVER["detection_model"],
            "ch_PP-OCRv3_rec_infer.tar": PP_OCR_V4_SERVER["recognition_model"],
            "ch_ppocr_mobile_v2.0_cls_slim_infer.tar": PP_OCR_V4_SERVER["cls_model"],
        }

        for filename, url in files_to_download.items():
            tar_path = os.path.join(base_path, filename)
            model_dir = tar_path.replace(".tar", "")

            if not os.path.exists(model_dir):
                if not os.path.exists(tar_path):
                    download_file(url, tar_path)
                try:
                    safe_extract_tar(tar_path, base_path)
                except RuntimeError:
                    print(f"Retrying download for: {filename}")
                    download_file(url, tar_path)
                    safe_extract_tar(tar_path, base_path)

        yaml_files = {
            "ch_PP-OCRv3_det_cml.yml": PP_OCR_V4_SERVER["detection_yml"],
            "ch_PP-OCRv3_rec_distillation.yml": PP_OCR_V4_SERVER["recognition_yml"],
        }

        for filename, url in yaml_files.items():
            yaml_path = os.path.join(base_path, filename)
            if not os.path.exists(yaml_path):
                download_file(url, yaml_path)

    def recognize(self, image: Image.Image) -> List[FlatOCRResult]:
        image_array = np.array(image)
        slice = {
            "horizontal_stride": 300,
            "vertical_stride": 500,
            "merge_x_thres": 50,
            "merge_y_thres": 35,
        }
        annotations = self.ocr.ocr(image_array, cls=True, slice=slice)
        paddle_ocr_results = []

        for annotation in annotations:
            if not isinstance(annotation, list):
                continue

            for res in annotation:
                boundingbox = res[0]
                text = res[1][0]
                confidence = res[1][1]
                paddle_ocr_results.append(
                    PaddleOCRResult(
                        text=text,
                        confidence=confidence,
                        bounding_box=boundingbox,
                        image_width=image.width,
                        image_height=image.height,
                        use_v4_server=(self.use_v4_server),
                    )
                )

        flat_ocr_results = [
            paddle_ocr_result.to_flat() for paddle_ocr_result in paddle_ocr_results
        ]

        return flat_ocr_results


@dataclass
class PaddleOCRResult:
    text: str
    bounding_box: List[List[float]]
    confidence: float
    image_width: int
    image_height: int
    use_v4_server: bool

    def to_flat(self):
        left = min(
            self.bounding_box[0][0],
            self.bounding_box[1][0],
            self.bounding_box[2][0],
            self.bounding_box[3][0],
        )
        right = max(
            self.bounding_box[0][0],
            self.bounding_box[1][0],
            self.bounding_box[2][0],
            self.bounding_box[3][0],
        )
        bottom = max(
            self.bounding_box[0][1],
            self.bounding_box[1][1],
            self.bounding_box[2][1],
            self.bounding_box[3][1],
        )
        top = min(
            self.bounding_box[0][1],
            self.bounding_box[1][1],
            self.bounding_box[2][1],
            self.bounding_box[3][1],
        )

        return FlatOCRResult(
            text=self.text,
            confidence=self.confidence,
            bounding_box=BoundingBox(
                left=left / self.image_width,
                top=top / self.image_height,
                right=right / self.image_width,
                bottom=bottom / self.image_height,
                width=(right - left) / self.image_width,
                height=(bottom - top) / self.image_height,
            ),
            engine=("PADDLE_OCR_SERVER_V4" if self.use_v4_server else "PADDLE_OCR"),
        )