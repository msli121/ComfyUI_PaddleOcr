import os
import numpy as np
import torch
import folder_paths
import torchvision.transforms.functional as F
from .onnx_paddleocr import ONNXPaddleOcr, out2Img
import json


class DownloadAndLoadPaddleOcrModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    [
                        "karmueo/PaddleOcr",
                    ],
                    {"default": "karmueo/PaddleOcr"},
                ),
            },
        }

    RETURN_TYPES = ("OCRMODEL",)
    RETURN_NAMES = ("ocr_model",)
    FUNCTION = "download_and_load_model"
    CATEGORY = "PaddleOCR"

    def download_and_load_model(self, model):
        """
        下载并加载模型
        """
        # 定义模型路径
        # 从一个包含路径的字符串中提取出模型的名称或文件名
        model_name = model.rsplit("/", 1)[-1]
        model_path = os.path.join(folder_paths.models_dir, "OCR", model_name)

        # 如果模型不存在，则从 HuggingFace 下载
        if not os.path.exists(model_path):
            print(f"Downloading model {model_name} to {model_path}")
            from huggingface_hub import snapshot_download

            # 从 HuggingFace Hub 下载模型
            # repo_id: 模型仓库的名称
            # local_dir: 下载到本地指定目录
            snapshot_download(
                repo_id=model,
                local_dir=model_path,
                force_download=False,  # 如果文件已存在，则跳过下载
            )

        # 加载模型
        det_model = os.path.join(model_path, "det.onnx")
        rec_model = os.path.join(model_path, "rec.onnx")
        cls_model = os.path.join(model_path, "cls.onnx")
        rec_char_dict_path = os.path.join(model_path, "ppocr_keys_v1.txt")
        vis_font_path = os.path.join(model_path, "simfang.ttf")

        model = ONNXPaddleOcr(
            det_model_dir=det_model,
            rec_model_dir=rec_model,
            cls_model_dir=cls_model,
            rec_char_dict_path=rec_char_dict_path,
            vis_font_path=vis_font_path,
            draw_img_save_dir=model_path,
            crop_res_save_dir=model_path,
            save_log_path=model_path,
            use_angle_cls=True,
            use_gpu=True,
        )

        # 返回模型
        return (model,)


class PaddleOcrRun:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {  # 必填输入项
                "image": ("IMAGE",),  # 输入图像
                "ocr_model": ("OCRMODEL",),  # 已加载模型
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "caption", "json_string")
    FUNCTION = "ocr"
    CATEGORY = "PaddleOCR"

    def ocr(
        self,
        image,
        ocr_model,
    ):
        image = image.permute(0, 3, 1, 2)
        processed_images = []
        out_results = []
        out_data = []
        for img in image:
            # 将张量转换为 PIL 图像
            image_pil = F.to_pil_image(img)
            cv_image = np.array(image_pil)
            result = ocr_model.ocr(cv_image)
            output_image = out2Img(
                cv_image, result, font_path=ocr_model.args.vis_font_path
            )
            output_image = torch.from_numpy(
                output_image.astype(np.float32) / 255.0
            ).unsqueeze(0)

            out_results.append(result)
            out_data.append(
                {
                    "boxes": [box[0] for box in result[0]],
                    "texts": [box[1][0] for box in result[0]],
                    "scores": [box[1][1] for box in result[0]],
                }
            )
            processed_images.append(output_image)

        output_images = torch.cat(processed_images, dim=0)

        json_string = json.dumps(out_data, ensure_ascii=False)
        return (output_images, out_results, json_string)
