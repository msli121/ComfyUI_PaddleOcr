# 当 Comfy 尝试导入模块时，会执行 __init__.py。为了使模块被识别为包含自定义节点定义，需要导出 NODE_CLASS_MAPPINGS。如果导入过程中没有出现任何问题，则模块中定义的节点将在 Comfy 中可用
# __init__.py 还可以导出 NODE_DISPLAY_NAME_MAPPINGS，它将相同的唯一名称映射到节点的显示名称。如果未提供 NODE_DISPLAY_NAME_MAPPINGS，Comfy 将使用唯一名称作为显示名称。
# 虽然可以手动安装自定义节点，但大多数人使用 ComfyUI Manager 来安装它们。 ComfyUI Manager 负责安装、更新和删除自定义节点以及任何依赖项。但它不是 Comfy 核心的一部分，因此您需要手动安装它。
from . import node

# NODE_CLASS_MAPPINGS 必须是将自定义节点名称映射到相应节点类的字典。
NODE_CLASS_MAPPINGS = {
    "My ModelDownload Node": node.DownloadAndLoadPaddleOcrModel,
    "My OcrRun Node": node.PaddleOcrRun,
}
__all__ = ["NODE_CLASS_MAPPINGS"]

# NODE_DISPLAY_NAME_MAPPINGS将相同的唯一名称映射到节点的显示名称。如果未提供 NODE_DISPLAY_NAME_MAPPINGS，Comfy 将使用唯一名称作为显示名称。
NODE_DISPLAY_NAME_MAPPINGS = {
    "My ModelDownload Node": "下载模型",
    "My OcrRun Node": "OCR识别",
}
