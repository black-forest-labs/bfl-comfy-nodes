import os
import time
from io import BytesIO

import numpy as np
import requests
import torch
from PIL import Image
from requests.models import PreparedRequest
import base64

ROOT_API = "https://api.bfl.ml/"
US1_API = "https://api.us1.bfl.ai/"
API_KEY = os.environ.get("BFL_API_KEY")


def get_api_key():
    global API_KEY
    if API_KEY:
        return API_KEY

    dir_path = os.path.dirname(os.path.realpath(__file__))
    key_file_path = os.path.join(dir_path, "bfl_api_key.txt")

    if os.path.exists(key_file_path):
        with open(key_file_path, "r") as f:
            API_KEY = f.read().strip()
            if API_KEY:
                return API_KEY

    return None


class FluxBase:
    API_ENDPOINT = ""
    POLL_ENDPOINT = ""
    ACCEPT = ""

    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = cls.INPUT_SPEC.copy()
        if "optional" not in base_inputs:
            base_inputs["optional"] = {}
        base_inputs["optional"]["region"] = (["EU1", "US1"], {"default": "EU1"})
        return base_inputs

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "call"
    CATEGORY = "Flux"

    def call(self, *args, **kwargs):
        data = {k: v for k, v in kwargs.items() if k != "region"}
        headers = {
            "Accept": self.ACCEPT,
            "x-key": kwargs.get("api_key_override") or get_api_key(),
        }
        if headers["x-key"] is None:
            raise Exception(
                "No Black Forest Labs API key set. Set environment variable BFL_API_KEY, insert key into bfl_api_key.txt, or through node field 'api_key_override'"
            )
        response = self._make_request(
            headers, data, files=None, region=kwargs.get("region", "EU1")
        )

        if response.status_code == 200:
            return self._handle_response(response, headers, kwargs.get("region", "EU1"))
        else:
            error_info = response.json()
            raise Exception(f"BFL API Message: {error_info}")

    def _make_request(self, headers, data, files, region):
        req = PreparedRequest()
        req.prepare_method("POST")
        base_url = US1_API if region == "US1" else ROOT_API
        req.prepare_url(f"{base_url}{self.API_ENDPOINT}", None)
        req.prepare_headers(headers)
        if files:
            req.prepare_body(data=data, files=files)
        else:
            req.prepare_body(data=None, files=None, json=data)
        return requests.Session().send(req)

    def _handle_response(self, response, headers, region="EU1"):
        if self.POLL_ENDPOINT:
            return self._poll_for_result(response.json().get("id"), headers, region)
        else:
            return self._process_image_response(response)

    def _poll_for_result(self, id, headers, region="EU1"):
        timeout, start_time = 240, time.time()
        retries = 0
        max_retries = 0
        base_url = US1_API if region == "US1" else ROOT_API
        while True:
            response = requests.get(
                f"{base_url}{self.POLL_ENDPOINT}", params={"id": id}, headers=headers
            )
            if response.status_code == 200:
                result = response.json()
                if result["status"] == "Ready":
                    image_url = result["result"]["sample"]
                    image_response = requests.get(image_url)
                    return self._process_image_response(image_response)
                elif result["status"] in ["Request Moderated", "Content Moderated"]:
                    raise Exception(f"BFL API Message: {result['status']}")
                elif result["status"] == "Error":
                    raise Exception(f"BFL API Error: {result}")
            elif response.status_code == 404:
                if retries < max_retries:
                    retries += 1
                    time.sleep(5)
                    continue
                raise Exception(
                    f"BFL API Error: Task not found after {max_retries} retries"
                )
            elif response.status_code == 202:
                time.sleep(10)
            elif time.time() - start_time > timeout:
                raise Exception("BFL API Timeout: Request took too long to complete")
            else:
                raise Exception(f"BFL API Error: {response.json()}")

    def _process_image_response(self, response):
        image = Image.open(BytesIO(response.content)).convert("RGBA")
        image_array = np.array(image).astype(np.float32) / 255.0
        return (torch.from_numpy(image_array)[None,],)

    def _convert_image_to_base64(self, image):
        if isinstance(image, torch.Tensor):
            image = Image.fromarray((image[0].cpu().numpy() * 255).astype(np.uint8))
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()
        return image

    def _convert_mask_to_base64(self, mask):
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy()
            if len(mask_np.shape) == 3:
                mask_np = mask_np[0]
            mask = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")
            buffered = BytesIO()
            mask.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()
        return mask


class Flux(FluxBase):
    ACCEPT = "image/*"
    API_ENDPOINT = "v1/flux-dev"
    INPUT_SPEC = {
        "required": {
            "accept": (["image/*"], {"default": "image/*"}),
            "api_endpoint": ([
                "v1/flux-dev",
                "v1/flux-pro",
                "v1/flux-pro-1.0-fill",
                "v1/flux-pro-1.0-canny",
                "v1/flux-pro-1.0-depth",
                "v1/flux-pro-1.1",
                "v1/flux-pro-1.1-ultra",
                "v1/flux-pro-finetuned",
                "v1/flux-pro-1.0-canny-finetuned",
                "v1/flux-pro-1.0-depth-finetuned",
                "v1/flux-pro-1.0-fill-finetuned",
                "v1/flux-pro-1.1-ultra-finetuned",
            ], {"default": "v1/flux-dev"}),
            "poll_endpoint": (["v1/get_result"], {"default": "v1/get_result"}),
            "prompt": ("STRING", {"multiline": True}),
        },
        "optional": {
            "api_key_override": ("STRING", {"multiline": False}),
            "aspect_ratio": ("STRING", {"default": "1:1"}),
            "control_image": ("IMAGE",),
            "guidance": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 100, "step": 0.01}),
            "finetune_id": ("STRING", {"multiline": False}),
            "finetune_strength": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 2.0}),
            "height": ("INT", {"default": 1024, "min": 0, "max": 1440, "step": 32}),
            "high_threshold": ("INT", {"default": 200, "min": 0, "max": 500}),
            "image_prompt": ("IMAGE",),
            "image_prompt_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "interval": ("INT", {"default": 1, "min": 1, "max": 10}),
            "mask": ("MASK",),
            "low_threshold": ("INT", {"default": 50, "min": 0, "max": 500}),
            "output_format": (["jpeg", "png"], {"default": "png"}),
            "prompt_upsampling": ("BOOLEAN", {"default": False, "label_on": "True", "label_off": "False"}),
            "raw": ("BOOLEAN", {"default": False, "label_on": "True", "label_off": "False"}),
            "safety_tolerance": ("INT", {"default": 2, "min": 0, "max": 6, "step": 1}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 4294967294}),
            "steps": ("INT", {"default": 50, "min": 10, "max": 100}),
            "webhook_secret": ("STRING", {"multilne": False}),
            "webhook_url": ("STRING", {"max": 2083, "multiline": False}),
            "width": ("INT", {"default": 1024, "min": 0, "max": 1440, "step": 32}),
        },
    }
    POLL_ENDPOINT = "v1/get_result"
    RETURN_TYPES = ("IMAGE",)

    def call(self, *args, **kwargs):
        if "control_image" in kwargs:
            kwargs["control_image"] = self._convert_image_to_base64(kwargs["control_image"])
        if "high_threshold" in kwargs:
            kwargs["canny_high_threshold"] = int(kwargs.pop("high_threshold"))
        if "image" in kwargs:
            kwargs["image"] = self._convert_image_to_base64(kwargs["image"])
        if "image_prompt" in kwargs and kwargs["image_prompt"] is not None:
            kwargs["image_prompt"] = self._convert_image_to_base64(kwargs["image_prompt"])
        if "low_threshold" in kwargs:
            kwargs["canny_low_threshold"] = int(kwargs.pop("low_threshold"))
        if "mask" in kwargs and kwargs["mask"] is not None:
            kwargs["mask"] = self._convert_mask_to_base64(kwargs["mask"])
        return super().call(*args, **kwargs)


class FluxPro(FluxBase):
    API_ENDPOINT = "v1/flux-pro"
    POLL_ENDPOINT = "v1/get_result"
    ACCEPT = "image/*"
    INPUT_SPEC = {
        "required": {
            "prompt": ("STRING", {"multiline": True}),
        },
        "optional": {
            "seed": ("INT", {"default": 0, "min": 0, "max": 4294967294}),
            "guidance": ("FLOAT", {"default": 2.5, "min": 1.5, "max": 5, "step": 0.01}),
            "width": (
                "INT",
                {"default": 1024, "min": 0, "max": 1440, "step": 32},
            ),
            "height": (
                "INT",
                {"default": 1024, "min": 0, "max": 1440, "step": 32},
            ),
            "steps": ("INT", {"default": 50, "min": 10, "max": 100}),
            "interval": ("INT", {"default": 1, "min": 1, "max": 10}),
            "prompt_upsampling": (
                "BOOLEAN",
                {"default": True, "label_on": "True", "label_off": "False"},
            ),
            "safety_tolerance": ("INT", {"default": 2, "min": 0, "max": 6}),
            "api_key_override": ("STRING", {"multiline": False}),
            "image_prompt": ("IMAGE",),
        },
    }

    def call(self, *args, **kwargs):
        if "image_prompt" in kwargs and kwargs["image_prompt"] is not None:
            kwargs["image_prompt"] = self._convert_image_to_base64(
                kwargs["image_prompt"]
            )
        return super().call(*args, **kwargs)


class FluxDev(FluxBase):
    API_ENDPOINT = "v1/flux-dev"
    POLL_ENDPOINT = "v1/get_result"
    ACCEPT = "image/*"
    INPUT_SPEC = {
        "required": {
            "prompt": ("STRING", {"multiline": True}),
        },
        "optional": {
            "seed": ("INT", {"default": 0, "min": 0, "max": 4294967294}),
            "guidance": ("FLOAT", {"default": 2.5, "min": 1.5, "max": 5, "step": 0.01}),
            "width": (
                "INT",
                {"default": 1024, "min": 0, "max": 1440, "step": 32},
            ),
            "height": (
                "INT",
                {"default": 1024, "min": 0, "max": 1440, "step": 32},
            ),
            "steps": ("INT", {"default": 50, "min": 10, "max": 100}),
            "interval": ("INT", {"default": 1, "min": 1, "max": 10}),
            "prompt_upsampling": (
                "BOOLEAN",
                {"default": True, "label_on": "True", "label_off": "False"},
            ),
            "safety_tolerance": ("INT", {"default": 2, "min": 0, "max": 6}),
            "api_key_override": ("STRING", {"multiline": False}),
            "image_prompt": ("IMAGE",),
        },
    }

    def call(self, *args, **kwargs):
        if "image_prompt" in kwargs and kwargs["image_prompt"] is not None:
            kwargs["image_prompt"] = self._convert_image_to_base64(
                kwargs["image_prompt"]
            )
        return super().call(*args, **kwargs)


class FluxPro11(FluxBase):
    API_ENDPOINT = "v1/flux-pro-1.1"
    POLL_ENDPOINT = "v1/get_result"
    ACCEPT = "image/*"
    INPUT_SPEC = {
        "required": {
            "prompt": ("STRING", {"multiline": True}),
        },
        "optional": {
            "seed": ("INT", {"default": 0, "min": 0, "max": 4294967294}),
            "guidance": ("FLOAT", {"default": 2.5, "min": 1.5, "max": 5, "step": 0.01}),
            "width": (
                "INT",
                {"default": 1024, "min": 0, "max": 1440, "step": 32},
            ),
            "height": (
                "INT",
                {"default": 1024, "min": 0, "max": 1440, "step": 32},
            ),
            "interval": ("INT", {"default": 1, "min": 1, "max": 10}),
            "prompt_upsampling": (
                "BOOLEAN",
                {"default": True, "label_on": "True", "label_off": "False"},
            ),
            "safety_tolerance": ("INT", {"default": 2, "min": 0, "max": 6}),
            "api_key_override": ("STRING", {"multiline": False}),
            "image_prompt": ("IMAGE",),
        },
    }

    def call(self, *args, **kwargs):
        if "image_prompt" in kwargs and kwargs["image_prompt"] is not None:
            kwargs["image_prompt"] = self._convert_image_to_base64(
                kwargs["image_prompt"]
            )
        return super().call(*args, **kwargs)


class FluxProFill(FluxBase):
    API_ENDPOINT = "v1/flux-pro-1.0-fill"
    POLL_ENDPOINT = "v1/get_result"
    ACCEPT = "image/*"
    INPUT_SPEC = {
        "required": {
            "image": ("IMAGE",),
            "prompt": ("STRING", {"multiline": True}),
        },
        "optional": {
            "mask": ("MASK",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 4294967294}),
            "guidance": (
                "FLOAT",
                {"default": 60.0, "min": 1.5, "max": 100, "step": 0.1},
            ),
            "steps": ("INT", {"default": 50, "min": 15, "max": 50}),
            "prompt_upsampling": ("BOOLEAN", {"default": False}),
            "safety_tolerance": ("INT", {"default": 2, "min": 0, "max": 6}),
            "api_key_override": ("STRING", {"multiline": False}),
        },
    }

    def call(self, *args, **kwargs):
        if "image" in kwargs:
            kwargs["image"] = self._convert_image_to_base64(kwargs["image"])
        if "mask" in kwargs and kwargs["mask"] is not None:
            kwargs["mask"] = self._convert_mask_to_base64(kwargs["mask"])
        return super().call(*args, **kwargs)


class FluxCanny(FluxBase):
    API_ENDPOINT = "v1/flux-pro-1.0-canny"
    POLL_ENDPOINT = "v1/get_result"
    ACCEPT = "image/*"
    INPUT_SPEC = {
        "required": {
            "prompt": ("STRING", {"multiline": True}),
            "control_image": ("IMAGE",),
        },
        "optional": {
            "seed": ("INT", {"default": 0, "min": 0, "max": 4294967294}),
            "guidance": (
                "FLOAT",
                {"default": 30.0, "min": 1.0, "max": 100, "step": 0.1},
            ),
            "steps": ("INT", {"default": 50, "min": 15, "max": 50}),
            "low_threshold": ("INT", {"default": 50, "min": 0, "max": 500}),
            "high_threshold": ("INT", {"default": 200, "min": 0, "max": 500}),
            "prompt_upsampling": ("BOOLEAN", {"default": False}),
            "safety_tolerance": ("INT", {"default": 2, "min": 0, "max": 6}),
            "api_key_override": ("STRING", {"multiline": False}),
        },
    }

    def call(self, *args, **kwargs):
        if "control_image" in kwargs:
            kwargs["control_image"] = self._convert_image_to_base64(
                kwargs["control_image"]
            )
        if "low_threshold" in kwargs:
            kwargs["canny_low_threshold"] = int(kwargs.pop("low_threshold"))
        if "high_threshold" in kwargs:
            kwargs["canny_high_threshold"] = int(kwargs.pop("high_threshold"))
        return super().call(*args, **kwargs)


class FluxDepth(FluxBase):
    API_ENDPOINT = "v1/flux-pro-1.0-depth"
    POLL_ENDPOINT = "v1/get_result"
    ACCEPT = "image/*"
    INPUT_SPEC = {
        "required": {
            "prompt": ("STRING", {"multiline": True}),
            "control_image": ("IMAGE",),
        },
        "optional": {
            "seed": ("INT", {"default": 0, "min": 0, "max": 4294967294}),
            "guidance": (
                "FLOAT",
                {"default": 15.0, "min": 1.0, "max": 100, "step": 0.1},
            ),
            "steps": ("INT", {"default": 50, "min": 15, "max": 50}),
            "prompt_upsampling": ("BOOLEAN", {"default": False}),
            "safety_tolerance": ("INT", {"default": 2, "min": 0, "max": 6}),
            "api_key_override": ("STRING", {"multiline": False}),
        },
    }

    def call(self, *args, **kwargs):
        if "control_image" in kwargs:
            kwargs["control_image"] = self._convert_image_to_base64(
                kwargs["control_image"]
            )
        return super().call(*args, **kwargs)


class FluxUltra11(FluxBase):
    API_ENDPOINT = "v1/flux-pro-1.1-ultra"
    POLL_ENDPOINT = "v1/get_result"
    ACCEPT = "image/*"

    INPUT_SPEC = {
        "required": {
            "prompt": ("STRING", {"multiline": True}),
        },
        "optional": {
            "image_prompt": ("IMAGE",),
            "aspect_ratio": ("STRING", {"default": "1:1"}),
            "raw": (
                "BOOLEAN",
                {"default": False, "label_on": "True", "label_off": "False"},
            ),
            "safety_tolerance": ("INT", {"default": 5, "min": 1, "max": 5, "step": 1}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 4294967294}),
            "image_prompt_strength": (
                "FLOAT",
                {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
            ),
            "output_format": ("STRING", {"default": "png"}),
            "api_key_override": ("STRING", {"multiline": False}),
        },
    }

    def call(self, *args, **kwargs):
        if "image_prompt" in kwargs and kwargs["image_prompt"] is not None:
            kwargs["image_prompt"] = self._convert_image_to_base64(
                kwargs["image_prompt"]
            )
        return super().call(*args, **kwargs)


class FluxProFinetune(FluxBase):
    API_ENDPOINT = "v1/flux-pro-finetuned"
    POLL_ENDPOINT = "v1/get_result"
    ACCEPT = "image/*"
    CATEGORY = "Flux Finetuned"
    INPUT_SPEC = {
        "required": {
            "prompt": ("STRING", {"multiline": True}),
            "finetune_id": ("STRING", {"multiline": False}),
        },
        "optional": {
            "finetune_strength": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 2.0}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 4294967294}),
            "guidance": ("FLOAT", {"default": 2.5, "min": 1.5, "max": 5.0}),
            "steps": ("INT", {"default": 40, "min": 1, "max": 50}),
            "prompt_upsampling": ("BOOLEAN", {"default": False}),
            "width": ("INT", {"default": 1024, "min": 256, "max": 1440, "step": 32}),
            "height": ("INT", {"default": 768, "min": 256, "max": 1440, "step": 32}),
            "safety_tolerance": ("INT", {"default": 2, "min": 0, "max": 6}),
            "output_format": ("STRING", {"default": "jpeg"}),
            "image_prompt": ("IMAGE",),
            "image_prompt_strength": (
                "FLOAT",
                {"default": 0.1, "min": 0.0, "max": 1.0},
            ),
        },
    }

    def call(self, *args, **kwargs):
        if "image_prompt" in kwargs and kwargs["image_prompt"] is not None:
            kwargs["image_prompt"] = self._convert_image_to_base64(
                kwargs["image_prompt"]
            )
        return super().call(*args, **kwargs)


class FluxProCannyFinetune(FluxBase):
    API_ENDPOINT = "v1/flux-pro-1.0-canny-finetuned"
    POLL_ENDPOINT = "v1/get_result"
    ACCEPT = "image/*"
    CATEGORY = "Flux Finetuned"
    INPUT_SPEC = {
        "required": {
            "prompt": ("STRING", {"multiline": True}),
            "control_image": ("IMAGE",),
            "finetune_id": ("STRING", {"multiline": False}),
        },
        "optional": {
            "finetune_strength": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 2.0}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 4294967294}),
            "guidance": (
                "FLOAT",
                {"default": 30.0, "min": 1.0, "max": 100, "step": 0.1},
            ),
            "steps": ("INT", {"default": 50, "min": 15, "max": 50}),
            "prompt_upsampling": ("BOOLEAN", {"default": False}),
            "safety_tolerance": ("INT", {"default": 2, "min": 0, "max": 6}),
            "output_format": ("STRING", {"default": "jpeg"}),
        },
    }

    def call(self, *args, **kwargs):
        if "control_image" in kwargs:
            kwargs["control_image"] = self._convert_image_to_base64(
                kwargs["control_image"]
            )
        return super().call(*args, **kwargs)


class FluxProDepthFinetune(FluxBase):
    API_ENDPOINT = "v1/flux-pro-1.0-depth-finetuned"
    POLL_ENDPOINT = "v1/get_result"
    ACCEPT = "image/*"
    CATEGORY = "Flux Finetuned"
    INPUT_SPEC = {
        "required": {
            "prompt": ("STRING", {"multiline": True}),
            "control_image": ("IMAGE",),
            "finetune_id": ("STRING", {"multiline": False}),
        },
        "optional": {
            "finetune_strength": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 2.0}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 4294967294}),
            "guidance": (
                "FLOAT",
                {"default": 15.0, "min": 1.0, "max": 100, "step": 0.1},
            ),
            "steps": ("INT", {"default": 50, "min": 15, "max": 50}),
            "prompt_upsampling": ("BOOLEAN", {"default": False}),
            "safety_tolerance": ("INT", {"default": 2, "min": 0, "max": 6}),
            "output_format": ("STRING", {"default": "jpeg"}),
        },
    }

    def call(self, *args, **kwargs):
        if "control_image" in kwargs:
            kwargs["control_image"] = self._convert_image_to_base64(
                kwargs["control_image"]
            )
        return super().call(*args, **kwargs)


class FluxProFillFinetune(FluxBase):
    API_ENDPOINT = "v1/flux-pro-1.0-fill-finetuned"
    POLL_ENDPOINT = "v1/get_result"
    ACCEPT = "image/*"
    INPUT_SPEC = {
        "required": {
            "image": ("IMAGE",),
            "finetune_id": ("STRING", {"multiline": False}),
            "prompt": ("STRING", {"multiline": True}),
        },
        "optional": {
            "mask": ("MASK",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 4294967294}),
            "guidance": (
                "FLOAT",
                {"default": 60.0, "min": 1.5, "max": 100, "step": 0.1},
            ),
            "steps": ("INT", {"default": 50, "min": 15, "max": 50}),
            "prompt_upsampling": ("BOOLEAN", {"default": False}),
            "safety_tolerance": ("INT", {"default": 2, "min": 0, "max": 6}),
            "api_key_override": ("STRING", {"multiline": False}),
        },
    }

    def call(self, *args, **kwargs):
        if "image" in kwargs:
            kwargs["image"] = self._convert_image_to_base64(kwargs["image"])
        if "mask" in kwargs and kwargs["mask"] is not None:
            kwargs["mask"] = self._convert_mask_to_base64(kwargs["mask"])
        return super().call(*args, **kwargs)


class FluxUltra11Finetune(FluxBase):
    API_ENDPOINT = "v1/flux-pro-1.1-ultra-finetuned"
    POLL_ENDPOINT = "v1/get_result"
    ACCEPT = "image/*"

    INPUT_SPEC = {
        "required": {
            "prompt": ("STRING", {"multiline": True}),
            "finetune_id": ("STRING", {"multiline": False}),
        },
        "optional": {
            "finetune_strength": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 2.0}),
            "image_prompt": ("IMAGE",),
            "aspect_ratio": ("STRING", {"default": "1:1"}),
            "raw": (
                "BOOLEAN",
                {"default": False, "label_on": "True", "label_off": "False"},
            ),
            "safety_tolerance": ("INT", {"default": 5, "min": 1, "max": 5, "step": 1}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 4294967294}),
            "image_prompt_strength": (
                "FLOAT",
                {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
            ),
            "output_format": ("STRING", {"default": "png"}),
            "api_key_override": ("STRING", {"multiline": False}),
        },
    }

    def call(self, *args, **kwargs):
        if "image_prompt" in kwargs and kwargs["image_prompt"] is not None:
            kwargs["image_prompt"] = self._convert_image_to_base64(
                kwargs["image_prompt"]
            )
        return super().call(*args, **kwargs)
