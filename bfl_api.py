import requests
from requests.models import PreparedRequest
from PIL import Image
import numpy as np
import torch
from torchvision.transforms import ToPILImage
from io import BytesIO
import os
import time

ROOT_API = "https://api.bfl.ml/"
API_KEY = os.environ.get("BFL_API_KEY")


def get_api_key():
    global API_KEY
    if API_KEY:
        return API_KEY

    dir_path = os.path.dirname(os.path.realpath(__file__))
    key_file_path = os.path.join(dir_path, "bfl_api_key.txt")
    error_message = (
        "API Key is required to use the BFL API. "
        f"Please set the BFL_API_KEY environment variable to your API key "
        f"or place it in {key_file_path}."
    )

    try:
        with open(key_file_path, "r") as f:
            API_KEY = f.read().strip()
        if not API_KEY:
            raise ValueError(error_message)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n\n***{error_message}***\n\n")
        raise

    return API_KEY


class FluxBase:
    API_ENDPOINT = ""
    POLL_ENDPOINT = ""
    ACCEPT = ""

    @classmethod
    def INPUT_TYPES(cls):
        return cls.INPUT_SPEC

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "call"
    CATEGORY = "Flux"

    def call(self, *args, **kwargs):
        data = {k: v for k, v in kwargs.items()}
        headers = {
            "Accept": self.ACCEPT,
            "x-key": kwargs.get("api_key_override") or get_api_key(),
        }
        if headers["x-key"] is None:
            raise Exception(
                "No Black Forest Labs API key set. Set environment variable BFL_API_KEY, insert key into bfl_api_key.txt, or through node field 'api_key_override'"
            )
        response = self._make_request(headers, data, files=None)

        if response.status_code == 200:
            return self._handle_response(response, headers)
        else:
            error_info = response.json()
            raise Exception(f"BFL API Message: {error_info}")

    def _make_request(self, headers, data, files):
        req = PreparedRequest()
        req.prepare_method("POST")
        req.prepare_url(f"{ROOT_API}{self.API_ENDPOINT}", None)
        req.prepare_headers(headers)
        if files:
            req.prepare_body(data=data, files=files)
        else:
            req.prepare_body(data=None, files=None, json=data)
        return requests.Session().send(req)

    def _handle_response(self, response, headers):
        if self.POLL_ENDPOINT:
            return self._poll_for_result(response.json().get("id"), headers)
        else:
            return self._process_image_response(response)

    def _poll_for_result(self, id, headers):
        timeout, start_time = 240, time.time()
        while True:
            response = requests.get(
                f"{ROOT_API}{self.POLL_ENDPOINT}", params={"id": id}, headers=headers
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
        },
    }


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
        },
    }


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
        },
    }


NODE_CLASS_MAPPINGS = {
    "FLUX .1 [pro]": FluxPro,
    "FLUX .1 [dev]": FluxDev,
    "FLUX 1.1 [pro]": FluxPro11,
}
