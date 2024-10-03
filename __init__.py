from .bfl_api import FluxPro, FluxDev, FluxPro11

NODE_CLASS_MAPPINGS = {
    "FLUX .1 [pro]": FluxPro,
    "FLUX .1 [dev]": FluxDev,
    "FLUX 1.1 [pro]": FluxPro11,
}
