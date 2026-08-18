"""Static regressions for the Qwen3.8 routed-expert MXFP4 checkpoint adapter."""

import ast
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=2, suite="base-a-test-cpu")


_REPO_ROOT = Path(__file__).resolve().parents[5]
_MODEL_CONFIG = _REPO_ROOT / "python/sglang/srt/configs/model_config.py"
_MXFP4 = _REPO_ROOT / "python/sglang/srt/layers/quantization/mxfp4.py"


def _source(path: Path) -> str:
    return path.read_text()


def _class_function(source: str, class_name: str, name: str) -> str:
    tree = ast.parse(source)
    lines = source.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if (
                    isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and child.name == name
                ):
                    return "\n".join(lines[child.lineno - 1 : child.end_lineno])
    raise AssertionError(f"{class_name}.{name} not found")


def _run_from_config(config):
    source = _source(_MXFP4)
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Mxfp4Config":
            function = next(
                child
                for child in node.body
                if isinstance(child, ast.FunctionDef) and child.name == "from_config"
            )
            function.decorator_list = []
            module = ast.fix_missing_locations(
                ast.Module(body=[function], type_ignores=[])
            )
            namespace = {"_is_hip": False}
            exec(compile(module, str(_MXFP4), "exec"), namespace)

            class Config:
                get_from_keys = staticmethod(
                    lambda values, keys: next(
                        values[key] for key in keys if key in values
                    )
                )
                get_from_keys_or = staticmethod(
                    lambda values, keys, default: next(
                        (values[key] for key in keys if key in values), default
                    )
                )

                def __init__(self, **kwargs):
                    self.__dict__.update(kwargs)

            return namespace["from_config"](Config, config)
    raise AssertionError("Mxfp4Config.from_config not found")


def test_qwen38_cli_mxfp4_accepts_checkpoint_mxfp8_metadata():
    source = _source(_MODEL_CONFIG)

    assert 'self.quantization == "mxfp4"' in source
    assert 'quant_method == "mxfp8"' in source
    assert 'quant_cfg.get("checkpoint_format")' in source
    assert '== "qwen38_routed_experts_mxfp4_v1"' in source
    assert "is_qwen38_mxfp4_compat" in source


def test_qwen38_mxfp4_config_normalizes_ignored_attention_layer():
    source = _source(_MXFP4)
    from_config = _class_function(source, "Mxfp4Config", "from_config")

    assert 'checkpoint_format == "qwen38_routed_experts_mxfp4_v1"' in from_config
    assert '"mxfp4" in quant_method or checkpoint_scales_are_fp32' in from_config
    assert '"modules_to_not_convert"' in from_config
    assert '"self_attn" if layer == "attn" else layer' in from_config
    assert "ignored_layers=ignored_layers" in from_config

    config = _run_from_config(
        {
            "quant_method": "mxfp8",
            "checkpoint_format": "qwen38_routed_experts_mxfp4_v1",
            "modules_to_not_convert": ["attn", "lm_head"],
        }
    )
    assert config.ignored_layers == ["self_attn", "lm_head"]
    assert config.is_checkpoint_mxfp4_serialized is True
    assert config.checkpoint_scales_are_fp32 is True


def test_regular_mxfp4_config_does_not_rewrite_ignored_attention_layer():
    config = _run_from_config(
        {
            "quant_method": "mxfp4",
            "modules_to_not_convert": ["attn"],
        }
    )
    assert config.ignored_layers == ["attn"]
    assert config.is_checkpoint_mxfp4_serialized is True
    assert config.checkpoint_scales_are_fp32 is False


def test_qwen38_mxfp4_allocates_fp32_checkpoint_scales():
    source = _source(_MXFP4)
    create_weights = _class_function(source, "Mxfp4MoEMethod", "create_weights")

    assert "torch.float32 if self.checkpoint_scales_are_fp32" in create_weights
    assert "scale_fill_value = 1.0 if self.checkpoint_scales_are_fp32" in create_weights
    assert create_weights.count("fill_value=scale_fill_value") == 2


def test_qwen38_mxfp4_does_not_reinterpret_fp32_scales():
    source = _source(_MXFP4)
    process = _class_function(
        source, "Mxfp4MoEMethod", "process_weights_after_loading"
    )

    assert "scale.data.dtype == torch.float32" in process
    assert "scale.data.view(torch.float8_e8m0fnu)" in process
    assert "transform_sf_into_required_layout(" in process
    assert "disable_ue8m0_cast=False" in process
