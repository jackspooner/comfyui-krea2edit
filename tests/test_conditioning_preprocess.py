import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn

from comfy.cli_args import args

args.cpu = True

import comfy.conds
from comfy.ldm.krea2.model import SingleStreamDiT
from comfy.model_patcher import ModelPatcher
from comfy.patcher_extension import CallbacksMP, WrappersMP


spec = importlib.util.spec_from_file_location("krea2edit_efficiency", Path(__file__).parents[1] / "__init__.py")
edit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(edit)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.current_patcher = None
        self.diffusion_model = SingleStreamDiT(features=64, heads=4, kvheads=2, layers=2,
            txtdim=32, txtheads=2, txtkvheads=1, txtlayers=3, channels=4, multiplier=1, operations=nn)
        for param in self.parameters():
            param.detach().uniform_(-0.1, 0.1)

    def process_latent_in(self, x):
        return x

    def get_dtype_inference(self):
        return torch.float32

    def extra_conds(self, **kwargs):
        return {"c_crossattn": comfy.conds.CONDRegular(kwargs["cross_attn"]),
                "other_condition": comfy.conds.CONDConstant("preserved")}


@pytest.fixture
def patched():
    torch.manual_seed(321)
    model = Model()
    source = torch.randn(1, 4, 4, 6)
    base = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
    patcher, = edit.Krea2EditModelPatch().patch(base, {"samples": source}, fit_mode="crop (legacy)")
    patcher.patch_model(load_weights=False)
    patcher.pre_run()
    yield model, patcher, source
    patcher.cleanup()
    patcher.unpatch_model(unpatch_weights=False)


@pytest.mark.parametrize("temporal", [False, True])
def test_fusion_once_per_condition_matches_each_raw_step(patched, temporal):
    model, patcher, source = patched
    dm = model.diffusion_model
    raw = torch.randn(1, 5, 96)
    calls = []
    handle = dm.txtfusion.register_forward_hook(lambda *unused: calls.append(1))
    conds = model.extra_conds(cross_attn=raw, device="cpu")
    assert len(calls) == 1
    assert conds["other_condition"].cond == "preserved"
    assert conds["krea2_context_preprocessed"].cond is True
    fused = conds["c_crossattn"].process_cond(batch_size=2).cond
    options = patcher.model_options["transformer_options"]
    x = torch.randn(2, 4, 5, 7)
    if temporal:
        x = x.unsqueeze(2)
    lengths = []
    dm.last.register_forward_pre_hook(lambda module, inputs: lengths.append(inputs[0].shape[1]))
    for sigma in [0.8, 0.4, 0.1]:
        t = torch.full((2,), sigma)
        before = len(calls)
        actual = dm(x, t, fused, transformer_options=options, krea2_context_preprocessed=True)
        assert len(calls) == before
        expected = edit.krea2_edit_forward(dm, x, t, raw.expand(2, -1, -1), source, {})
        torch.testing.assert_close(actual, expected)
    assert set(lengths) == {12}
    handle.remove()

    # A new scheduled condition and a later run both receive fresh fused values.
    other = model.extra_conds(cross_attn=raw + 1, device="cpu")
    assert not torch.equal(other["c_crossattn"].cond, conds["c_crossattn"].cond)
    again = model.extra_conds(cross_attn=raw, device="cpu")
    torch.testing.assert_close(again["c_crossattn"].cond, conds["c_crossattn"].cond)
    assert again["c_crossattn"].cond.data_ptr() != conds["c_crossattn"].cond.data_ptr()


@pytest.mark.parametrize("extension", ["hooks", "registered_hooks", "forced_hooks", "patches", "patches_replace", "optimized_attention_override", "callback", "hook_callback", "wrapper", "sampler_wrapper"])
def test_dynamic_extensions_keep_per_step_fusion(patched, extension):
    model, patcher, source = patched
    raw = torch.randn(1, 5, 96)
    kwargs = {}
    options = patcher.model_options["transformer_options"]
    if extension == "hooks":
        kwargs["hooks"] = object()
    elif extension == "registered_hooks":
        patcher.hook_patches["test"] = {}
    elif extension == "forced_hooks":
        patcher.forced_hooks = object()
    elif extension == "callback":
        patcher.add_callback(CallbacksMP.ON_PREPARE_STATE, lambda *unused: None)
    elif extension == "hook_callback":
        patcher.add_callback(CallbacksMP.ON_APPLY_HOOKS, lambda *unused: None)
    elif extension == "wrapper":
        options["wrappers"][WrappersMP.DIFFUSION_MODEL]["other"] = [lambda *unused: None]
    elif extension == "sampler_wrapper":
        options["wrappers"][WrappersMP.SAMPLER_SAMPLE] = {"other": [lambda *unused: None]}
    elif extension == "patches":
        options[extension] = {"attn1_patch": [lambda *unused: None]}
    elif extension == "patches_replace":
        options[extension] = {"dit": {("single", 0): lambda *unused: None}}
    else:
        options[extension] = lambda *unused: None
    conds = model.extra_conds(cross_attn=raw, device="cpu", **kwargs)
    assert conds["c_crossattn"].cond is raw
    assert "krea2_context_preprocessed" not in conds
    patcher.forced_hooks = None
    patcher.hook_patches.clear()
