import torch
import habana_frameworks.torch.core as htcore

from torchvision import models
from torch.fx.passes.graph_drawer import FxGraphDrawer
from IPython.display import Markdown as md
import torch._dynamo
from functorch.compile import make_boxed_func
from torch._functorch.aot_autograd import aot_module_simplified
from torch._decomp import core_aten_decompositions


decompositions= core_aten_decompositions()
decompositions.update(
    torch._decomp.get_decompositions([
        torch.ops.aten.sin,
        torch.ops.aten.cos,
        torch.ops.aten.add,
        torch.ops.aten.sub,
        torch.ops.aten.mul,
        torch.ops.aten.sum,
        torch.ops.aten.mean,
        torch.ops.aten.pow.Tensor_Scalar
    ])
)

def inspect_backend(gm, sample_inputs): 
    # Forward compiler capture
    def fw(gm, sample_inputs):
        gm.print_readable()
        g = FxGraphDrawer(gm, 'fn')
        with open("forward_aot.svg", "wb") as file:
            file.write(g.get_dot_graph().create_svg())
        return make_boxed_func(gm.forward)
    
    # Backward compiler capture
    def bw(gm, sample_inputs):
        gm.print_readable()
        g = FxGraphDrawer(gm, 'fn')
        with open("backward_aot.svg", "wb") as file:
            file.write(g.get_dot_graph().create_svg())
        return make_boxed_func(gm.forward)
    
    # Call AOTAutograd
    gm_forward = aot_module_simplified(gm,sample_inputs,
                                       fw_compiler=fw,
                                       bw_compiler=bw,
                                       decompositions=decompositions)

    return gm_forward

def f(x):
    return torch.sin(x)**2 + torch.cos(x)**2

def f_loss(x,y):
    f_x = torch.sin(x)**2 + torch.cos(x)**2
    return torch.nn.functional.mse_loss(f_x, y)

torch.manual_seed(0)
x = torch.rand(1000, requires_grad=True).to('hpu')
y = torch.ones_like(x)

torch._dynamo.reset()
compiled_f = torch.compile(f_loss, backend=inspect_backend)
out = compiled_f(x,y).backward()