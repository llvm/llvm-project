import contextlib
import logging
import os
import sys
import tempfile
from contextlib import ExitStack
from io import StringIO
from typing import Optional, List, Union

from ..ir import StringAttr, Module
from ..passmanager import PassManager


@contextlib.contextmanager
def disable_multithreading(context=None):
    from ..ir import Context

    if context is None:
        context = Context.current

    context.enable_multithreading(False)
    yield
    context.enable_multithreading(True)


logger = logging.getLogger(__name__)


def get_module_name_for_debug_dump(module):
    if "debug_module_name" not in module.operation.attributes:
        return "UnnammedMLIRModule"
    return StringAttr(module.operation.attributes["debug_module_name"]).value


def run_pipeline(
    module,
    pipeline: Union[str, "Pipeline"],
    description: Optional[str] = None,
    enable_ir_printing=False,
    print_pipeline=False,
    verify=True,
):
    module = Module.parse(str(module))
    if isinstance(pipeline, Pipeline):
        pipeline = str(pipeline)
    module_name = get_module_name_for_debug_dump(module)
    try:
        original_stderr = sys.stderr
        sys.stderr = StringIO()
        with ExitStack() as stack:
            stack.enter_context(module.context)
            asm_for_error_report = module.operation.get_asm(
                large_elements_limit=10, enable_debug_info=True
            )
            pm = PassManager.parse(pipeline)
            pm.enable_verifier(verify)
            if print_pipeline:
                print(pm)
            if enable_ir_printing:
                stack.enter_context(disable_multithreading())
                pm.enable_ir_printing()
            pm.run(module.operation)
    except Exception as e:
        print(e, file=sys.stderr)
        filename = os.path.join(tempfile.gettempdir(), module_name + ".mlir")
        with open(filename, "w") as f:
            f.write(asm_for_error_report)
        debug_options = "-mlir-print-ir-after-all -mlir-disable-threading"
        description = description or f"{module_name} compile"

        message = f"""\
            {description} failed with the following diagnostics:

            {'*' * 80}
            {sys.stderr.getvalue().strip()}
            {'*' * 80}

            For developers, the error can be reproduced with:
            $ mlir-opt {debug_options} -pass-pipeline='{pipeline}' {filename}
            """
        trimmed_message = "\n".join([m.lstrip() for m in message.split("\n")])
        raise RuntimeError(trimmed_message)
    finally:
        sys.stderr = original_stderr

    return module


class Pipeline:
    _pipeline: List[str] = []

    def __init__(self, pipeline=None, wrapper=None):
        if pipeline is None:
            pipeline = []
        self._pipeline = pipeline

    def Nested(self, context, p: "Pipeline"):
        self._pipeline.append(f"{context}({p.materialize(module=False)})")
        return self

    def Func(self, p: "Pipeline"):
        return self.Nested("func.func", p)

    def Spirv(self, p: "Pipeline"):
        return self.Nested("spirv.module", p)

    def Gpu(self, p: "Pipeline"):
        assert isinstance(p, Pipeline)
        return self.Nested("gpu.module", p)

    def materialize(self, module=True):
        pipeline_str = ",".join(self._pipeline)
        if module:
            pipeline_str = f"builtin.module({pipeline_str})"
        logger.debug(f"{pipeline_str}")
        return pipeline_str

    def __str__(self):
        return self.materialize()

    def __iadd__(self, other: "Pipeline"):
        self._pipeline.extend(other._pipeline)
        return self

    def __add__(self, other: "Pipeline"):
        return Pipeline(self._pipeline + other._pipeline)

    def add_pass(self, pass_name, **kwargs):
        kwargs = {
            k.replace("_", "-"): int(v) if isinstance(v, bool) else v
            for k, v in kwargs.items()
            if v is not None
        }
        if kwargs:
            args_str = " ".join(f"{k}={v}" for k, v in kwargs.items())
            pass_str = f"{pass_name}{{ {args_str} }}"
        else:
            pass_str = f"{pass_name}"
        self._pipeline.append(pass_str)
        return self
