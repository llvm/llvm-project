#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#  This file contains the Nvgpu class.

from mlir import execution_engine
from mlir import ir
from mlir import passmanager
from typing import Sequence
import errno
import os
import sys

_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_SCRIPT_PATH)


class NvgpuCompiler:
    """Nvgpu class for compiling and building MLIR modules."""

    def __init__(self, options: str, opt_level: int, shared_libs: Sequence[str]):
        pipeline = f"builtin.module(gpu-lower-to-nvvm-pipeline{{{options}}})"
        self.pipeline = pipeline
        self.shared_libs = shared_libs
        self.opt_level = opt_level

    def __call__(self, module: ir.Module):
        """Convenience application method."""
        self.compile(module)

    def compile(self, module: ir.Module):
        """Compiles the module by invoking the nvgpu pipeline."""
        passmanager.PassManager.parse(self.pipeline).run(module.operation)

    def jit(self, module: ir.Module) -> execution_engine.ExecutionEngine:
        """Wraps the module in a JIT execution engine."""
        return execution_engine.ExecutionEngine(
            module, opt_level=self.opt_level, shared_libs=self.shared_libs
        )

    def compile_and_jit(self, module: ir.Module) -> execution_engine.ExecutionEngine:
        """Compiles and jits the module."""
        self.compile(module)
        return self.jit(module)
