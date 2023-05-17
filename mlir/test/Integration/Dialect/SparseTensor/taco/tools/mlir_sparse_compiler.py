#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#  This file contains the sparse compiler class. It is copied from
#  test/Integration/Dialect/SparseTensor/python/ until we have a better
#  solution.

from mlir import execution_engine
from mlir import ir
from mlir import passmanager
from typing import Sequence


class SparseCompiler:
    """Sparse compiler class for compiling and building MLIR modules."""

    def __init__(self, options: str, opt_level: int, shared_libs: Sequence[str]):
        pipeline = f"builtin.module(sparse-compiler{{{options} reassociate-fp-reductions=1 enable-index-optimizations=1}})"
        self.pipeline = pipeline
        self.opt_level = opt_level
        self.shared_libs = shared_libs

    def __call__(self, module: ir.Module):
        """Convenience application method."""
        self.compile(module)

    def compile(self, module: ir.Module):
        """Compiles the module by invoking the sparse copmiler pipeline."""
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
