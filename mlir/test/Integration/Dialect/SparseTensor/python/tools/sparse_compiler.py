#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#  This file contains the sparse compiler class.

from mlir import all_passes_registration
from mlir import ir
from mlir import passmanager

class SparseCompiler:
  """Sparse compiler definition."""

  def __init__(self, options: str):
    pipeline = f'sparse-compiler{{{options} reassociate-fp-reductions=1 enable-index-optimizations=1}}'
    self.pipeline = pipeline

  def __call__(self, module: ir.Module):
    passmanager.PassManager.parse(self.pipeline).run(module)
