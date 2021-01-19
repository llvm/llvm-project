#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This is a work in progress example to do end2end build and code generation
# of a small linalg program with configuration options. It is currently non
# functional and is being used to elaborate the APIs.

from typing import Tuple

from mlir.ir import *
from mlir.dialects import linalg
from mlir.dialects import std


# TODO: This should be in the core API.
def FuncOp(name: str, func_type: Type) -> Tuple[Operation, Block]:
  """Creates a |func| op.
    TODO: This should really be in the MLIR API.
    Returns:
      (operation, entry_block)
    """
  attrs = {
      "type": TypeAttr.get(func_type),
      "sym_name": StringAttr.get(name),
  }
  op = Operation.create("func", regions=1, attributes=attrs)
  body_region = op.regions[0]
  entry_block = body_region.blocks.append(*func_type.inputs)
  return op, entry_block


def build_matmul_buffers_func(func_name, m, k, n, dtype):
  lhs_type = MemRefType.get(dtype, [m, k])
  rhs_type = MemRefType.get(dtype, [k, n])
  result_type = MemRefType.get(dtype, [m, n])
  # TODO: There should be a one-liner for this.
  func_type = FunctionType.get([lhs_type, rhs_type, result_type], [])
  _, entry = FuncOp(func_name, func_type)
  lhs, rhs, result = entry.arguments
  with InsertionPoint(entry):
    op = linalg.MatmulOp([lhs, rhs], [result])
    # TODO: Implement support for SingleBlockImplicitTerminator
    block = op.regions[0].blocks.append()
    with InsertionPoint(block):
        linalg.YieldOp(values=[])

    std.ReturnOp([])


def build_matmul_tensors_func(func_name, m, k, n, dtype):
  # TODO: MemRefType and TensorTypes should not have inverted dtype/shapes
  # from each other.
  lhs_type = RankedTensorType.get([m, k], dtype)
  rhs_type = RankedTensorType.get([k, n], dtype)
  result_type = RankedTensorType.get([m, n], dtype)
  # TODO: There should be a one-liner for this.
  func_type = FunctionType.get([lhs_type, rhs_type], [result_type])
  _, entry = FuncOp(func_name, func_type)
  lhs, rhs = entry.arguments
  with InsertionPoint(entry):
    op = linalg.MatmulOp([lhs, rhs], results=[result_type])
    # TODO: Implement support for SingleBlockImplicitTerminator
    block = op.regions[0].blocks.append()
    with InsertionPoint(block):
        linalg.YieldOp(values=[])
    std.ReturnOp([op.result])


def run():
  with Context() as c, Location.unknown():
    module = Module.create()
    # TODO: This at_block_terminator vs default construct distinction feels
    # wrong and is error-prone.
    with InsertionPoint.at_block_terminator(module.body):
      build_matmul_buffers_func('main_buffers', 18, 32, 96, F32Type.get())
      build_matmul_tensors_func('main_tensors', 18, 32, 96, F32Type.get())

    print(module)


if __name__ == '__main__':
  run()
