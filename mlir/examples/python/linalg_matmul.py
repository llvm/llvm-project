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


# TODO: Generate customs builder vs patching one in.
def PatchMatmulOpInit(self, lhs, rhs, result, loc=None, ip=None):
    super(linalg.MatmulOp, self).__init__(
        self._ods_build_default(operands=[[lhs, rhs], [result]],
                                results=[],
                                loc=loc,
                                ip=ip))
    # TODO: Implement support for SingleBlockImplicitTerminator
    block = self.regions[0].blocks.append()
    with InsertionPoint(block):
        linalg.YieldOp(values=[])

linalg.MatmulOp.__init__ = PatchMatmulOpInit


def build_matmul_func(func_name, m, k, n, dtype):
    lhs_type = MemRefType.get(dtype, [m, k])
    rhs_type = MemRefType.get(dtype, [k, n])
    result_type = MemRefType.get(dtype, [m, n])
    # TODO: There should be a one-liner for this.
    func_type = FunctionType.get([lhs_type, rhs_type, result_type], [])
    _, entry = FuncOp(func_name, func_type)
    lhs, rhs, result = entry.arguments
    with InsertionPoint(entry):
        linalg.MatmulOp(lhs, rhs, result)
        std.ReturnOp([])


def run():
    with Context() as c, Location.unknown():
        module = Module.create()
        # TODO: This at_block_terminator vs default construct distinction feels
        # wrong and is error-prone.
        with InsertionPoint.at_block_terminator(module.body):
            build_matmul_func('main', 18, 32, 96, F32Type.get())

        print(module)
        print(module.operation.get_asm(print_generic_op_form=True))


if __name__ == '__main__': run()
