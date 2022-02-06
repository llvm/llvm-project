# RUN: %PYTHON %s | FileCheck %s

import gc
import io
import itertools
from mlir.ir import *
from mlir.dialects import builtin
from mlir.dialects import cf
# Note: std dialect needed for terminators.
from mlir.dialects import std


def run(f):
  print("\nTEST:", f.__name__)
  f()
  gc.collect()
  assert Context._get_live_count() == 0
  return f


# CHECK-LABEL: TEST: testBlockCreation
# CHECK: func @test(%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i16)
# CHECK:   cf.br ^bb1(%[[ARG1]] : i16)
# CHECK: ^bb1(%[[PHI0:.*]]: i16):
# CHECK:   cf.br ^bb2(%[[ARG0]] : i32)
# CHECK: ^bb2(%[[PHI1:.*]]: i32):
# CHECK:   return
@run
def testBlockCreation():
  with Context() as ctx, Location.unknown():
    module = Module.create()
    with InsertionPoint(module.body):
      f_type = FunctionType.get(
          [IntegerType.get_signless(32),
           IntegerType.get_signless(16)], [])
      f_op = builtin.FuncOp("test", f_type)
      entry_block = f_op.add_entry_block()
      i32_arg, i16_arg = entry_block.arguments
      successor_block = entry_block.create_after(i32_arg.type)
      with InsertionPoint(successor_block) as successor_ip:
        assert successor_ip.block == successor_block
        std.ReturnOp([])
      middle_block = successor_block.create_before(i16_arg.type)

      with InsertionPoint(entry_block) as entry_ip:
        assert entry_ip.block == entry_block
        cf.BranchOp([i16_arg], dest=middle_block)

      with InsertionPoint(middle_block) as middle_ip:
        assert middle_ip.block == middle_block
        cf.BranchOp([i32_arg], dest=successor_block)
    print(module.operation)
    # Ensure region back references are coherent.
    assert entry_block.region == middle_block.region == successor_block.region


# CHECK-LABEL: TEST: testFirstBlockCreation
# CHECK: func @test(%{{.*}}: f32)
# CHECK:   return
@run
def testFirstBlockCreation():
  with Context() as ctx, Location.unknown():
    module = Module.create()
    f32 = F32Type.get()
    with InsertionPoint(module.body):
      func = builtin.FuncOp("test", ([f32], []))
      entry_block = Block.create_at_start(func.operation.regions[0], [f32])
      with InsertionPoint(entry_block):
        std.ReturnOp([])

    print(module)
    assert module.operation.verify()
    assert func.body.blocks[0] == entry_block
