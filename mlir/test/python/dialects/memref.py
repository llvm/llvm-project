# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
import mlir.dialects.func as func
import mlir.dialects.memref as memref


def run(f):
  print("\nTEST:", f.__name__)
  f()
  return f


# CHECK-LABEL: TEST: testSubViewAccessors
@run
def testSubViewAccessors():
  ctx = Context()
  module = Module.parse(
      r"""
    func.func @f1(%arg0: memref<?x?xf32>) {
      %0 = arith.constant 0 : index
      %1 = arith.constant 1 : index
      %2 = arith.constant 2 : index
      %3 = arith.constant 3 : index
      %4 = arith.constant 4 : index
      %5 = arith.constant 5 : index
      memref.subview %arg0[%0, %1][%2, %3][%4, %5] : memref<?x?xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      return
    }
  """, ctx)
  func_body = module.body.operations[0].regions[0].blocks[0]
  subview = func_body.operations[6]

  assert subview.source == subview.operands[0]
  assert len(subview.offsets) == 2
  assert len(subview.sizes) == 2
  assert len(subview.strides) == 2
  assert subview.result == subview.results[0]

  # CHECK: SubViewOp
  print(type(subview).__name__)

  # CHECK: constant 0
  print(subview.offsets[0])
  # CHECK: constant 1
  print(subview.offsets[1])
  # CHECK: constant 2
  print(subview.sizes[0])
  # CHECK: constant 3
  print(subview.sizes[1])
  # CHECK: constant 4
  print(subview.strides[0])
  # CHECK: constant 5
  print(subview.strides[1])


# CHECK-LABEL: TEST: testCustomBuidlers
@run
def testCustomBuidlers():
  with Context() as ctx, Location.unknown(ctx):
    module = Module.parse(r"""
      func.func @f1(%arg0: memref<?x?xf32>, %arg1: index, %arg2: index) {
        return
      }
    """)
    f = module.body.operations[0]
    func_body = f.regions[0].blocks[0]
    with InsertionPoint.at_block_terminator(func_body):
      memref.LoadOp(f.arguments[0], f.arguments[1:])

    # CHECK: func @f1(%[[ARG0:.*]]: memref<?x?xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
    # CHECK: memref.load %[[ARG0]][%[[ARG1]], %[[ARG2]]]
    print(module)
    assert module.operation.verify()
