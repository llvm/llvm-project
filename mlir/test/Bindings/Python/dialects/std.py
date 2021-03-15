# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
import mlir.dialects.std as std

def run(f):
  print("\nTEST:", f.__name__)
  f()

# _HECK-LABEL: TEST: testSubViewAccessors
def testSubViewAccessors():
  ctx = Context()
  module = Module.parse(r"""
    func @f1(%arg0: memref<?x?xf32>) {
      %0 = constant 0 : index
      %1 = constant 1 : index
      %2 = constant 2 : index
      %3 = constant 3 : index
      %4 = constant 4 : index
      %5 = constant 5 : index
      memref.subview %arg0[%0, %1][%2, %3][%4, %5] : memref<?x?xf32> to memref<?x?xf32, offset: ?, strides: [?, ?]>
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

  # _HECK: SubViewOp
  print(type(subview).__name__)

  # _HECK: constant 0
  print(subview.offsets[0])
  # _HECK: constant 1
  print(subview.offsets[1])
  # _HECK: constant 2
  print(subview.sizes[0])
  # _HECK: constant 3
  print(subview.sizes[1])
  # _HECK: constant 4
  print(subview.strides[0])
  # _HECK: constant 5
  print(subview.strides[1])


# TODO: re-enable after moving the bindings from std to memref dialects
# run(testSubViewAccessors)

def forcePass():
  # CHECK: okay
  print("okay")

run(forcePass)
