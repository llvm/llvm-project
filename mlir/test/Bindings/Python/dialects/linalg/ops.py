# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import builtin
from mlir.dialects import linalg
from mlir.dialects import std


def run(f):
  print("\nTEST:", f.__name__)
  f()


# CHECK-LABEL: TEST: testStructuredOpOnTensors
def testStructuredOpOnTensors():
  with Context() as ctx, Location.unknown():
    module = Module.create()
    f32 = F32Type.get()
    tensor_type = RankedTensorType.get((2, 3, 4), f32)
    with InsertionPoint(module.body):
      func = builtin.FuncOp(name="matmul_test",
                            type=FunctionType.get(
                                inputs=[tensor_type, tensor_type],
                                results=[tensor_type]))
      with InsertionPoint(func.add_entry_block()):
        lhs, rhs = func.entry_block.arguments
        result = linalg.MatmulOp([lhs, rhs], results=[tensor_type]).result
        std.ReturnOp([result])

  # CHECK: %[[R:.*]] = linalg.matmul ins(%arg0, %arg1 : tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  print(module)


run(testStructuredOpOnTensors)


# CHECK-LABEL: TEST: testStructuredOpOnBuffers
def testStructuredOpOnBuffers():
  with Context() as ctx, Location.unknown():
    module = Module.create()
    f32 = F32Type.get()
    memref_type = MemRefType.get((2, 3, 4), f32)
    with InsertionPoint(module.body):
      func = builtin.FuncOp(name="matmul_test",
                            type=FunctionType.get(
                                inputs=[memref_type, memref_type, memref_type],
                                results=[]))
      with InsertionPoint(func.add_entry_block()):
        lhs, rhs, result = func.entry_block.arguments
        linalg.MatmulOp([lhs, rhs], outputs=[result])
        std.ReturnOp([])

  # CHECK: linalg.matmul ins(%arg0, %arg1 : memref<2x3x4xf32>, memref<2x3x4xf32>) outs(%arg2 : memref<2x3x4xf32>)
  print(module)


run(testStructuredOpOnBuffers)
