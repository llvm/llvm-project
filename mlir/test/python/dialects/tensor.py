# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
import mlir.dialects.arith as arith
import mlir.dialects.func as func
import mlir.dialects.tensor as tensor


def run(f):
  print("\nTEST:", f.__name__)
  f()
  return f


# CHECK-LABEL: TEST: testDimOp
@run
def testDimOp():
  with Context() as ctx, Location.unknown():
    module = Module.create()
    f32Type = F32Type.get()
    indexType = IndexType.get()
    with InsertionPoint(module.body):

      @func.FuncOp.from_py_func(RankedTensorType.get((-1, -1), f32Type))
      #      CHECK: func @tensor_static_dim
      # CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?xf32>
      #  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
      #  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
      #      CHECK:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
      #      CHECK:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
      #      CHECK:   return %[[D0]], %[[D1]]
      def tensor_static_dim(t):
        c0 = arith.ConstantOp(indexType, 0)
        c1 = arith.ConstantOp(indexType, 1)
        d0 = tensor.DimOp(t, c0)
        d1 = tensor.DimOp(t, c1)
        return [d0.result, d1.result]

    print(module)
