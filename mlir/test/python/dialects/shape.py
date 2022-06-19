# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
import numpy as np
import mlir.dialects.func as func
import mlir.dialects.shape as shape


def run(f):
  print("\nTEST:", f.__name__)
  f()
  return f


# CHECK-LABEL: TEST: testConstShape
@run
def testConstShape():
  with Context() as ctx, Location.unknown():
    module = Module.create()
    f32 = F32Type.get()
    with InsertionPoint(module.body):
      @func.FuncOp.from_py_func(
          RankedTensorType.get((12, -1), f32))
      def const_shape_tensor(arg):
        return shape.ConstShapeOp(
          DenseElementsAttr.get(np.array([10, 20], dtype=np.int64), type=IndexType.get()))

    # CHECK-LABEL: func @const_shape_tensor(%arg0: tensor<12x?xf32>)
    # CHECK: shape.const_shape [10, 20] : tensor<2xindex>
    print(module)
