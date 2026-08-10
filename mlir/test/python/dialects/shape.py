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
                RankedTensorType.get((12, ShapedType.get_dynamic_size()), f32)
            )
            def const_shape_tensor(arg):
                shape.ConstWitnessOp(False)
                shape.ConstSizeOp(30)
                shape.ConstSizeOp(IntegerAttr.get(IndexType.get(), 40))
                x = shape.ConstShapeOp([1, 2])
                shape.MeetOp(x, x, error="impossible")
                return shape.ConstShapeOp(
                    DenseElementsAttr.get(
                        np.array([3, 4], dtype=np.int64), type=IndexType.get()
                    )
                )

        # CHECK-LABEL: func @const_shape_tensor(%arg0: tensor<12x?xf32>)
        # CHECK-DAG: shape.const_witness false
        # CHECK-DAG: shape.const_size 30
        # CHECK-DAG: shape.const_size 40
        # CHECK-DAG: shape.const_shape [1, 2] : tensor<2xindex>
        # CHECK-DAG: shape.const_shape [3, 4] : tensor<2xindex>
        print(module)
