# RUN: %PYTHON %s | FileCheck %s

from mlir.dialects import arith, builtin, func, linalg, tensor
from mlir.dialects.linalg.opdsl.lang import *
from mlir.ir import *


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# CHECK-LABEL: TEST: testFill
@run
def testFill():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        f32 = F32Type.get()
        with InsertionPoint(module.body):
            # CHECK-LABEL: func @fill_tensor
            #  CHECK-SAME:   %[[OUT:[0-9a-z]+]]: tensor<12x?xf32>
            #  CHECK-NEXT: %[[CST:.*]] = arith.constant 0.0{{.*}} : f32
            #  CHECK-NEXT: %[[RES:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[OUT]] : tensor<12x?xf32>) -> tensor<12x?xf32>
            #  CHECK-NEXT: return %[[RES]] : tensor<12x?xf32>
            @func.FuncOp.from_py_func(
                RankedTensorType.get((12, ShapedType.get_dynamic_size()), f32)
            )
            def fill_tensor(out):
                zero = arith.ConstantOp(
                    value=FloatAttr.get(f32, 0.0), result=f32
                ).result
                return linalg.fill(zero, outs=[out])

            # CHECK-LABEL: func @fill_buffer
            #  CHECK-SAME:   %[[OUT:[0-9a-z]+]]: memref<12x?xf32>
            #  CHECK-NEXT: %[[CST:.*]] = arith.constant 0.0{{.*}} : f32
            #  CHECK-NEXT: linalg.fill ins(%[[CST]] : f32) outs(%[[OUT]] : memref<12x?xf32>)
            #  CHECK-NEXT: return
            @func.FuncOp.from_py_func(
                MemRefType.get((12, ShapedType.get_dynamic_size()), f32)
            )
            def fill_buffer(out):
                zero = arith.ConstantOp(
                    value=FloatAttr.get(f32, 0.0), result=f32
                ).result
                linalg.fill(zero, outs=[out])

    print(module)


# CHECK-LABEL: TEST: testNamedStructuredOpCustomForm
@run
def testNamedStructuredOpCustomForm():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        f32 = F32Type.get()
        with InsertionPoint(module.body):

            @func.FuncOp.from_py_func(
                RankedTensorType.get((4, 8), f32), RankedTensorType.get((4, 8), f32)
            )
            def named_form(lhs, rhs):
                init_result = tensor.EmptyOp([4, 8], f32)
                # Check for the named form with custom format
                #      CHECK: linalg.elemwise_unary
                # CHECK-SAME:    cast = #linalg.type_fn<cast_signed>
                # CHECK-SAME:    fun = #linalg.unary_fn<exp>
                # CHECK-SAME:    ins(%{{.*}} : tensor<4x8xf32>) outs(%{{.*}} : tensor<4x8xf32>)
                unary_result = linalg.elemwise_unary(lhs, outs=[init_result.result])
                #      CHECK: linalg.elemwise_binary
                # CHECK-SAME:    cast = #linalg.type_fn<cast_unsigned>
                # CHECK-SAME:    fun = #linalg.binary_fn<mul>
                # CHECK-SAME:    ins(%{{.*}}, %{{.*}} : tensor<4x8xf32>, tensor<4x8xf32>) outs(%{{.*}} : tensor<4x8xf32>)
                #      CHECK: return
                binary_result = linalg.elemwise_binary(
                    lhs,
                    rhs,
                    outs=[init_result.result],
                    fun=BinaryFn.mul,
                    cast=TypeFn.cast_unsigned,
                )
                return unary_result, binary_result

    print(module)

# CHECK-LABEL: TEST: testIdentityRegionOps
@run
def testIdentityRegionOps():
    with Context(), Location.unknown():
        module = Module.create()
        f32 = F32Type.get()
        with InsertionPoint(module.body):
            # CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<1x13xf32>
            # CHECK: %[[VAL_1:.*]] = tensor.empty() : tensor<13x1xf32>
            op1 = tensor.EmptyOp([1, 13], f32)
            op2 = tensor.EmptyOp([13, 1], f32)
            # CHECK: %[[VAL_2:.*]] = linalg.transpose ins(%[[VAL_0]] : tensor<1x13xf32>) outs(%[[VAL_1]] : tensor<13x1xf32>) permutation = [1, 0]
            op3 = linalg.TransposeOp(
                result=[RankedTensorType.get((13, 1), f32)],
                input=op1,
                init=op2,
                permutation=[1, 0],
            )
            linalg.fill_builtin_region(op3.operation)

            # CHECK: %[[VAL_3:.*]] = linalg.transpose ins(%[[VAL_1]] : tensor<13x1xf32>) outs(%[[VAL_0]] : tensor<1x13xf32>) permutation = [1, 0]
            op4 = linalg.transpose(op2, outs=[op1], permutation=[1, 0])

            # CHECK:         func.func @transpose_op(%[[VAL_4:.*]]: memref<1x13xf32>, %[[VAL_5:.*]]: memref<13x1xf32>)
            @func.FuncOp.from_py_func(
                MemRefType.get((1, 13), f32),
                MemRefType.get((13, 1), f32),
            )
            def transpose_op(op1, op2):
                # CHECK: linalg.transpose ins(%[[VAL_4]] : memref<1x13xf32>) outs(%[[VAL_5]] : memref<13x1xf32>) permutation = [1, 0]
                op3 = linalg.TransposeOp(
                    result=[],
                    input=op1,
                    init=op2,
                    permutation=[1, 0],
                )
                linalg.fill_builtin_region(op3.operation)
                # CHECK: linalg.transpose ins(%[[VAL_5]] : memref<13x1xf32>) outs(%[[VAL_4]] : memref<1x13xf32>) permutation = [1, 0]
                op4 = linalg.transpose(op2, outs=[op1], permutation=[1, 0])

            # CHECK: %[[VAL_6:.*]] = tensor.empty() : tensor<16xf32>
            # CHECK: %[[VAL_7:.*]] = tensor.empty() : tensor<16x64xf32>
            op1 = tensor.EmptyOp([16], f32)
            op2 = tensor.EmptyOp([16, 64], f32)
            # CHECK: %[[VAL_8:.*]] = linalg.broadcast ins(%[[VAL_6]] : tensor<16xf32>) outs(%[[VAL_7]] : tensor<16x64xf32>) dimensions = [1]
            op3 = linalg.BroadcastOp(
                result=[RankedTensorType.get((16, 64), f32)],
                input=op1,
                init=op2,
                dimensions=[1],
            )
            linalg.fill_builtin_region(op3.operation)

            # CHECK: %[[VAL_9:.*]] = tensor.empty() : tensor<64xf32>
            op4 = tensor.EmptyOp([64], f32)
            # CHECK: %[[VAL_10:.*]] = linalg.broadcast ins(%[[VAL_9]] : tensor<64xf32>) outs(%[[VAL_7]] : tensor<16x64xf32>) dimensions = [0]
            op5 = linalg.broadcast(op4, outs=[op2], dimensions=[0])

            # CHECK: func.func @broadcast_op(%[[VAL_11:.*]]: memref<16xf32>, %[[VAL_12:.*]]: memref<16x64xf32>, %[[VAL_13:.*]]: memref<64xf32>)
            @func.FuncOp.from_py_func(
                MemRefType.get((16,), f32),
                MemRefType.get((16, 64), f32),
                MemRefType.get((64,), f32),
            )
            def broadcast_op(op1, op2, op3):
                # CHECK: linalg.broadcast ins(%[[VAL_11]] : memref<16xf32>) outs(%[[VAL_12]] : memref<16x64xf32>) dimensions = [1]
                op4 = linalg.BroadcastOp(
                    result=[],
                    input=op1,
                    init=op2,
                    dimensions=[1],
                )
                linalg.fill_builtin_region(op4.operation)
                # CHECK: linalg.broadcast ins(%[[VAL_13]] : memref<64xf32>) outs(%[[VAL_12]] : memref<16x64xf32>) dimensions = [0]
                op5 = linalg.broadcast(op3, outs=[op2], dimensions=[0])

    print(module)
