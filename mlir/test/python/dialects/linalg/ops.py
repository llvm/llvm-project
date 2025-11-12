# RUN: %PYTHON %s | FileCheck %s

from mlir.dialects import arith, func, linalg, tensor, memref, builtin
from mlir.dialects.linalg.opdsl.lang import *
from mlir.extras import types as T
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


# CHECK-LABEL: TEST: testGenericOp
@run
def testGenericOp():
    with Context(), Location.unknown():
        module = Module.create()
        f32 = F32Type.get()
        memref_t = MemRefType.get([10, 10], f32)
        with InsertionPoint(module.body):
            id_map_1 = AffineMap.get_identity(2)
            # CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<16x16xf32>
            # CHECK: %[[VAL_1:.*]] = tensor.empty() : tensor<16x16xf32>
            x = tensor.empty((16, 16), f32)
            y = tensor.empty((16, 16), f32)

            # CHECK: %[[VAL_2:.*]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_0]] : tensor<16x16xf32>) outs(%[[VAL_1]] : tensor<16x16xf32>) {
            # CHECK: ^bb0(%in: f32, %out: f32):
            # CHECK:   linalg.yield %in : f32
            # CHECK: } -> tensor<16x16xf32>
            @linalg.generic(
                [x],
                [y],
                [id_map_1, id_map_1],
                [linalg.IteratorType.parallel, linalg.IteratorType.parallel],
            )
            def f(a, b):
                assert isinstance(a, Value)
                assert isinstance(a.type, F32Type)
                assert isinstance(b, Value)
                assert isinstance(b.type, F32Type)
                return a

            assert isinstance(f, Value)
            assert isinstance(f.type, RankedTensorType)

            # CHECK: %[[VAL_3:.*]] = tensor.empty() : tensor<16x16x16xf32>
            z = tensor.empty((16, 16, 16), f32)

            minor_id = AffineMap.get_minor_identity(3, 2)
            id_map_2 = AffineMap.get_identity(3)

            # CHECK: %[[VAL_4:.+]]:2 = linalg.generic {indexing_maps = [#map1, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[VAL_0]] : tensor<16x16xf32>) outs(%[[VAL_3]], %[[VAL_3]] : tensor<16x16x16xf32>, tensor<16x16x16xf32>) {
            # CHECK: ^bb0(%in: f32, %out: f32, %out_1: f32):
            # CHECK:   linalg.yield %in, %out : f32, f32
            # CHECK: } -> (tensor<16x16x16xf32>, tensor<16x16x16xf32>)
            @linalg.generic(
                [x],
                [z, z],
                [minor_id, id_map_2, id_map_2],
                [
                    linalg.IteratorType.parallel,
                    linalg.IteratorType.parallel,
                    linalg.IteratorType.parallel,
                ],
            )
            def g(a, b, c):
                assert isinstance(a, Value)
                assert isinstance(a.type, F32Type)
                assert isinstance(b, Value)
                assert isinstance(b.type, F32Type)
                assert isinstance(c, Value)
                assert isinstance(c.type, F32Type)
                return a, b

            assert isinstance(g, OpResultList)
            assert len(g) == 2
            assert isinstance(g[0].type, RankedTensorType)
            assert isinstance(g[1].type, RankedTensorType)

            # CHECK: %[[VAL_5:.*]] = memref.alloc() : memref<10x10xf32>
            # CHECK: %[[VAL_6:.*]] = memref.alloc() : memref<10x10xf32>
            xx = memref.alloc(memref_t, [], [])
            yy = memref.alloc(memref_t, [], [])

            # CHECK: linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_5]] : memref<10x10xf32>) outs(%[[VAL_6]] : memref<10x10xf32>) {
            # CHECK: ^bb0(%in: f32, %out: f32):
            # CHECK:   linalg.yield %in : f32
            # CHECK: }
            @linalg.generic(
                [xx],
                [yy],
                [id_map_1, id_map_1],
                [linalg.IteratorType.parallel, linalg.IteratorType.parallel],
            )
            def f(a, b):
                assert isinstance(a, Value)
                assert isinstance(a.type, F32Type)
                assert isinstance(b, Value)
                assert isinstance(b.type, F32Type)
                return a

    module.operation.verify()
    print(module)


# CHECK-LABEL: TEST: testMatmulOp
@run
def testMatmulOp():
    with Context(), Location.unknown():
        module = Module.create()
        f32 = F32Type.get()
        with InsertionPoint(module.body):
            a_shape = (4, 8)
            b_shape = (8, 12)
            b_transposed_shape = (12, 8)
            c_shape = (4, 12)

            dimM = ir.AffineDimExpr.get(0)
            dimN = ir.AffineDimExpr.get(1)
            dimK = ir.AffineDimExpr.get(2)

            # CHECK: #[[$A_MAP:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
            # CHECK: #[[$BTrans_MAP:.*]] = affine_map<(d0, d1, d2) -> (d1, d2)>
            # CHECK: #[[$C_MAP:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
            a_map = ir.AffineMap.get(3, 0, [dimM, dimK])
            b_map = ir.AffineMap.get(3, 0, [dimK, dimN])
            c_map = ir.AffineMap.get(3, 0, [dimM, dimN])
            b_transposed_map = ir.AffineMap.get(3, 0, [dimN, dimK])

            # CHECK: func.func @matmul_op(
            @func.FuncOp.from_py_func(
                # CHECK-SAME:                         %[[A:.*]]: tensor<4x8xf32>,
                RankedTensorType.get(a_shape, f32),
                # CHECK-SAME:                         %[[Amem:.*]]: memref<4x8xf32>,
                MemRefType.get(a_shape, f32),
                # CHECK-SAME:                         %[[B:.*]]: tensor<8x12xf32>,
                RankedTensorType.get(b_shape, f32),
                # CHECK-SAME:                         %[[Bmem:.*]]: memref<8x12xf32>,
                MemRefType.get(b_shape, f32),
                # CHECK-SAME:                         %[[BTrans:.*]]: tensor<12x8xf32>,
                RankedTensorType.get(b_transposed_shape, f32),
                # CHECK-SAME:                         %[[BTransmem:.*]]: memref<12x8xf32>,
                MemRefType.get(b_transposed_shape, f32),
                # CHECK-SAME:                         %[[C:.*]]: tensor<4x12xf32>,
                RankedTensorType.get(c_shape, f32),
                # CHECK-SAME:                         %[[Cmem:.*]]: memref<4x12xf32>)
                MemRefType.get(c_shape, f32),
            )
            def matmul_op(A, Amem, B, Bmem, Btransposed, Btransposedmem, C, Cmem):
                # CHECK: linalg.matmul ins(%[[A]], %[[B]] : tensor<4x8xf32>, tensor<8x12xf32>) outs(%[[C]] : tensor<4x12xf32>)
                res = linalg.MatmulOp(
                    result_tensors=(C.type,),
                    inputs=(A, B),
                    outputs=(C,),
                )
                linalg.fill_builtin_region(res.operation)
                # CHECK: linalg.matmul ins(%[[A]], %[[B]] : tensor<4x8xf32>, tensor<8x12xf32>) outs(%[[C]] : tensor<4x12xf32>)
                res = linalg.matmul(A, B, outs=(C,))

                # CHECK: linalg.matmul indexing_maps = [#[[$A_MAP]], #[[$BTrans_MAP]], #[[$C_MAP]]] ins(%[[A]], %[[BTrans]] : tensor<4x8xf32>, tensor<12x8xf32>) outs(%[[C]] : tensor<4x12xf32>)
                res = linalg.MatmulOp(
                    result_tensors=(C.type,),
                    inputs=(A, Btransposed),
                    outputs=(C,),
                    indexing_maps=[a_map, b_transposed_map, c_map],
                )
                linalg.fill_builtin_region(res.operation)
                # CHECK: linalg.matmul indexing_maps = [#[[$A_MAP]], #[[$BTrans_MAP]], #[[$C_MAP]]] ins(%[[A]], %[[BTrans]] : tensor<4x8xf32>, tensor<12x8xf32>) outs(%[[C]] : tensor<4x12xf32>)
                res = linalg.matmul(
                    A,
                    Btransposed,
                    outs=(C,),
                    indexing_maps=[a_map, b_transposed_map, c_map],
                )

                # And now with memrefs...

                # CHECK: linalg.matmul ins(%[[Amem]], %[[Bmem]] : memref<4x8xf32>, memref<8x12xf32>) outs(%[[Cmem]] : memref<4x12xf32>)
                res = linalg.MatmulOp(
                    result_tensors=[],
                    inputs=(Amem, Bmem),
                    outputs=(Cmem,),
                )
                linalg.fill_builtin_region(res.operation)
                # CHECK: linalg.matmul ins(%[[Amem]], %[[Bmem]] : memref<4x8xf32>, memref<8x12xf32>) outs(%[[Cmem]] : memref<4x12xf32>)
                linalg.matmul(Amem, Bmem, outs=(Cmem,))

                # CHECK: linalg.matmul indexing_maps = [#[[$A_MAP]], #[[$BTrans_MAP]], #[[$C_MAP]]] ins(%[[Amem]], %[[BTransmem]] : memref<4x8xf32>, memref<12x8xf32>) outs(%[[Cmem]] : memref<4x12xf32>)
                res = linalg.MatmulOp(
                    result_tensors=[],
                    inputs=(Amem, Btransposedmem),
                    outputs=(Cmem,),
                    indexing_maps=[a_map, b_transposed_map, c_map],
                )
                linalg.fill_builtin_region(res.operation)
                # CHECK: linalg.matmul indexing_maps = [#[[$A_MAP]], #[[$BTrans_MAP]], #[[$C_MAP]]] ins(%[[Amem]], %[[BTransmem]] : memref<4x8xf32>, memref<12x8xf32>) outs(%[[Cmem]] : memref<4x12xf32>)
                linalg.matmul(
                    Amem,
                    Btransposedmem,
                    outs=(Cmem,),
                    indexing_maps=[a_map, b_transposed_map, c_map],
                )

        print(module)


# CHECK-LABEL: TEST: testContractOp
@run
def testContractOp():
    with Context(), Location.unknown():
        module = Module.create()
        f32 = F32Type.get()
        with InsertionPoint(module.body):
            a_shape = (4, 8)
            b_shape = (8, 12)
            b_transposed_shape = (12, 8)
            c_shape = (4, 12)

            dimM = ir.AffineDimExpr.get(0)
            dimN = ir.AffineDimExpr.get(1)
            dimK = ir.AffineDimExpr.get(2)

            # CHECK: #[[$A_MAP:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
            # CHECK: #[[$B_MAP:.*]] = affine_map<(d0, d1, d2) -> (d2, d1)>
            # CHECK: #[[$C_MAP:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
            # CHECK: #[[$BTrans_MAP:.*]] = affine_map<(d0, d1, d2) -> (d1, d2)>
            a_map = ir.AffineMap.get(3, 0, [dimM, dimK])
            b_map = ir.AffineMap.get(3, 0, [dimK, dimN])
            c_map = ir.AffineMap.get(3, 0, [dimM, dimN])
            b_transposed_map = ir.AffineMap.get(3, 0, [dimN, dimK])

            # CHECK: func.func @matmul_as_contract_op(
            @func.FuncOp.from_py_func(
                # CHECK-SAME:                         %[[A:.*]]: tensor<4x8xf32>,
                RankedTensorType.get(a_shape, f32),
                # CHECK-SAME:                         %[[Amem:.*]]: memref<4x8xf32>,
                MemRefType.get(a_shape, f32),
                # CHECK-SAME:                         %[[B:.*]]: tensor<8x12xf32>,
                RankedTensorType.get(b_shape, f32),
                # CHECK-SAME:                         %[[Bmem:.*]]: memref<8x12xf32>,
                MemRefType.get(b_shape, f32),
                # CHECK-SAME:                         %[[BTrans:.*]]: tensor<12x8xf32>,
                RankedTensorType.get(b_transposed_shape, f32),
                # CHECK-SAME:                         %[[BTransmem:.*]]: memref<12x8xf32>,
                MemRefType.get(b_transposed_shape, f32),
                # CHECK-SAME:                         %[[C:.*]]: tensor<4x12xf32>,
                RankedTensorType.get(c_shape, f32),
                # CHECK-SAME:                         %[[Cmem:.*]]: memref<4x12xf32>)
                MemRefType.get(c_shape, f32),
            )
            def matmul_as_contract_op(
                A, Amem, B, Bmem, Btransposed, Btransposedmem, C, Cmem
            ):
                # CHECK: linalg.contract indexing_maps = [#[[$A_MAP]], #[[$B_MAP]], #[[$C_MAP]]] ins(%[[A]], %[[B]] : tensor<4x8xf32>, tensor<8x12xf32>) outs(%[[C]] : tensor<4x12xf32>)
                op4 = linalg.ContractOp(
                    result_tensors=(C.type,),
                    inputs=(A, B),
                    outputs=(C,),
                    indexing_maps=[a_map, b_map, c_map],
                )
                linalg.fill_builtin_region(op4.operation)
                # CHECK: linalg.contract indexing_maps = [#[[$A_MAP]], #[[$B_MAP]], #[[$C_MAP]]] ins(%[[A]], %[[B]] : tensor<4x8xf32>, tensor<8x12xf32>) outs(%[[C]] : tensor<4x12xf32>)
                op5 = linalg.contract(
                    A, B, outs=(C,), indexing_maps=[a_map, b_map, c_map]
                )

                # CHECK: linalg.contract indexing_maps = [#[[$A_MAP]], #[[$BTrans_MAP]], #[[$C_MAP]]] ins(%[[A]], %[[BTrans]] : tensor<4x8xf32>, tensor<12x8xf32>) outs(%[[C]] : tensor<4x12xf32>)
                op4 = linalg.ContractOp(
                    result_tensors=(C.type,),
                    inputs=(A, Btransposed),
                    outputs=(C,),
                    indexing_maps=[a_map, b_transposed_map, c_map],
                )
                linalg.fill_builtin_region(op4.operation)
                # CHECK: linalg.contract indexing_maps = [#[[$A_MAP]], #[[$BTrans_MAP]], #[[$C_MAP]]] ins(%[[A]], %[[BTrans]] : tensor<4x8xf32>, tensor<12x8xf32>) outs(%[[C]] : tensor<4x12xf32>)
                op5 = linalg.contract(
                    A,
                    Btransposed,
                    outs=(C,),
                    indexing_maps=[a_map, b_transposed_map, c_map],
                )
                # And now with memrefs...

                # CHECK: linalg.contract indexing_maps = [#[[$A_MAP]], #[[$B_MAP]], #[[$C_MAP]]] ins(%[[Amem]], %[[Bmem]] : memref<4x8xf32>, memref<8x12xf32>) outs(%[[Cmem]] : memref<4x12xf32>)
                op4 = linalg.ContractOp(
                    result_tensors=[],
                    inputs=(Amem, Bmem),
                    outputs=(Cmem,),
                    indexing_maps=[a_map, b_map, c_map],
                )
                linalg.fill_builtin_region(op4.operation)
                # CHECK: linalg.contract indexing_maps = [#[[$A_MAP]], #[[$B_MAP]], #[[$C_MAP]]] ins(%[[Amem]], %[[Bmem]] : memref<4x8xf32>, memref<8x12xf32>) outs(%[[Cmem]] : memref<4x12xf32>)
                linalg.contract(
                    Amem, Bmem, outs=(Cmem,), indexing_maps=[a_map, b_map, c_map]
                )

                # CHECK: linalg.contract indexing_maps = [#[[$A_MAP]], #[[$BTrans_MAP]], #[[$C_MAP]]] ins(%[[Amem]], %[[BTransmem]] : memref<4x8xf32>, memref<12x8xf32>) outs(%[[Cmem]] : memref<4x12xf32>)
                op4 = linalg.ContractOp(
                    result_tensors=[],
                    inputs=(Amem, Btransposedmem),
                    outputs=(Cmem,),
                    indexing_maps=[a_map, b_transposed_map, c_map],
                )
                linalg.fill_builtin_region(op4.operation)
                # CHECK: linalg.contract indexing_maps = [#[[$A_MAP]], #[[$BTrans_MAP]], #[[$C_MAP]]] ins(%[[Amem]], %[[BTransmem]] : memref<4x8xf32>, memref<12x8xf32>) outs(%[[Cmem]] : memref<4x12xf32>)
                linalg.contract(
                    Amem,
                    Btransposedmem,
                    outs=(Cmem,),
                    indexing_maps=[a_map, b_transposed_map, c_map],
                )

        print(module)


# CHECK-LABEL: TEST: testBatchMatmulOp
@run
def testBatchMatmulOp():
    with Context(), Location.unknown():
        module = Module.create()
        f32 = F32Type.get()
        with InsertionPoint(module.body):
            a_shape = (2, 4, 8)
            b_shape = (2, 8, 12)
            b_transposed_shape = (2, 12, 8)
            c_shape = (2, 4, 12)

            dimBatch = ir.AffineDimExpr.get(0)
            dimM = ir.AffineDimExpr.get(1)
            dimN = ir.AffineDimExpr.get(2)
            dimK = ir.AffineDimExpr.get(3)

            # CHECK: #[[$A_MAP:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
            # CHECK: #[[$BTrans_MAP:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
            # CHECK: #[[$C_MAP:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

            a_map = ir.AffineMap.get(4, 0, [dimBatch, dimM, dimK])
            b_transposed_map = ir.AffineMap.get(4, 0, [dimBatch, dimN, dimK])
            c_map = ir.AffineMap.get(4, 0, [dimBatch, dimM, dimN])

            # CHECK: func.func @batch_matmul_op(
            @func.FuncOp.from_py_func(
                # CHECK-SAME:                         %[[A:.*]]: tensor<2x4x8xf32>,
                RankedTensorType.get(a_shape, f32),
                # CHECK-SAME:                         %[[Amem:.*]]: memref<2x4x8xf32>,
                MemRefType.get(a_shape, f32),
                # CHECK-SAME:                         %[[B:.*]]: tensor<2x8x12xf32>,
                RankedTensorType.get(b_shape, f32),
                # CHECK-SAME:                         %[[Bmem:.*]]: memref<2x8x12xf32>,
                MemRefType.get(b_shape, f32),
                # CHECK-SAME:                         %[[BTrans:.*]]: tensor<2x12x8xf32>,
                RankedTensorType.get(b_transposed_shape, f32),
                # CHECK-SAME:                         %[[BTransmem:.*]]: memref<2x12x8xf32>,
                MemRefType.get(b_transposed_shape, f32),
                # CHECK-SAME:                         %[[C:.*]]: tensor<2x4x12xf32>,
                RankedTensorType.get(c_shape, f32),
                # CHECK-SAME:                         %[[Cmem:.*]]: memref<2x4x12xf32>)
                MemRefType.get(c_shape, f32),
            )
            def batch_matmul_op(A, Amem, B, Bmem, Btransposed, Btransposedmem, C, Cmem):
                # CHECK: linalg.batch_matmul ins(%[[A]], %[[B]] : tensor<2x4x8xf32>, tensor<2x8x12xf32>) outs(%[[C]] : tensor<2x4x12xf32>)
                res = linalg.BatchMatmulOp(
                    result_tensors=(C.type,),
                    inputs=(A, B),
                    outputs=(C,),
                )
                linalg.fill_builtin_region(res.operation)
                # CHECK: linalg.batch_matmul ins(%[[A]], %[[B]] : tensor<2x4x8xf32>, tensor<2x8x12xf32>) outs(%[[C]] : tensor<2x4x12xf32>)
                res = linalg.batch_matmul(A, B, outs=(C,))

                # CHECK: linalg.batch_matmul indexing_maps = [#[[$A_MAP]], #[[$BTrans_MAP]], #[[$C_MAP]]] ins(%[[A]], %[[BTrans]] : tensor<2x4x8xf32>, tensor<2x12x8xf32>) outs(%[[C]] : tensor<2x4x12xf32>)
                res = linalg.BatchMatmulOp(
                    result_tensors=(C.type,),
                    inputs=(A, Btransposed),
                    outputs=(C,),
                    indexing_maps=[a_map, b_transposed_map, c_map],
                )
                linalg.fill_builtin_region(res.operation)
                # CHECK: linalg.batch_matmul indexing_maps = [#[[$A_MAP]], #[[$BTrans_MAP]], #[[$C_MAP]]] ins(%[[A]], %[[BTrans]] : tensor<2x4x8xf32>, tensor<2x12x8xf32>) outs(%[[C]] : tensor<2x4x12xf32>)
                res = linalg.batch_matmul(
                    A,
                    Btransposed,
                    outs=(C,),
                    indexing_maps=[a_map, b_transposed_map, c_map],
                )

                # CHECK: linalg.batch_matmul ins(%[[Amem]], %[[Bmem]] : memref<2x4x8xf32>, memref<2x8x12xf32>) outs(%[[Cmem]] : memref<2x4x12xf32>)
                res = linalg.BatchMatmulOp(
                    result_tensors=[],
                    inputs=(Amem, Bmem),
                    outputs=(Cmem,),
                )
                linalg.fill_builtin_region(res.operation)
                # CHECK: linalg.batch_matmul ins(%[[Amem]], %[[Bmem]] : memref<2x4x8xf32>, memref<2x8x12xf32>) outs(%[[Cmem]] : memref<2x4x12xf32>)
                linalg.batch_matmul(Amem, Bmem, outs=(Cmem,))

                # CHECK: linalg.batch_matmul indexing_maps = [#[[$A_MAP]], #[[$BTrans_MAP]], #[[$C_MAP]]] ins(%[[Amem]], %[[BTransmem]] : memref<2x4x8xf32>, memref<2x12x8xf32>) outs(%[[Cmem]] : memref<2x4x12xf32>)
                res = linalg.BatchMatmulOp(
                    result_tensors=[],
                    inputs=(Amem, Btransposedmem),
                    outputs=(Cmem,),
                    indexing_maps=[a_map, b_transposed_map, c_map],
                )
                linalg.fill_builtin_region(res.operation)
                # CHECK: linalg.batch_matmul indexing_maps = [#[[$A_MAP]], #[[$BTrans_MAP]], #[[$C_MAP]]] ins(%[[Amem]], %[[BTransmem]] : memref<2x4x8xf32>, memref<2x12x8xf32>) outs(%[[Cmem]] : memref<2x4x12xf32>)
                linalg.batch_matmul(
                    Amem,
                    Btransposedmem,
                    outs=(Cmem,),
                    indexing_maps=[a_map, b_transposed_map, c_map],
                )

    print(module)


# CHECK-LABEL: TEST: testBatchReduceMatmulOp
@run
def testBatchReduceMatmulOp():
    with Context(), Location.unknown():
        module = Module.create()
        f32 = F32Type.get()
        with InsertionPoint(module.body):
            a_shape = (5, 4, 8)
            b_shape = (5, 8, 12)
            b_transposed_shape = (5, 12, 8)
            c_shape = (4, 12)

            dimBatch = ir.AffineDimExpr.get(0)
            dimM = ir.AffineDimExpr.get(1)
            dimN = ir.AffineDimExpr.get(2)
            dimK = ir.AffineDimExpr.get(3)

            # CHECK: #[[$A_MAP:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
            # CHECK: #[[$BTrans_MAP:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
            # CHECK: #[[$C_MAP:.*]] = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
            a_map = ir.AffineMap.get(4, 0, [dimBatch, dimM, dimK])
            b_transposed_map = ir.AffineMap.get(4, 0, [dimBatch, dimN, dimK])
            c_map = ir.AffineMap.get(4, 0, [dimM, dimN])

            # CHECK: func.func @batch_reduce_matmul_op(
            @func.FuncOp.from_py_func(
                # CHECK-SAME:                         %[[A:.*]]: tensor<5x4x8xf32>,
                RankedTensorType.get(a_shape, f32),
                # CHECK-SAME:                         %[[Amem:.*]]: memref<5x4x8xf32>,
                MemRefType.get(a_shape, f32),
                # CHECK-SAME:                         %[[B:.*]]: tensor<5x8x12xf32>,
                RankedTensorType.get(b_shape, f32),
                # CHECK-SAME:                         %[[Bmem:.*]]: memref<5x8x12xf32>,
                MemRefType.get(b_shape, f32),
                # CHECK-SAME:                         %[[BTrans:.*]]: tensor<5x12x8xf32>,
                RankedTensorType.get(b_transposed_shape, f32),
                # CHECK-SAME:                         %[[BTransmem:.*]]: memref<5x12x8xf32>,
                MemRefType.get(b_transposed_shape, f32),
                # CHECK-SAME:                         %[[C:.*]]: tensor<4x12xf32>,
                RankedTensorType.get(c_shape, f32),
                # CHECK-SAME:                         %[[Cmem:.*]]: memref<4x12xf32>)
                MemRefType.get(c_shape, f32),
            )
            def batch_reduce_matmul_op(
                A, Amem, B, Bmem, Btransposed, Btransposedmem, C, Cmem
            ):
                # CHECK: linalg.batch_reduce_matmul ins(%[[A]], %[[B]] : tensor<5x4x8xf32>, tensor<5x8x12xf32>) outs(%[[C]] : tensor<4x12xf32>)
                res = linalg.BatchReduceMatmulOp(
                    result_tensors=(C.type,),
                    inputs=(A, B),
                    outputs=(C,),
                )
                linalg.fill_builtin_region(res.operation)
                # CHECK: linalg.batch_reduce_matmul ins(%[[A]], %[[B]] : tensor<5x4x8xf32>, tensor<5x8x12xf32>) outs(%[[C]] : tensor<4x12xf32>)
                res = linalg.batch_reduce_matmul(A, B, outs=(C,))

                # CHECK: linalg.batch_reduce_matmul indexing_maps = [#[[$A_MAP]], #[[$BTrans_MAP]], #[[$C_MAP]]] ins(%[[A]], %[[BTrans]] : tensor<5x4x8xf32>, tensor<5x12x8xf32>) outs(%[[C]] : tensor<4x12xf32>)
                res = linalg.BatchReduceMatmulOp(
                    result_tensors=(C.type,),
                    inputs=(A, Btransposed),
                    outputs=(C,),
                    indexing_maps=[a_map, b_transposed_map, c_map],
                )
                linalg.fill_builtin_region(res.operation)
                # CHECK: linalg.batch_reduce_matmul indexing_maps = [#[[$A_MAP]], #[[$BTrans_MAP]], #[[$C_MAP]]] ins(%[[A]], %[[BTrans]] : tensor<5x4x8xf32>, tensor<5x12x8xf32>) outs(%[[C]] : tensor<4x12xf32>)
                res = linalg.batch_reduce_matmul(
                    A,
                    Btransposed,
                    outs=(C,),
                    indexing_maps=[a_map, b_transposed_map, c_map],
                )

                # CHECK: linalg.batch_reduce_matmul ins(%[[Amem]], %[[Bmem]] : memref<5x4x8xf32>, memref<5x8x12xf32>) outs(%[[Cmem]] : memref<4x12xf32>)
                res = linalg.BatchReduceMatmulOp(
                    result_tensors=[],
                    inputs=(Amem, Bmem),
                    outputs=(Cmem,),
                )
                linalg.fill_builtin_region(res.operation)
                # CHECK: linalg.batch_reduce_matmul ins(%[[Amem]], %[[Bmem]] : memref<5x4x8xf32>, memref<5x8x12xf32>) outs(%[[Cmem]] : memref<4x12xf32>)
                linalg.batch_reduce_matmul(Amem, Bmem, outs=(Cmem,))

                # CHECK: linalg.batch_reduce_matmul indexing_maps = [#[[$A_MAP]], #[[$BTrans_MAP]], #[[$C_MAP]]] ins(%[[Amem]], %[[BTransmem]] : memref<5x4x8xf32>, memref<5x12x8xf32>) outs(%[[Cmem]] : memref<4x12xf32>)
                res = linalg.BatchReduceMatmulOp(
                    result_tensors=[],
                    inputs=(Amem, Btransposedmem),
                    outputs=(Cmem,),
                    indexing_maps=[a_map, b_transposed_map, c_map],
                )
                linalg.fill_builtin_region(res.operation)
                # CHECK: linalg.batch_reduce_matmul indexing_maps = [#[[$A_MAP]], #[[$BTrans_MAP]], #[[$C_MAP]]] ins(%[[Amem]], %[[BTransmem]] : memref<5x4x8xf32>, memref<5x12x8xf32>) outs(%[[Cmem]] : memref<4x12xf32>)
                linalg.batch_reduce_matmul(
                    Amem,
                    Btransposedmem,
                    outs=(Cmem,),
                    indexing_maps=[a_map, b_transposed_map, c_map],
                )

        print(module)


# CHECK-LABEL: TEST: testPackUnPackOp
@run
def testPackUnPackOp():
    with Context(), Location.unknown():
        module = Module.create()
        f32 = F32Type.get()
        with InsertionPoint(module.body):

            @func.FuncOp.from_py_func(
                RankedTensorType.get((128, 128), f32),
                RankedTensorType.get((16, 16, 8, 8), f32),
            )
            def tensor_pack(src, dst):
                packed = linalg.pack(
                    src,
                    dst,
                    inner_dims_pos=[1, 0],
                    inner_tiles=[8, 8],
                    padding_value=arith.constant(f32, 0.0),
                )

                unpacked = linalg.unpack(
                    packed,
                    src,
                    inner_dims_pos=[0, 1],
                    inner_tiles=[8, 8],
                )

                return unpacked

        # CHECK-LABEL:   func.func @tensor_pack(
        # CHECK-SAME:      %[[VAL_0:.*]]: tensor<128x128xf32>, %[[VAL_1:.*]]: tensor<16x16x8x8xf32>) -> tensor<128x128xf32> {
        # CHECK:           %[[VAL_2:.*]] = arith.constant 0.000000e+00 : f32
        # CHECK:           %[[VAL_3:.*]] = linalg.pack %[[VAL_0]] padding_value(%[[VAL_2]] : f32) inner_dims_pos = [1, 0] inner_tiles = [8, 8] into %[[VAL_1]] : tensor<128x128xf32> -> tensor<16x16x8x8xf32>
        # CHECK:           %[[VAL_4:.*]] = linalg.unpack %[[VAL_3]] inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %[[VAL_0]] : tensor<16x16x8x8xf32> -> tensor<128x128xf32>
        # CHECK:           return %[[VAL_4]] : tensor<128x128xf32>
        # CHECK:         }
        print(module)


# CHECK-LABEL: TEST: testElementwiseOp
@run
def testElementwiseOp():
    with Context(), Location.unknown():
        module = Module.create()
        f32 = F32Type.get()
        with InsertionPoint(module.body):
            rect_shape = (8, 16)
            vert_line_shape = (8,)
            hor_line_shape = (16,)
            transposed_rect_shape = (16, 8)

            # CHECK-DAG: #[[$IdentMap2D:.*]] = affine_map<(d0, d1) -> (d0, d1)>
            # CHECK-DAG: #[[$TransMap2D:.*]] = affine_map<(d0, d1) -> (d1, d0)>
            # CHECK-DAG: #[[$VertLineBCastMap:.*]] = affine_map<(d0, d1) -> (d0)>
            # CHECK-DAG: #[[$HorLineBCastMap:.*]] = affine_map<(d0, d1) -> (d1)>

            ident_map_2d = AffineMap.get_identity(2)
            transposed_map_2d = AffineMap.get_permutation((1, 0))
            vert_line_bcast_map = AffineMap.get(2, 0, [AffineDimExpr.get(0)])
            hor_line_bcast_map = AffineMap.get(2, 0, [AffineDimExpr.get(1)])

            # CHECK: func.func @elementwise_op(
            @func.FuncOp.from_py_func(
                # CHECK-SAME:                         %[[Rect:.*]]: tensor<8x16xf32>,
                RankedTensorType.get(rect_shape, f32),
                # CHECK-SAME:                         %[[RectMem:.*]]: memref<8x16xf32>,
                MemRefType.get(rect_shape, f32),
                # CHECK-SAME:                         %[[VertLine:.*]]: tensor<8xf32>,
                RankedTensorType.get(vert_line_shape, f32),
                # CHECK-SAME:                         %[[VertLineMem:.*]]: memref<8xf32>,
                MemRefType.get(vert_line_shape, f32),
                # CHECK-SAME:                         %[[HorLine:.*]]: tensor<16xf32>,
                RankedTensorType.get(hor_line_shape, f32),
                # CHECK-SAME:                         %[[HorLineMem:.*]]: memref<16xf32>,
                MemRefType.get(hor_line_shape, f32),
                # CHECK-SAME:                         %[[TransRect:.*]]: tensor<16x8xf32>,
                RankedTensorType.get(transposed_rect_shape, f32),
                # CHECK-SAME:                         %[[TransRectMem:.*]]: memref<16x8xf32>)
                MemRefType.get(transposed_rect_shape, f32),
            )
            def elementwise_op(
                rect,
                rect_mem,
                vert_line,
                vert_line_mem,
                hor_line,
                hor_line_mem,
                trans_rect,
                trans_rect_mem,
            ):
                # CHECK: %[[OutRect:.*]] = tensor.empty() : tensor<8x16xf32>
                out_rect = tensor.EmptyOp(rect_shape, f32)
                # CHECK: %[[OutRectMem:.*]] = memref.alloca() : memref<8x16xf32>
                out_rect_mem = memref.alloca(MemRefType.get(rect_shape, f32), [], [])

                if _inferred_affine_maps := True:
                    # CHECK: linalg.elementwise
                    # CHECK-SAME: kind=#linalg.elementwise_kind<exp>
                    # CHECK-SAME: ins(%[[Rect]] : tensor<8x16xf32>)
                    # CHECK-SAME: outs(%[[OutRect]] : tensor<8x16xf32>) -> tensor<8x16xf32>
                    op1 = linalg.ElementwiseOp(
                        result_tensors=(out_rect.result.type,),
                        inputs=(rect,),
                        outputs=(out_rect,),
                        kind=linalg.ElementwiseKind.exp,
                    )
                    linalg.fill_builtin_region(op1.operation)

                    # CHECK: linalg.elementwise
                    # CHECK-SAME: kind=#linalg.elementwise_kind<exp>
                    # CHECK-SAME: ins(%[[Rect]] : tensor<8x16xf32>)
                    # CHECK-SAME: outs(%[[OutRect]] : tensor<8x16xf32>) -> tensor<8x16xf32>
                    linalg.elementwise(
                        rect,
                        outs=(out_rect,),
                        kind=linalg.ElementwiseKind.exp,
                    )

                    # CHECK: linalg.elementwise
                    # CHECK-SAME: kind=#linalg.elementwise_kind<exp>
                    # CHECK-SAME: ins(%[[RectMem]] : memref<8x16xf32>)
                    # CHECK-SAME: outs(%[[OutRectMem]] : memref<8x16xf32>)
                    linalg.elementwise(
                        rect_mem,
                        outs=(out_rect_mem,),
                        kind=linalg.ElementwiseKind.exp,
                    )

                if _explicit_ident_affine_maps := True:
                    # Same as above but with default identity indexing_maps explicitly provided.
                    # CHECK: linalg.elementwise
                    # CHECK-SAME: kind=#linalg.elementwise_kind<exp>
                    # CHECK-SAME: ins(%[[Rect]] : tensor<8x16xf32>)
                    # CHECK-SAME: outs(%[[OutRect]] : tensor<8x16xf32>) -> tensor<8x16xf32>
                    op3 = linalg.ElementwiseOp(
                        result_tensors=(out_rect.result.type,),
                        inputs=(rect,),
                        outputs=(out_rect,),
                        kind=linalg.ElementwiseKind.exp,
                        indexing_maps=[ident_map_2d, ident_map_2d],
                    )
                    linalg.fill_builtin_region(op3.operation)

                    # CHECK: linalg.elementwise
                    # CHECK-SAME: kind=#linalg.elementwise_kind<exp>
                    # CHECK-SAME: ins(%[[RectMem]] : memref<8x16xf32>)
                    # CHECK-SAME: outs(%[[OutRectMem]] : memref<8x16xf32>)
                    linalg.elementwise(
                        rect_mem,
                        outs=(out_rect_mem,),
                        kind=linalg.ElementwiseKind.exp,
                        indexing_maps=[ident_map_2d, ident_map_2d],
                    )

                if _ops_with_non_ident_input_maps := True:
                    # CHECK: linalg.elementwise kind=#linalg.elementwise_kind<exp>
                    # CHECK-SAME: indexing_maps = [#[[$VertLineBCastMap]], #[[$IdentMap2D]]]
                    # CHECK-SAME: ins(%[[VertLine]] : tensor<8xf32>)
                    # CHECK-SAME: outs(%[[OutRect]] : tensor<8x16xf32>) -> tensor<8x16xf32>
                    op4 = linalg.ElementwiseOp(
                        result_tensors=(out_rect.result.type,),
                        inputs=(vert_line,),
                        outputs=(out_rect,),
                        kind=linalg.ElementwiseKind.exp,
                        indexing_maps=[vert_line_bcast_map, ident_map_2d],
                    )
                    linalg.fill_builtin_region(op4.operation)

                    # CHECK: linalg.elementwise kind=#linalg.elementwise_kind<add>
                    # CHECK-SAME: indexing_maps = [#[[$IdentMap2D]], #[[$VertLineBCastMap]], #[[$IdentMap2D]]]
                    # CHECK-SAME: ins(%[[Rect]], %[[VertLine]] : tensor<8x16xf32>, tensor<8xf32>)
                    # CHECK-SAME: outs(%[[OutRect]] : tensor<8x16xf32>) -> tensor<8x16xf32>
                    op4 = linalg.ElementwiseOp(
                        result_tensors=(out_rect.result.type,),
                        inputs=(rect, vert_line),
                        outputs=(out_rect,),
                        kind=linalg.ElementwiseKind.add,
                        indexing_maps=[ident_map_2d, vert_line_bcast_map, ident_map_2d],
                    )
                    linalg.fill_builtin_region(op4.operation)

                    # CHECK: linalg.elementwise kind=#linalg.elementwise_kind<div>
                    # CHECK-SAME: indexing_maps = [#[[$VertLineBCastMap]], #[[$HorLineBCastMap]], #[[$IdentMap2D]]]
                    # CHECK-SAME: ins(%[[VertLine]], %[[HorLine]] : tensor<8xf32>, tensor<16xf32>)
                    # CHECK-SAME: outs(%[[OutRect]] : tensor<8x16xf32>) -> tensor<8x16xf32>
                    linalg.elementwise(
                        vert_line,
                        hor_line,
                        outs=(out_rect,),
                        kind=linalg.ElementwiseKind.div,
                        indexing_maps=[
                            vert_line_bcast_map,
                            hor_line_bcast_map,
                            ident_map_2d,
                        ],
                    )

                if _ops_with_non_ident_and_transposed_input_maps := True:
                    # CHECK: %[[VertLineBoolsMem:.*]] = memref.alloca() : memref<8xi1>
                    vert_line_bools_mem = memref.alloca(
                        MemRefType.get(vert_line_shape, IntegerType.get_signless(1)),
                        [],
                        [],
                    )
                    # CHECK: linalg.elementwise kind=#linalg.elementwise_kind<select>
                    # CHECK-SAME: indexing_maps = [#[[$VertLineBCastMap]], #[[$HorLineBCastMap]], #[[$TransMap2D]], #[[$IdentMap2D]]]
                    # CHECK-SAME: ins(%[[VertLineBoolsMem]], %[[HorLineMem]], %[[TransRectMem]] : memref<8xi1>, memref<16xf32>, memref<16x8xf32>)
                    # CHECK-SAME: outs(%[[OutRectMem]] : memref<8x16xf32>)
                    linalg.elementwise(
                        vert_line_bools_mem,
                        hor_line_mem,
                        trans_rect_mem,
                        outs=(out_rect_mem,),
                        kind=linalg.ElementwiseKind.select,
                        indexing_maps=[
                            vert_line_bcast_map,
                            hor_line_bcast_map,
                            transposed_map_2d,
                            ident_map_2d,
                        ],
                    )

        print(module)


@run
def testReduceOp():
    with Context(), Location.unknown():
        f32 = T.f32()
        tensor_type = T.tensor(10, f32)

        @builtin.module
        def module():
            @func.func(tensor_type)
            def reduce_op(input):
                c1 = arith.constant(f32, 1.0)
                single_result = ir.RankedTensorType.get((), f32)
                dims = ir.DenseI64ArrayAttr.get([0])
                init = tensor.splat(single_result, c1, [])

                @linalg.reduce(
                    result=[single_result],
                    inputs=[input],
                    inits=[init],
                    dimensions=dims,
                )
                def reduced(element: f32, acc: f32):
                    return arith.mulf(acc, element)

                return tensor.extract(reduced, [])

        print(module)


# CHECK-LABEL:   func.func @reduce_op(
# CHECK-SAME:      %[[ARG0:.*]]: tensor<10xf32>) -> f32 {
# CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1.000000e+00 : f32
# CHECK:           %[[SPLAT_0:.*]] = tensor.splat %[[CONSTANT_0]] : tensor<f32>
# CHECK:           %[[REDUCE_0:.*]] = linalg.reduce { arith.mulf } ins(%[[ARG0]] : tensor<10xf32>) outs(%[[SPLAT_0]] : tensor<f32>) dimensions = [0]
# CHECK:           %[[EXTRACT_0:.*]] = tensor.extract %[[REDUCE_0]][] : tensor<f32>
# CHECK:           return %[[EXTRACT_0]] : f32
# CHECK:         }


@run
def testMapOp():
    with Context(), Location.unknown():
        f32 = T.f32()
        tensor_type = T.tensor(10, f32)

        @builtin.module
        def module():
            @func.func(tensor_type)
            def map_op(input):
                empty = tensor.empty(tensor_type.shape, f32)

                @linalg.map(
                    result=[tensor_type],
                    inputs=[input, input],
                    init=empty,
                )
                def add(element: f32, acc: f32, init: f32):
                    return arith.addf(element, acc)

                return add

        module.verify()
        print(module)


# CHECK-LABEL:   func.func @map_op(
# CHECK-SAME:                      %[[ARG0:.*]]: tensor<10xf32>) -> tensor<10xf32> {
# CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<10xf32>
# CHECK:           %[[MAP_0:.*]] = linalg.map { arith.addf } ins(%[[ARG0]], %[[ARG0]] : tensor<10xf32>, tensor<10xf32>) outs(%[[EMPTY_0]] : tensor<10xf32>)
# CHECK:           return %[[MAP_0]] : tensor<10xf32>
# CHECK:         }
