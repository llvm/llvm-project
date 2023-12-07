# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import func
from mlir.dialects import arith
from mlir.dialects import memref
from mlir.dialects import affine


def constructAndPrintInModule(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            f()
        print(module)
    return f


# CHECK-LABEL: TEST: testAffineStoreOp
@constructAndPrintInModule
def testAffineStoreOp():
    f32 = F32Type.get()
    index_type = IndexType.get()
    memref_type_out = MemRefType.get([12, 12], f32)

    # CHECK: func.func @affine_store_test(%[[ARG0:.*]]: index) -> memref<12x12xf32> {
    @func.FuncOp.from_py_func(index_type)
    def affine_store_test(arg0):
        # CHECK: %[[O_VAR:.*]] = memref.alloc() : memref<12x12xf32>
        mem = memref.AllocOp(memref_type_out, [], []).result

        d0 = AffineDimExpr.get(0)
        s0 = AffineSymbolExpr.get(0)
        map = AffineMap.get(1, 1, [s0 * 3, d0 + s0 + 1])

        # CHECK: %[[A1:.*]] = arith.constant 2.100000e+00 : f32
        a1 = arith.ConstantOp(f32, 2.1)

        # CHECK: affine.store %[[A1]], %alloc[symbol(%[[ARG0]]) * 3, %[[ARG0]] + symbol(%[[ARG0]]) + 1] : memref<12x12xf32>
        affine.AffineStoreOp(a1, mem, indices=[arg0, arg0], map=map)

        return mem


# CHECK-LABEL: TEST: testAffineLoadOp
@constructAndPrintInModule
def testAffineLoadOp():
    f32 = F32Type.get()
    index_type = IndexType.get()
    memref_type_in = MemRefType.get([10, 10], f32)

    # CHECK: func.func @affine_load_test(%[[I_VAR:.*]]: memref<10x10xf32>, %[[ARG0:.*]]: index) -> f32 {
    @func.FuncOp.from_py_func(memref_type_in, index_type)
    def affine_load_test(I, arg0):
        d0 = AffineDimExpr.get(0)
        s0 = AffineSymbolExpr.get(0)
        map = AffineMap.get(1, 1, [s0 * 3, d0 + s0 + 1])

        # CHECK: {{.*}} = affine.load %[[I_VAR]][symbol(%[[ARG0]]) * 3, %[[ARG0]] + symbol(%[[ARG0]]) + 1] : memref<10x10xf32>
        a1 = affine.AffineLoadOp(f32, I, indices=[arg0, arg0], map=map)

        return a1


# CHECK-LABEL: TEST: testAffineForOp
@constructAndPrintInModule
def testAffineForOp():
    f32 = F32Type.get()
    index_type = IndexType.get()
    memref_type = MemRefType.get([1024], f32)

    # CHECK: #[[MAP0:.*]] = affine_map<(d0)[s0] -> (0, d0 + s0)>
    # CHECK: #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0 - 2, d1 * 32)>
    # CHECK: func.func @affine_for_op_test(%[[BUFFER:.*]]: memref<1024xf32>) {
    @func.FuncOp.from_py_func(memref_type)
    def affine_for_op_test(buffer):
        # CHECK: %[[C1:.*]] = arith.constant 1 : index
        c1 = arith.ConstantOp(index_type, 1)
        # CHECK: %[[C2:.*]] = arith.constant 2 : index
        c2 = arith.ConstantOp(index_type, 2)
        # CHECK: %[[C3:.*]] = arith.constant 3 : index
        c3 = arith.ConstantOp(index_type, 3)
        # CHECK: %[[C9:.*]] = arith.constant 9 : index
        c9 = arith.ConstantOp(index_type, 9)
        # CHECK: %[[AC0:.*]] = arith.constant 0.000000e+00 : f32
        ac0 = AffineConstantExpr.get(0)

        d0 = AffineDimExpr.get(0)
        d1 = AffineDimExpr.get(1)
        s0 = AffineSymbolExpr.get(0)
        lb = AffineMap.get(1, 1, [ac0, d0 + s0])
        ub = AffineMap.get(2, 0, [d0 - 2, 32 * d1])
        sum_0 = arith.ConstantOp(f32, 0.0)

        # CHECK: %0 = affine.for %[[INDVAR:.*]] = max #[[MAP0]](%[[C2]])[%[[C3]]] to min #[[MAP1]](%[[C9]], %[[C1]]) step 2 iter_args(%[[SUM0:.*]] = %[[AC0]]) -> (f32) {
        sum = affine.AffineForOp(
            lb,
            ub,
            2,
            iter_args=[sum_0],
            lower_bound_operands=[c2, c3],
            upper_bound_operands=[c9, c1],
        )

        with InsertionPoint(sum.body):
            # CHECK: %[[TMP:.*]] = memref.load %[[BUFFER]][%[[INDVAR]]] : memref<1024xf32>
            tmp = memref.LoadOp(buffer, [sum.induction_variable])
            sum_next = arith.AddFOp(sum.inner_iter_args[0], tmp)

            affine.AffineYieldOp([sum_next])

        return


@constructAndPrintInModule
def testForSugar():
    index_type = IndexType.get()
    memref_t = MemRefType.get([10], index_type)
    range = affine.for_

    # CHECK:  func.func @range_loop_1(%[[VAL_0:.*]]: index, %[[VAL_1:.*]]: index, %[[VAL_2:.*]]: index, %[[VAL_3:.*]]: memref<10xindex>) {
    # CHECK:    %[[VAL_4:.*]] = arith.constant 10 : index
    # CHECK:    affine.for %[[VAL_6:.*]] = %[[VAL_0]] to %[[VAL_4]] step 2 {
    # CHECK:      %[[VAL_7:.*]] = arith.addi %[[VAL_6]], %[[VAL_6]] : index
    # CHECK:      affine.store %[[VAL_7]], %[[VAL_3]]{{\[symbol\(}}%[[VAL_6]]{{\)\]}} : memref<10xindex>
    # CHECK:    }
    # CHECK:    return
    # CHECK:  }
    @func.FuncOp.from_py_func(index_type, index_type, index_type, memref_t)
    def range_loop_1(lb, ub, step, memref_v):
        for i in range(lb, 10, 2):
            add = arith.addi(i, i)
            s0 = AffineSymbolExpr.get(0)
            map = AffineMap.get(0, 1, [s0])
            affine.store(add, memref_v, [i], map=map)
            affine.AffineYieldOp([])

    # CHECK:  func.func @range_loop_2(%[[VAL_0:.*]]: index, %[[VAL_1:.*]]: index, %[[VAL_2:.*]]: index, %[[VAL_3:.*]]: memref<10xindex>) {
    # CHECK:    %[[VAL_4:.*]] = arith.constant 0 : index
    # CHECK:    %[[VAL_5:.*]] = arith.constant 10 : index
    # CHECK:    affine.for %[[VAL_7:.*]] = %[[VAL_4]] to %[[VAL_5]] {
    # CHECK:      %[[VAL_8:.*]] = arith.addi %[[VAL_7]], %[[VAL_7]] : index
    # CHECK:      affine.store %[[VAL_8]], %[[VAL_3]]{{\[symbol\(}}%[[VAL_7]]{{\)\]}} : memref<10xindex>
    # CHECK:    }
    # CHECK:    return
    # CHECK:  }
    @func.FuncOp.from_py_func(index_type, index_type, index_type, memref_t)
    def range_loop_2(lb, ub, step, memref_v):
        for i in range(0, 10, 1):
            add = arith.addi(i, i)
            s0 = AffineSymbolExpr.get(0)
            map = AffineMap.get(0, 1, [s0])
            affine.store(add, memref_v, [i], map=map)
            affine.AffineYieldOp([])

    # CHECK:  func.func @range_loop_3(%[[VAL_0:.*]]: index, %[[VAL_1:.*]]: index, %[[VAL_2:.*]]: index, %[[VAL_3:.*]]: memref<10xindex>) {
    # CHECK:    %[[VAL_4:.*]] = arith.constant 0 : index
    # CHECK:    affine.for %[[VAL_6:.*]] = %[[VAL_4]] to %[[VAL_1]] {
    # CHECK:      %[[VAL_7:.*]] = arith.addi %[[VAL_6]], %[[VAL_6]] : index
    # CHECK:      affine.store %[[VAL_7]], %[[VAL_3]]{{\[symbol\(}}%[[VAL_6]]{{\)\]}} : memref<10xindex>
    # CHECK:    }
    # CHECK:    return
    # CHECK:  }
    @func.FuncOp.from_py_func(index_type, index_type, index_type, memref_t)
    def range_loop_3(lb, ub, step, memref_v):
        for i in range(0, ub, 1):
            add = arith.addi(i, i)
            s0 = AffineSymbolExpr.get(0)
            map = AffineMap.get(0, 1, [s0])
            affine.store(add, memref_v, [i], map=map)
            affine.AffineYieldOp([])
