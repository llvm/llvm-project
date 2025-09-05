# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import arith
from mlir.dialects import func
from mlir.dialects import memref
from mlir.dialects import scf
from mlir.dialects import tensor


def constructAndPrintInModule(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            f()
        assert module.operation.verify()
        print(module)
    return f


# CHECK-LABEL: TEST: testSimpleForall
# CHECK: scf.forall (%[[IV0:.*]], %[[IV1:.*]]) in (4, 8) shared_outs(%[[BOUND_ARG:.*]] = %{{.*}}) -> (tensor<4x8xf32>)
# CHECK:   arith.addi %[[IV0]], %[[IV1]]
# CHECK:   scf.forall.in_parallel
@constructAndPrintInModule
def testSimpleForall():
    f32 = F32Type.get()
    tensor_type = RankedTensorType.get([4, 8], f32)

    @func.FuncOp.from_py_func(tensor_type)
    def forall_loop(tensor):
        loop = scf.ForallOp([0, 0], [4, 8], [1, 1], [tensor])
        with InsertionPoint(loop.body):
            i, j = loop.induction_variables
            arith.addi(i, j)
            loop.terminator()
        # The verifier will check that the regions have been created properly.
        assert loop.verify()


# CHECK-LABEL: TEST: test_forall_insert_slice_no_region_with_for
@constructAndPrintInModule
def test_forall_insert_slice_no_region_with_for():
    i32 = IntegerType.get_signless(32)
    f32 = F32Type.get()
    ten = tensor.empty([10, 10], i32)

    for i, j, shared_outs in scf.forall([1, 1], [2, 2], [3, 3], shared_outs=[ten]):
        one = arith.constant(f32, 1.0)

        scf.parallel_insert_slice(
            ten,
            shared_outs,
            offsets=[i, j],
            sizes=[10, 10],
            strides=[1, 1],
        )

    # CHECK:  %[[VAL_0:.*]] = tensor.empty() : tensor<10x10xi32>
    # CHECK:  %[[VAL_1:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_2:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_3:.*]] = arith.constant 2 : index
    # CHECK:  %[[VAL_4:.*]] = arith.constant 2 : index
    # CHECK:  %[[VAL_5:.*]] = arith.constant 3 : index
    # CHECK:  %[[VAL_6:.*]] = arith.constant 3 : index
    # CHECK:  %[[VAL_7:.*]] = scf.forall (%[[VAL_8:.*]], %[[VAL_9:.*]]) = (%[[VAL_1]], %[[VAL_2]]) to (%[[VAL_3]], %[[VAL_4]]) step (%[[VAL_5]], %[[VAL_6]]) shared_outs(%[[VAL_10:.*]] = %[[VAL_0]]) -> (tensor<10x10xi32>) {
    # CHECK:    %[[VAL_11:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:    scf.forall.in_parallel {
    # CHECK:      tensor.parallel_insert_slice %[[VAL_0]] into %[[VAL_10]]{{\[}}%[[VAL_8]], %[[VAL_9]]] [10, 10] [1, 1] : tensor<10x10xi32> into tensor<10x10xi32>
    # CHECK:    }
    # CHECK:  }

    for ii, jj, shared_outs_1 in scf.forall([1, 1], [2, 2], [3, 3], shared_outs=[ten]):
        ten_dynamic = tensor.empty([ii, 10], i32)
        scf.parallel_insert_slice(
            ten_dynamic,
            shared_outs_1,
            offsets=[ii, 0],
            sizes=[ii, 10],
            strides=[ii, 1],
        )

    # CHECK:  %[[VAL_12:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_13:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_14:.*]] = arith.constant 2 : index
    # CHECK:  %[[VAL_15:.*]] = arith.constant 2 : index
    # CHECK:  %[[VAL_16:.*]] = arith.constant 3 : index
    # CHECK:  %[[VAL_17:.*]] = arith.constant 3 : index
    # CHECK:  %[[VAL_18:.*]] = scf.forall (%[[VAL_19:.*]], %[[VAL_20:.*]]) = (%[[VAL_12]], %[[VAL_13]]) to (%[[VAL_14]], %[[VAL_15]]) step (%[[VAL_16]], %[[VAL_17]]) shared_outs(%[[VAL_21:.*]] = %[[VAL_0]]) -> (tensor<10x10xi32>) {
    # CHECK:    %[[VAL_22:.*]] = tensor.empty(%[[VAL_19]]) : tensor<?x10xi32>
    # CHECK:    scf.forall.in_parallel {
    # CHECK:      tensor.parallel_insert_slice %[[VAL_22]] into %[[VAL_21]]{{\[}}%[[VAL_19]], 0] {{\[}}%[[VAL_19]], 10] {{\[}}%[[VAL_19]], 1] : tensor<?x10xi32> into tensor<10x10xi32>
    # CHECK:    }
    # CHECK:  }


# CHECK-LABEL: TEST: test_parange_inits_with_for
@constructAndPrintInModule
def test_parange_inits_with_for():
    i32 = IntegerType.get_signless(32)
    f32 = F32Type.get()
    tensor_type = RankedTensorType.get([10, 10], f32)
    ten = tensor.empty([10, 10], i32)

    for i, j in scf.parallel([1, 1], [2, 2], [3, 3], inits=[ten]):
        one = arith.constant(f32, 1.0)
        ten2 = tensor.empty([10, 10], i32)

        @scf.reduce(ten2)
        def res(lhs: tensor_type, rhs: tensor_type):
            return arith.addi(lhs, rhs)

    # CHECK:  %[[VAL_0:.*]] = tensor.empty() : tensor<10x10xi32>
    # CHECK:  %[[VAL_1:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_2:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_3:.*]] = arith.constant 2 : index
    # CHECK:  %[[VAL_4:.*]] = arith.constant 2 : index
    # CHECK:  %[[VAL_5:.*]] = arith.constant 3 : index
    # CHECK:  %[[VAL_6:.*]] = arith.constant 3 : index
    # CHECK:  %[[VAL_7:.*]] = scf.parallel (%[[VAL_8:.*]], %[[VAL_9:.*]]) = (%[[VAL_1]], %[[VAL_2]]) to (%[[VAL_3]], %[[VAL_4]]) step (%[[VAL_5]], %[[VAL_6]]) init (%[[VAL_0]]) -> tensor<10x10xi32> {
    # CHECK:    %[[VAL_10:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:    %[[VAL_11:.*]] = tensor.empty() : tensor<10x10xi32>
    # CHECK:    scf.reduce(%[[VAL_11]] : tensor<10x10xi32>) {
    # CHECK:    ^bb0(%[[VAL_12:.*]]: tensor<10x10xi32>, %[[VAL_13:.*]]: tensor<10x10xi32>):
    # CHECK:      %[[VAL_14:.*]] = arith.addi %[[VAL_12]], %[[VAL_13]] : tensor<10x10xi32>
    # CHECK:      scf.reduce.return %[[VAL_14]] : tensor<10x10xi32>
    # CHECK:    }
    # CHECK:  }


# CHECK-LABEL: TEST: test_parange_inits_with_for_with_two_reduce
@constructAndPrintInModule
def test_parange_inits_with_for_with_two_reduce():
    index_type = IndexType.get()
    one = arith.constant(index_type, 1)

    for i, j in scf.parallel([1, 1], [2, 2], [3, 3], inits=[one, one]):

        @scf.reduce(i, j, num_reductions=2)
        def res1(lhs: index_type, rhs: index_type):
            return arith.addi(lhs, rhs)

        @scf.another_reduce(res1)
        def res2(lhs: index_type, rhs: index_type):
            return arith.addi(lhs, rhs)

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_1:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_2:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_3:.*]] = arith.constant 2 : index
    # CHECK:  %[[VAL_4:.*]] = arith.constant 2 : index
    # CHECK:  %[[VAL_5:.*]] = arith.constant 3 : index
    # CHECK:  %[[VAL_6:.*]] = arith.constant 3 : index
    # CHECK:  %[[VAL_7:.*]]:2 = scf.parallel (%[[VAL_8:.*]], %[[VAL_9:.*]]) = (%[[VAL_1]], %[[VAL_2]]) to (%[[VAL_3]], %[[VAL_4]]) step (%[[VAL_5]], %[[VAL_6]]) init (%[[VAL_0]], %[[VAL_0]]) -> (index, index) {
    # CHECK:    scf.reduce(%[[VAL_8]], %[[VAL_9]] : index, index) {
    # CHECK:    ^bb0(%[[VAL_10:.*]]: index, %[[VAL_11:.*]]: index):
    # CHECK:      %[[VAL_12:.*]] = arith.addi %[[VAL_10]], %[[VAL_11]] : index
    # CHECK:      scf.reduce.return %[[VAL_12]] : index
    # CHECK:    }, {
    # CHECK:    ^bb0(%[[VAL_13:.*]]: index, %[[VAL_14:.*]]: index):
    # CHECK:      %[[VAL_15:.*]] = arith.addi %[[VAL_13]], %[[VAL_14]] : index
    # CHECK:      scf.reduce.return %[[VAL_15]] : index
    # CHECK:    }
    # CHECK:  }


# CHECK-LABEL: TEST: test_parange_inits_with_for_with_three_reduce
@constructAndPrintInModule
def test_parange_inits_with_for_with_three_reduce():
    index_type = IndexType.get()
    one = arith.constant(index_type, 1)

    for i, j, k in scf.parallel([1, 1, 1], [2, 2, 2], [3, 3, 3], inits=[one, one, one]):

        @scf.reduce(i, j, k, num_reductions=3)
        def res1(lhs: index_type, rhs: index_type):
            return arith.addi(lhs, rhs)

        @scf.another_reduce(res1)
        def res2(lhs: index_type, rhs: index_type):
            return arith.addi(lhs, rhs)

        @scf.another_reduce(res2)
        def res3(lhs: index_type, rhs: index_type):
            return arith.addi(lhs, rhs)

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_1:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_2:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_3:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_4:.*]] = arith.constant 2 : index
    # CHECK:  %[[VAL_5:.*]] = arith.constant 2 : index
    # CHECK:  %[[VAL_6:.*]] = arith.constant 2 : index
    # CHECK:  %[[VAL_7:.*]] = arith.constant 3 : index
    # CHECK:  %[[VAL_8:.*]] = arith.constant 3 : index
    # CHECK:  %[[VAL_9:.*]] = arith.constant 3 : index
    # CHECK:  %[[VAL_10:.*]]:3 = scf.parallel (%[[VAL_11:.*]], %[[VAL_12:.*]], %[[VAL_13:.*]]) = (%[[VAL_1]], %[[VAL_2]], %[[VAL_3]]) to (%[[VAL_4]], %[[VAL_5]], %[[VAL_6]]) step (%[[VAL_7]], %[[VAL_8]], %[[VAL_9]]) init (%[[VAL_0]], %[[VAL_0]], %[[VAL_0]]) -> (index, index, index) {
    # CHECK:    scf.reduce(%[[VAL_11]], %[[VAL_12]], %[[VAL_13]] : index, index, index) {
    # CHECK:    ^bb0(%[[VAL_14:.*]]: index, %[[VAL_15:.*]]: index):
    # CHECK:      %[[VAL_16:.*]] = arith.addi %[[VAL_14]], %[[VAL_15]] : index
    # CHECK:      scf.reduce.return %[[VAL_16]] : index
    # CHECK:    }, {
    # CHECK:    ^bb0(%[[VAL_17:.*]]: index, %[[VAL_18:.*]]: index):
    # CHECK:      %[[VAL_19:.*]] = arith.addi %[[VAL_17]], %[[VAL_18]] : index
    # CHECK:      scf.reduce.return %[[VAL_19]] : index
    # CHECK:    }, {
    # CHECK:    ^bb0(%[[VAL_20:.*]]: index, %[[VAL_21:.*]]: index):
    # CHECK:      %[[VAL_22:.*]] = arith.addi %[[VAL_20]], %[[VAL_21]] : index
    # CHECK:      scf.reduce.return %[[VAL_22]] : index
    # CHECK:    }
    # CHECK:  }


# CHECK-LABEL: TEST: testSimpleLoop
@constructAndPrintInModule
def testSimpleLoop():
    index_type = IndexType.get()

    @func.FuncOp.from_py_func(index_type, index_type, index_type)
    def simple_loop(lb, ub, step):
        loop = scf.ForOp(lb, ub, step, [lb, lb])
        with InsertionPoint(loop.body):
            scf.YieldOp(loop.inner_iter_args)
        return


# CHECK: func @simple_loop(%[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
# CHECK: scf.for %{{.*}} = %[[ARG0]] to %[[ARG1]] step %[[ARG2]]
# CHECK: iter_args(%[[I1:.*]] = %[[ARG0]], %[[I2:.*]] = %[[ARG0]])
# CHECK: scf.yield %[[I1]], %[[I2]]


# CHECK-LABEL: TEST: testInductionVar
@constructAndPrintInModule
def testInductionVar():
    index_type = IndexType.get()

    @func.FuncOp.from_py_func(index_type, index_type, index_type)
    def induction_var(lb, ub, step):
        loop = scf.ForOp(lb, ub, step, [lb])
        with InsertionPoint(loop.body):
            scf.YieldOp([loop.induction_variable])
        return


# CHECK: func @induction_var(%[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
# CHECK: scf.for %[[IV:.*]] = %[[ARG0]] to %[[ARG1]] step %[[ARG2]]
# CHECK: scf.yield %[[IV]]


# CHECK-LABEL: TEST: testForSugar
@constructAndPrintInModule
def testForSugar():
    index_type = IndexType.get()
    memref_t = MemRefType.get([10], index_type)
    range = scf.for_

    # CHECK:  func.func @range_loop_1(%[[VAL_0:.*]]: index, %[[VAL_1:.*]]: index, %[[VAL_2:.*]]: index, %[[VAL_3:.*]]: memref<10xindex>) {
    # CHECK:    scf.for %[[VAL_4:.*]] = %[[VAL_0]] to %[[VAL_1]] step %[[VAL_2]] {
    # CHECK:      %[[VAL_5:.*]] = arith.addi %[[VAL_4]], %[[VAL_4]] : index
    # CHECK:      memref.store %[[VAL_5]], %[[VAL_3]]{{\[}}%[[VAL_4]]] : memref<10xindex>
    # CHECK:    }
    # CHECK:    return
    # CHECK:  }
    @func.FuncOp.from_py_func(index_type, index_type, index_type, memref_t)
    def range_loop_1(lb, ub, step, memref_v):
        for i in range(lb, ub, step):
            add = arith.addi(i, i)
            memref.store(add, memref_v, [i])

            scf.yield_([])

    # CHECK:  func.func @range_loop_2(%[[VAL_0:.*]]: index, %[[VAL_1:.*]]: index, %[[VAL_2:.*]]: index, %[[VAL_3:.*]]: memref<10xindex>) {
    # CHECK:    %[[VAL_4:.*]] = arith.constant 10 : index
    # CHECK:    %[[VAL_5:.*]] = arith.constant 1 : index
    # CHECK:    scf.for %[[VAL_6:.*]] = %[[VAL_0]] to %[[VAL_4]] step %[[VAL_5]] {
    # CHECK:      %[[VAL_7:.*]] = arith.addi %[[VAL_6]], %[[VAL_6]] : index
    # CHECK:      memref.store %[[VAL_7]], %[[VAL_3]]{{\[}}%[[VAL_6]]] : memref<10xindex>
    # CHECK:    }
    # CHECK:    return
    # CHECK:  }
    @func.FuncOp.from_py_func(index_type, index_type, index_type, memref_t)
    def range_loop_2(lb, ub, step, memref_v):
        for i in range(lb, 10, 1):
            add = arith.addi(i, i)
            memref.store(add, memref_v, [i])
            scf.yield_([])

    # CHECK:  func.func @range_loop_3(%[[VAL_0:.*]]: index, %[[VAL_1:.*]]: index, %[[VAL_2:.*]]: index, %[[VAL_3:.*]]: memref<10xindex>) {
    # CHECK:    %[[VAL_4:.*]] = arith.constant 0 : index
    # CHECK:    %[[VAL_5:.*]] = arith.constant 1 : index
    # CHECK:    scf.for %[[VAL_6:.*]] = %[[VAL_4]] to %[[VAL_1]] step %[[VAL_5]] {
    # CHECK:      %[[VAL_7:.*]] = arith.addi %[[VAL_6]], %[[VAL_6]] : index
    # CHECK:      memref.store %[[VAL_7]], %[[VAL_3]]{{\[}}%[[VAL_6]]] : memref<10xindex>
    # CHECK:    }
    # CHECK:    return
    # CHECK:  }
    @func.FuncOp.from_py_func(index_type, index_type, index_type, memref_t)
    def range_loop_3(lb, ub, step, memref_v):
        for i in range(0, ub, 1):
            add = arith.addi(i, i)
            memref.store(add, memref_v, [i])
            scf.yield_([])

    # CHECK:  func.func @range_loop_4(%[[VAL_0:.*]]: index, %[[VAL_1:.*]]: index, %[[VAL_2:.*]]: index, %[[VAL_3:.*]]: memref<10xindex>) {
    # CHECK:    %[[VAL_4:.*]] = arith.constant 0 : index
    # CHECK:    %[[VAL_5:.*]] = arith.constant 10 : index
    # CHECK:    scf.for %[[VAL_6:.*]] = %[[VAL_4]] to %[[VAL_5]] step %[[VAL_2]] {
    # CHECK:      %[[VAL_7:.*]] = arith.addi %[[VAL_6]], %[[VAL_6]] : index
    # CHECK:      memref.store %[[VAL_7]], %[[VAL_3]]{{\[}}%[[VAL_6]]] : memref<10xindex>
    # CHECK:    }
    # CHECK:    return
    # CHECK:  }
    @func.FuncOp.from_py_func(index_type, index_type, index_type, memref_t)
    def range_loop_4(lb, ub, step, memref_v):
        for i in range(0, 10, step):
            add = arith.addi(i, i)
            memref.store(add, memref_v, [i])
            scf.yield_([])

    # CHECK:  func.func @range_loop_5(%[[VAL_0:.*]]: index, %[[VAL_1:.*]]: index, %[[VAL_2:.*]]: index, %[[VAL_3:.*]]: memref<10xindex>) {
    # CHECK:    %[[VAL_4:.*]] = arith.constant 0 : index
    # CHECK:    %[[VAL_5:.*]] = arith.constant 10 : index
    # CHECK:    %[[VAL_6:.*]] = arith.constant 1 : index
    # CHECK:    scf.for %[[VAL_7:.*]] = %[[VAL_4]] to %[[VAL_5]] step %[[VAL_6]] {
    # CHECK:      %[[VAL_8:.*]] = arith.addi %[[VAL_7]], %[[VAL_7]] : index
    # CHECK:      memref.store %[[VAL_8]], %[[VAL_3]]{{\[}}%[[VAL_7]]] : memref<10xindex>
    # CHECK:    }
    # CHECK:    return
    # CHECK:  }
    @func.FuncOp.from_py_func(index_type, index_type, index_type, memref_t)
    def range_loop_5(lb, ub, step, memref_v):
        for i in range(0, 10, 1):
            add = arith.addi(i, i)
            memref.store(add, memref_v, [i])
            scf.yield_([])

    # CHECK:  func.func @range_loop_6(%[[VAL_0:.*]]: index, %[[VAL_1:.*]]: index, %[[VAL_2:.*]]: index, %[[VAL_3:.*]]: memref<10xindex>) {
    # CHECK:    %[[VAL_4:.*]] = arith.constant 0 : index
    # CHECK:    %[[VAL_5:.*]] = arith.constant 10 : index
    # CHECK:    %[[VAL_6:.*]] = arith.constant 1 : index
    # CHECK:    scf.for %[[VAL_7:.*]] = %[[VAL_4]] to %[[VAL_5]] step %[[VAL_6]] {
    # CHECK:      %[[VAL_8:.*]] = arith.addi %[[VAL_7]], %[[VAL_7]] : index
    # CHECK:      memref.store %[[VAL_8]], %[[VAL_3]]{{\[}}%[[VAL_7]]] : memref<10xindex>
    # CHECK:    }
    # CHECK:    return
    # CHECK:  }
    @func.FuncOp.from_py_func(index_type, index_type, index_type, memref_t)
    def range_loop_6(lb, ub, step, memref_v):
        for i in range(0, 10):
            add = arith.addi(i, i)
            memref.store(add, memref_v, [i])
            scf.yield_([])

    # CHECK:  func.func @range_loop_7(%[[VAL_0:.*]]: index, %[[VAL_1:.*]]: index, %[[VAL_2:.*]]: index, %[[VAL_3:.*]]: memref<10xindex>) {
    # CHECK:    %[[VAL_4:.*]] = arith.constant 0 : index
    # CHECK:    %[[VAL_5:.*]] = arith.constant 10 : index
    # CHECK:    %[[VAL_6:.*]] = arith.constant 1 : index
    # CHECK:    scf.for %[[VAL_7:.*]] = %[[VAL_4]] to %[[VAL_5]] step %[[VAL_6]] {
    # CHECK:      %[[VAL_8:.*]] = arith.addi %[[VAL_7]], %[[VAL_7]] : index
    # CHECK:      memref.store %[[VAL_8]], %[[VAL_3]]{{\[}}%[[VAL_7]]] : memref<10xindex>
    # CHECK:    }
    # CHECK:    return
    # CHECK:  }
    @func.FuncOp.from_py_func(index_type, index_type, index_type, memref_t)
    def range_loop_7(lb, ub, step, memref_v):
        for i in range(10):
            add = arith.addi(i, i)
            memref.store(add, memref_v, [i])
            scf.yield_([])

    # CHECK:  func.func @loop_yield_1(%[[VAL_0:.*]]: index, %[[VAL_1:.*]]: index, %[[VAL_2:.*]]: index, %[[VAL_3:.*]]: memref<10xindex>) {
    # CHECK:    %[[VAL_4:.*]] = arith.constant 0 : index
    # CHECK:    %[[VAL_5:.*]] = arith.constant 0 : index
    # CHECK:    %[[VAL_6:.*]] = arith.constant 0 : index
    # CHECK:    %[[VAL_7:.*]] = arith.constant 100 : index
    # CHECK:    %[[VAL_8:.*]] = arith.constant 1 : index
    # CHECK:    %[[VAL_10:.*]] = scf.for %[[IV:.*]] = %[[VAL_6]] to %[[VAL_7]] step %[[VAL_8]] iter_args(%[[ITER:.*]] = %[[VAL_4]]) -> (index) {
    # CHECK:      %[[VAL_9:.*]] = arith.addi %[[ITER]], %[[IV]] : index
    # CHECK:      scf.yield %[[VAL_9]] : index
    # CHECK:    }
    # CHECK:    memref.store %[[VAL_10]], %[[VAL_3]]{{\[}}%[[VAL_5]]] : memref<10xindex>
    # CHECK:    return
    # CHECK:  }
    @func.FuncOp.from_py_func(index_type, index_type, index_type, memref_t)
    def loop_yield_1(lb, ub, step, memref_v):
        sum = arith.ConstantOp.create_index(0)
        c0 = arith.ConstantOp.create_index(0)
        for i, loc_sum, sum in scf.for_(0, 100, 1, [sum]):
            loc_sum = arith.addi(loc_sum, i)
            scf.yield_([loc_sum])
        memref.store(sum, memref_v, [c0])

    # CHECK:  func.func @loop_yield_2(%[[VAL_0:.*]]: index, %[[VAL_1:.*]]: index, %[[VAL_2:.*]]: index, %[[VAL_3:.*]]: memref<10xindex>) {
    # CHECK:    %[[c0:.*]] = arith.constant 0 : index
    # CHECK:    %[[c2:.*]] = arith.constant 2 : index
    # CHECK:    %[[REF1:.*]] = arith.constant 0 : index
    # CHECK:    %[[REF2:.*]] = arith.constant 1 : index
    # CHECK:    %[[VAL_6:.*]] = arith.constant 0 : index
    # CHECK:    %[[VAL_7:.*]] = arith.constant 100 : index
    # CHECK:    %[[VAL_8:.*]] = arith.constant 1 : index
    # CHECK:    %[[RES:.*]] = scf.for %[[IV:.*]] = %[[VAL_6]] to %[[VAL_7]] step %[[VAL_8]] iter_args(%[[ITER1:.*]] = %[[c0]], %[[ITER2:.*]] = %[[c2]]) -> (index, index) {
    # CHECK:      %[[VAL_9:.*]] = arith.addi %[[ITER1]], %[[IV]] : index
    # CHECK:      %[[VAL_10:.*]] = arith.addi %[[ITER2]], %[[IV]] : index
    # CHECK:      scf.yield %[[VAL_9]], %[[VAL_10]] : index, index
    # CHECK:    }
    # CHECK:    return
    # CHECK:  }
    @func.FuncOp.from_py_func(index_type, index_type, index_type, memref_t)
    def loop_yield_2(lb, ub, step, memref_v):
        sum1 = arith.ConstantOp.create_index(0)
        sum2 = arith.ConstantOp.create_index(2)
        c0 = arith.ConstantOp.create_index(0)
        c1 = arith.ConstantOp.create_index(1)
        for i, [loc_sum1, loc_sum2], [sum1, sum2] in scf.for_(0, 100, 1, [sum1, sum2]):
            loc_sum1 = arith.addi(loc_sum1, i)
            loc_sum2 = arith.addi(loc_sum2, i)
            scf.yield_([loc_sum1, loc_sum2])
        memref.store(sum1, memref_v, [c0])
        memref.store(sum2, memref_v, [c1])


@constructAndPrintInModule
def testOpsAsArguments():
    index_type = IndexType.get()
    callee = func.FuncOp("callee", ([], [index_type, index_type]), visibility="private")
    f = func.FuncOp("ops_as_arguments", ([], []))
    with InsertionPoint(f.add_entry_block()):
        lb = arith.ConstantOp.create_index(0)
        ub = arith.ConstantOp.create_index(42)
        step = arith.ConstantOp.create_index(2)
        iter_args = func.CallOp(callee, [])
        loop = scf.ForOp(lb, ub, step, iter_args)
        with InsertionPoint(loop.body):
            scf.YieldOp(loop.inner_iter_args)
        func.ReturnOp([])


# CHECK-LABEL: TEST: testOpsAsArguments
# CHECK: func private @callee() -> (index, index)
# CHECK: func @ops_as_arguments() {
# CHECK:   %[[LB:.*]] = arith.constant 0
# CHECK:   %[[UB:.*]] = arith.constant 42
# CHECK:   %[[STEP:.*]] = arith.constant 2
# CHECK:   %[[ARGS:.*]]:2 = call @callee()
# CHECK:   scf.for %arg0 = %c0 to %c42 step %c2
# CHECK:   iter_args(%{{.*}} = %[[ARGS]]#0, %{{.*}} = %[[ARGS]]#1)
# CHECK:     scf.yield %{{.*}}, %{{.*}}
# CHECK:   return


@constructAndPrintInModule
def testIfWithoutElse():
    bool = IntegerType.get_signless(1)
    i32 = IntegerType.get_signless(32)

    @func.FuncOp.from_py_func(bool)
    def simple_if(cond):
        if_op = scf.IfOp(cond)
        with InsertionPoint(if_op.then_block):
            one = arith.ConstantOp(i32, 1)
            add = arith.AddIOp(one, one)
            scf.YieldOp([])
        return


# CHECK: func @simple_if(%[[ARG0:.*]]: i1)
# CHECK: scf.if %[[ARG0:.*]]
# CHECK:   %[[ONE:.*]] = arith.constant 1
# CHECK:   %[[ADD:.*]] = arith.addi %[[ONE]], %[[ONE]]
# CHECK: return


@constructAndPrintInModule
def testNestedIf():
    bool = IntegerType.get_signless(1)
    i32 = IntegerType.get_signless(32)

    @func.FuncOp.from_py_func(bool, bool)
    def nested_if(b, c):
        if_op = scf.IfOp(b)
        with InsertionPoint(if_op.then_block) as ip:
            if_op = scf.IfOp(c, ip=ip)
            with InsertionPoint(if_op.then_block):
                one = arith.ConstantOp(i32, 1)
                add = arith.AddIOp(one, one)
                scf.YieldOp([])
            scf.YieldOp([])
        return


# CHECK: func @nested_if(%[[ARG0:.*]]: i1, %[[ARG1:.*]]: i1)
# CHECK: scf.if %[[ARG0:.*]]
# CHECK:   scf.if %[[ARG1:.*]]
# CHECK:     %[[ONE:.*]] = arith.constant 1
# CHECK:     %[[ADD:.*]] = arith.addi %[[ONE]], %[[ONE]]
# CHECK: return


@constructAndPrintInModule
def testIfWithElse():
    bool = IntegerType.get_signless(1)
    i32 = IntegerType.get_signless(32)

    @func.FuncOp.from_py_func(bool)
    def simple_if_else(cond):
        if_op = scf.IfOp(cond, [i32, i32], hasElse=True)
        with InsertionPoint(if_op.then_block):
            x_true = arith.ConstantOp(i32, 0)
            y_true = arith.ConstantOp(i32, 1)
            scf.YieldOp([x_true, y_true])
        with InsertionPoint(if_op.else_block):
            x_false = arith.ConstantOp(i32, 2)
            y_false = arith.ConstantOp(i32, 3)
            scf.YieldOp([x_false, y_false])
        add = arith.AddIOp(if_op.results[0], if_op.results[1])
        return


# CHECK: func @simple_if_else(%[[ARG0:.*]]: i1)
# CHECK: %[[RET:.*]]:2 = scf.if %[[ARG0:.*]]
# CHECK:   %[[ZERO:.*]] = arith.constant 0
# CHECK:   %[[ONE:.*]] = arith.constant 1
# CHECK:   scf.yield %[[ZERO]], %[[ONE]]
# CHECK: } else {
# CHECK:   %[[TWO:.*]] = arith.constant 2
# CHECK:   %[[THREE:.*]] = arith.constant 3
# CHECK:   scf.yield %[[TWO]], %[[THREE]]
# CHECK: arith.addi %[[RET]]#0, %[[RET]]#1
# CHECK: return
