# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import arith
from mlir.dialects import func
from mlir.dialects import memref
from mlir.dialects import scf
from mlir.passmanager import PassManager


def constructAndPrintInModule(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            f()
        print(module)
    return f


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
