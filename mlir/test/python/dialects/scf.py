# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.extras import types as T
from mlir.dialects import (
    arith,
    func,
    memref,
    scf,
    cf,
)
from mlir.passmanager import PassManager


def constructAndPrintInModule(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            f()
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


@constructAndPrintInModule
def testIndexSwitch():
    i32 = T.i32()

    @func.FuncOp.from_py_func(T.index(), results=[i32])
    def index_switch(index):
        c1 = arith.constant(i32, 1)
        c0 = arith.constant(i32, 0)
        value = arith.constant(i32, 5)
        switch_op = scf.IndexSwitchOp([i32], index, range(3))

        assert switch_op.regions[0] == switch_op.default_region
        assert switch_op.regions[1] == switch_op.case_regions[0]
        assert switch_op.regions[1] == switch_op.case_region(0)
        assert len(switch_op.case_regions) == 3
        assert len(switch_op.regions) == 4

        with InsertionPoint(switch_op.default_block):
            cf.assert_(arith.constant(T.bool(), 0), "Whoops!")
            scf.yield_([c1])

        for i, block in enumerate(switch_op.case_blocks):
            with InsertionPoint(block):
                scf.yield_([arith.constant(i32, i)])

        func.return_([switch_op.results[0]])

    return index_switch


# CHECK-LABEL:   func.func @index_switch(
# CHECK-SAME:      %[[ARG0:.*]]: index) -> i32 {
# CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : i32
# CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : i32
# CHECK:           %[[CONSTANT_2:.*]] = arith.constant 5 : i32
# CHECK:           %[[INDEX_SWITCH_0:.*]] = scf.index_switch %[[ARG0]] -> i32
# CHECK:           case 0 {
# CHECK:             %[[CONSTANT_3:.*]] = arith.constant 0 : i32
# CHECK:             scf.yield %[[CONSTANT_3]] : i32
# CHECK:           }
# CHECK:           case 1 {
# CHECK:             %[[CONSTANT_4:.*]] = arith.constant 1 : i32
# CHECK:             scf.yield %[[CONSTANT_4]] : i32
# CHECK:           }
# CHECK:           case 2 {
# CHECK:             %[[CONSTANT_5:.*]] = arith.constant 2 : i32
# CHECK:             scf.yield %[[CONSTANT_5]] : i32
# CHECK:           }
# CHECK:           default {
# CHECK:             %[[CONSTANT_6:.*]] = arith.constant false
# CHECK:             cf.assert %[[CONSTANT_6]], "Whoops!"
# CHECK:             scf.yield %[[CONSTANT_0]] : i32
# CHECK:           }
# CHECK:           return %[[INDEX_SWITCH_0]] : i32
# CHECK:         }


@constructAndPrintInModule
def testIndexSwitchWithBodyBuilders():
    i32 = T.i32()

    @func.FuncOp.from_py_func(T.index(), results=[i32])
    def index_switch(index):
        c1 = arith.constant(i32, 1)
        c0 = arith.constant(i32, 0)
        value = arith.constant(i32, 5)

        def default_body_builder(switch_op):
            cf.assert_(arith.constant(T.bool(), 0), "Whoops!")
            scf.yield_([c1])

        def case_body_builder(switch_op, case_index: int, case_value: int):
            scf.yield_([arith.constant(i32, case_value)])

        result = scf.index_switch(
            results=[i32],
            arg=index,
            cases=range(3),
            case_body_builder=case_body_builder,
            default_body_builder=default_body_builder,
        )

        func.return_([result])

    return index_switch


# CHECK-LABEL:   func.func @index_switch(
# CHECK-SAME:      %[[ARG0:.*]]: index) -> i32 {
# CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : i32
# CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : i32
# CHECK:           %[[CONSTANT_2:.*]] = arith.constant 5 : i32
# CHECK:           %[[INDEX_SWITCH_0:.*]] = scf.index_switch %[[ARG0]] -> i32
# CHECK:           case 0 {
# CHECK:             %[[CONSTANT_3:.*]] = arith.constant 0 : i32
# CHECK:             scf.yield %[[CONSTANT_3]] : i32
# CHECK:           }
# CHECK:           case 1 {
# CHECK:             %[[CONSTANT_4:.*]] = arith.constant 1 : i32
# CHECK:             scf.yield %[[CONSTANT_4]] : i32
# CHECK:           }
# CHECK:           case 2 {
# CHECK:             %[[CONSTANT_5:.*]] = arith.constant 2 : i32
# CHECK:             scf.yield %[[CONSTANT_5]] : i32
# CHECK:           }
# CHECK:           default {
# CHECK:             %[[CONSTANT_6:.*]] = arith.constant false
# CHECK:             cf.assert %[[CONSTANT_6]], "Whoops!"
# CHECK:             scf.yield %[[CONSTANT_0]] : i32
# CHECK:           }
# CHECK:           return %[[INDEX_SWITCH_0]] : i32
# CHECK:         }
