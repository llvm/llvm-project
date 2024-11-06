# RUN: %PYTHON %s 2>&1 | FileCheck %s

from mlir.passmanager import PassManager
from mlir.ir import Context, Location, Module, InsertionPoint, UnitAttr
from mlir.dialects import scf, pdl, func, arith, linalg
from mlir.dialects.transform import (
    get_parent_op,
    apply_patterns_canonicalization,
    apply_cse,
    any_op_t,
)
from mlir.dialects.transform.structured import structured_match
from mlir.dialects.transform.loop import loop_unroll
from mlir.dialects.transform.extras import named_sequence, apply_patterns
from mlir.extras import types as T
from mlir.dialects.builtin import module, ModuleOp


def construct_and_print_in_module(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            module = f(module)
        if module is not None:
            print(module)
    return f


# CHECK-LABEL: TEST: test_named_sequence
@construct_and_print_in_module
def test_named_sequence(module_):
    # CHECK-LABEL:   func.func @loop_unroll_op() {
    # CHECK:           %[[VAL_0:.*]] = arith.constant 0 : index
    # CHECK:           %[[VAL_1:.*]] = arith.constant 42 : index
    # CHECK:           %[[VAL_2:.*]] = arith.constant 5 : index
    # CHECK:           scf.for %[[VAL_3:.*]] = %[[VAL_0]] to %[[VAL_1]] step %[[VAL_2]] {
    # CHECK:             %[[VAL_4:.*]] = arith.addi %[[VAL_3]], %[[VAL_3]] : index
    # CHECK:           }
    # CHECK:           return
    # CHECK:         }
    @func.func()
    def loop_unroll_op():
        for i in scf.for_(0, 42, 5):
            v = arith.addi(i, i)
            scf.yield_([])

    # CHECK-LABEL:   module attributes {transform.with_named_sequence} {
    # CHECK:           transform.named_sequence @__transform_main(%[[VAL_0:.*]]: !transform.any_op) {
    # CHECK:             %[[VAL_1:.*]] = transform.structured.match ops{["arith.addi"]} in %[[VAL_0]] : (!transform.any_op) -> !transform.any_op
    # CHECK:             %[[VAL_2:.*]] = transform.get_parent_op %[[VAL_1]] {op_name = "scf.for"} : (!transform.any_op) -> !pdl.operation
    # CHECK:             transform.loop.unroll %[[VAL_2]] {factor = 4 : i64} : !pdl.operation
    # CHECK:             transform.yield
    # CHECK:           }
    # CHECK:         }
    @module(attrs={"transform.with_named_sequence": UnitAttr.get()})
    def mod():
        @named_sequence("__transform_main", [any_op_t()], [])
        def basic(target: any_op_t()):
            m = structured_match(any_op_t(), target, ops=["arith.addi"])
            loop = get_parent_op(pdl.op_t(), m, op_name="scf.for")
            loop_unroll(loop, 4)

    # The identifier (name) of the function becomes the Operation
    assert isinstance(mod.opview, ModuleOp)

    print(module_)

    pm = PassManager.parse("builtin.module(transform-interpreter)")
    pm.run(module_.operation)

    # CHECK-LABEL: func.func @loop_unroll_op() {
    # CHECK:         %[[VAL_0:.*]] = arith.constant 0 : index
    # CHECK:         %[[VAL_1:.*]] = arith.constant 42 : index
    # CHECK:         %[[VAL_2:.*]] = arith.constant 5 : index
    # CHECK:         %[[VAL_6:.*]] = arith.constant 40 : index
    # CHECK:         %[[VAL_7:.*]] = arith.constant 20 : index
    # CHECK:         scf.for %[[VAL_3:.*]] = %[[VAL_0]] to %[[VAL_6]] step %[[VAL_7]] {
    # CHECK:           %[[VAL_5:.*]] = arith.addi %[[VAL_3]], %[[VAL_3]] : index
    # CHECK:           %[[VAL_8:.*]] = arith.constant 1 : index
    # CHECK:           %[[VAL_9:.*]] = arith.muli %[[VAL_2]], %[[VAL_8]] : index
    # CHECK:           %[[VAL_10:.*]] = arith.addi %[[VAL_3]], %[[VAL_9]] : index
    # CHECK:           %[[VAL_11:.*]] = arith.addi %[[VAL_10]], %[[VAL_10]] : index
    # CHECK:           %[[VAL_12:.*]] = arith.constant 2 : index
    # CHECK:           %[[VAL_13:.*]] = arith.muli %[[VAL_2]], %[[VAL_12]] : index
    # CHECK:           %[[VAL_14:.*]] = arith.addi %[[VAL_3]], %[[VAL_13]] : index
    # CHECK:           %[[VAL_15:.*]] = arith.addi %[[VAL_14]], %[[VAL_14]] : index
    # CHECK:           %[[VAL_16:.*]] = arith.constant 3 : index
    # CHECK:           %[[VAL_17:.*]] = arith.muli %[[VAL_2]], %[[VAL_16]] : index
    # CHECK:           %[[VAL_18:.*]] = arith.addi %[[VAL_3]], %[[VAL_17]] : index
    # CHECK:           %[[VAL_19:.*]] = arith.addi %[[VAL_18]], %[[VAL_18]] : index
    # CHECK:         }
    # CHECK:         %[[VAL_4:.*]] = arith.addi %[[VAL_6]], %[[VAL_6]] : index
    # CHECK:         return
    # CHECK:       }
    print(module_)
