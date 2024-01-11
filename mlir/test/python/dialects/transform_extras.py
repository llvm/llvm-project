# RUN: %PYTHON %s | FileCheck %s

from typing import Callable
from mlir import ir
from mlir.dialects import scf, pdl
from mlir.dialects.transform import (
    structured,
    get_parent_op,
    apply_patterns_canonicalization,
    apply_cse,
    any_op_t,
)
from mlir.dialects.transform import FailurePropagationMode
from mlir.dialects.transform.structured import structured_match
from mlir.dialects.transform.loop import loop_unroll
from mlir.dialects.transform.extras import (
    OpHandle,
    insert_transform_script,
    sequence,
    apply_patterns,
)
from mlir.extras import types as T


def construct_and_print_in_module(f):
    print("\nTEST:", f.__name__)
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            f()
        print(module)
    return f


def build_transform_script(script: Callable[[OpHandle], None]):
    print("\nTEST:", script.__name__)
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        module.operation.attributes["transform.with_named_sequence"] = ir.UnitAttr.get()
        insert_transform_script(module.body, script=script, dump_script=True)
        module.operation.verify()


def build_transform_script_at_insertion_point(script: Callable[[OpHandle], None]):
    print("\nTEST:", script.__name__)
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        module.operation.attributes["transform.with_named_sequence"] = ir.UnitAttr.get()
        insert_transform_script(
            ir.InsertionPoint.at_block_begin(module.body),
            script=script,
            dump_script=True,
        )
        module.operation.verify()


# CHECK-LABEL: TEST: test_build_script_at_insertion_point
@build_transform_script_at_insertion_point
def test_build_script_at_insertion_point(op: OpHandle):
    pass
    # CHECK: transform.named_sequence {{.*}}(%[[VAL_0:.*]]: !transform.any_op) {
    # CHECK-NEXT: transform.yield
    # CHECK-NEXT: }


# CHECK-LABEL: TEST: test_match_ops_single
@build_transform_script
def test_match_ops_single(op: OpHandle):
    op.match_ops(scf.ForOp)
    # CHECK: transform.named_sequence {{.*}}(%[[VAL_0:.*]]: !transform.any_op) {
    # CHECK-NEXT: %[[VAL_1:.*]] = transform.structured.match ops{["scf.for"]}
    # CHECK-SAME:    in %[[VAL_0]]
    # CHECK-SAME:      -> !transform.op<"scf.for">


# CHECK-LABEL: TEST: test_match_ops_string_name
@build_transform_script
def test_match_ops_string_name(op: OpHandle):
    op.match_ops("linalg.matmul")
    # CHECK: transform.named_sequence {{.*}}(%[[VAL_0:.*]]: !transform.any_op) {
    # CHECK-NEXT: %[[VAL_1:.*]] = transform.structured.match
    # CHECK-SAME:   ops{["linalg.matmul"]} in %[[VAL_0]]


# CHECK-LABEL: TEST: test_match_ops_string_iface
@build_transform_script
def test_match_ops_string_iface(op: OpHandle):
    op.match_ops("LinalgOp")
    # CHECK: transform.named_sequence {{.*}}(%[[VAL_0:.*]]: !transform.any_op) {
    # CHECK-NEXT: %[[VAL_1:.*]] = transform.structured.match
    # CHECK-SAME:   interface{LinalgOp} in %[[VAL_0]]


# CHECK-LABEL: TEST: test_match_ops_iface
@build_transform_script
def test_match_ops_iface(op: OpHandle):
    op.match_ops(structured.MatchInterfaceEnum.LinalgOp)
    # CHECK: transform.named_sequence {{.*}}(%[[VAL_0:.*]]: !transform.any_op) {
    # CHECK-NEXT: %[[VAL_1:.*]] = transform.structured.match
    # CHECK-SAME:   interface{LinalgOp} in %[[VAL_0]]


# CHECK-LABEL: TEST: test_match_ops_multiple
@build_transform_script
def test_match_ops_multiple(op: OpHandle):
    op.match_ops([scf.ForOp, scf.ForallOp])
    # CHECK: transform.named_sequence {{.*}}(%[[VAL_0:.*]]: !transform.any_op) {
    # CHECK-NEXT: %[[VAL_1:.*]] = transform.structured.match
    # CHECK-SAME:   ops{["scf.for", "scf.forall"]} in %[[VAL_0]]
    # CHECK-SAME:     -> !transform.any_op


# CHECK-LABEL: TEST: test_match_ops_mixed
@build_transform_script
def test_match_ops_mixed(op: OpHandle):
    op.match_ops([scf.ForOp, "linalg.matmul", scf.ForallOp])
    # CHECK: transform.named_sequence {{.*}}(%[[VAL_0:.*]]: !transform.any_op) {
    # CHECK-NEXT: %[[VAL_1:.*]] = transform.structured.match
    # CHECK-SAME:   ops{["scf.for", "linalg.matmul", "scf.forall"]} in %[[VAL_0]]
    # CHECK-SAME:     -> !transform.any_op


# CHECK-LABEL: TEST: test_sequence_region
@construct_and_print_in_module
def test_sequence_region():
    # CHECK:   transform.sequence  failures(propagate) {
    # CHECK:   ^{{.*}}(%[[VAL_0:.*]]: !transform.any_op):
    # CHECK:     %[[VAL_1:.*]] = transform.structured.match ops{["arith.addi"]} in %[[VAL_0]] : (!transform.any_op) -> !transform.any_op
    # CHECK:     %[[VAL_2:.*]] = get_parent_op %[[VAL_1]] {op_name = "scf.for"} : (!transform.any_op) -> !pdl.operation
    # CHECK:     transform.loop.unroll %[[VAL_2]] {factor = 4 : i64} : !pdl.operation
    # CHECK:   }
    @sequence([], FailurePropagationMode.Propagate, [])
    def basic(target: any_op_t()):
        m = structured_match(any_op_t(), target, ops=["arith.addi"])
        loop = get_parent_op(pdl.op_t(), m, op_name="scf.for")
        loop_unroll(loop, 4)


# CHECK-LABEL: TEST: test_apply_patterns
@construct_and_print_in_module
def test_apply_patterns():
    # CHECK:   transform.sequence  failures(propagate) {
    # CHECK:   ^{{.*}}(%[[VAL_0:.*]]: !transform.any_op):
    # CHECK:     %[[VAL_1:.*]] = transform.structured.match ops{["linalg.matmul"]} in %[[VAL_0]] : (!transform.any_op) -> !transform.any_op
    # CHECK:     %[[VAL_2:.*]] = get_parent_op %[[VAL_1]] {op_name = "func.func"} : (!transform.any_op) -> !pdl.operation
    # CHECK:     apply_patterns to %[[VAL_2]] {
    # CHECK:       transform.apply_patterns.canonicalization
    # CHECK:     } : !pdl.operation
    # CHECK:     %[[VAL_3:.*]] = transform.structured.match ops{["func.func"]} in %[[VAL_0]] : (!transform.any_op) -> !transform.any_op
    # CHECK:     apply_cse to %[[VAL_3]] : !transform.any_op
    # CHECK:   }
    @sequence([], FailurePropagationMode.Propagate, [])
    def basic(variant_op: any_op_t()):
        matmul = structured_match(any_op_t(), variant_op, ops=["linalg.matmul"])
        top_func = get_parent_op(pdl.op_t(), matmul, op_name="func.func")

        @apply_patterns(top_func)
        def pats():
            apply_patterns_canonicalization()

        top_func = structured_match(any_op_t(), variant_op, ops=["func.func"])
        apply_cse(top_func)
