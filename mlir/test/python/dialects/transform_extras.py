# RUN: %PYTHON %s | FileCheck %s

from typing import Callable
from mlir import ir
from mlir.dialects import scf
from mlir.dialects.transform import structured
from mlir.dialects.transform.extras import OpHandle, insert_transform_script


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
