// RUN: mlir-opt %s --transform-infer-effects | FileCheck %s

module attributes { transform.with_named_sequence } {
  // CHECK-LABEL: @infer
  // CHECK-SAME: %{{.*}}: !transform.any_op {transform.consumed}
  // CHECK-SAME: %{{.*}}: !transform.any_op {transform.readonly}
  // CHECK-SAME: %{{.*}}: !transform.param<i32> {transform.readonly}
  transform.named_sequence @infer(%op: !transform.any_op, %other: !transform.any_op, %param: !transform.param<i32>) {
    transform.test_consume_operand %op : !transform.any_op
    transform.debug.emit_remark_at %other, "" : !transform.any_op
    transform.yield
  }
}
