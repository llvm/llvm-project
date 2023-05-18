// RUN: mlir-opt %s

module attributes {transform.with_named_sequence} {
  // expected-note @below {{previously defined here}}
  transform.named_sequence @print_message(%arg0: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %arg0, "message" : !transform.any_op
    transform.yield
  }

  transform.named_sequence @consuming(%arg0: !transform.any_op {transform.consumed}) {
    transform.test_consume_operand %arg0 : !transform.any_op
    transform.yield
  }
}