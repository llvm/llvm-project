// RUN: mlir-opt %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) {
    transform.test_print_remark_at_operand %arg0, "message" : !transform.any_op
    transform.yield
  }

  transform.named_sequence @consuming(%arg0: !transform.any_op {transform.consumed}) {
    transform.test_consume_operand %arg0 : !transform.any_op
    transform.yield
  }

  transform.named_sequence @unannotated(%arg0: !transform.any_op) {
    transform.test_print_remark_at_operand %arg0, "unannotated" : !transform.any_op
    transform.yield
  }
}
