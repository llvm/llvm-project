// RUN: mlir-opt %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @foo(%arg0: !transform.any_op) {
    transform.test_print_remark_at_operand %arg0, "message" : !transform.any_op
    transform.yield
  }
}
