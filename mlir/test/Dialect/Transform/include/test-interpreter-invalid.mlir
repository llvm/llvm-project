// RUN: mlir-opt %s --verify-diagnostics

module attributes {transform.with_named_sequence} {
  transform.named_sequence private @private_helper(%arg0: !transform.any_op {transform.readonly}) {
    // expected-error @below {{expected ','}}
    transform.test_print_remark_at_operand %arg0 "message" : !transform.any_op
  }
}
