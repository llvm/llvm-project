// RUN: mlir-opt %s --verify-diagnostics

// The only thing we check here is that it should fail to parse. The other
// check is in the preload test.

module attributes {transform.with_named_sequence} {
  transform.named_sequence private @private_helper(%arg0: !transform.any_op {transform.readonly}) {
    // expected-error @below {{expected ','}}
    transform.test_print_remark_at_operand %arg0 "should have ',' prior to this" : !transform.any_op
  }
}
