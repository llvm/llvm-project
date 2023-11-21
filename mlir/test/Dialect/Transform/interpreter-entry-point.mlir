// RUN: mlir-opt %s -transform-interpreter=entry-point=entry_point \
// RUN:   -split-input-file -verify-diagnostics

module attributes { transform.with_named_sequence } {
  transform.named_sequence @entry_point(!transform.any_op {transform.readonly}) {
  ^bb0(%arg0: !transform.any_op):
    // expected-remark @below {{applying transformation}}
    transform.test_transform_op
    transform.yield
  }

  transform.named_sequence @__transform_main(!transform.any_op {transform.readonly}) {
  ^bb0(%arg0: !transform.any_op):
    transform.test_transform_op // Note: does not yield remark.
    transform.yield
  }
}
