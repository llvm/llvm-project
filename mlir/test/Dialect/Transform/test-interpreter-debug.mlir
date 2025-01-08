// RUN: mlir-opt %s --pass-pipeline="builtin.module(transform-interpreter{\
// RUN:         debug-payload-root-tag=payload \
// RUN:         entry-point=transform})" \
// RUN:   --allow-unregistered-dialect --split-input-file --verify-diagnostics

// expected-error @below {{could not find the operation with transform.target_tag="payload" attribute}}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @transform(%arg0: !transform.any_op) {
    transform.yield
  }
}

// -----

// expected-error @below {{could not find a nested named sequence with name: transform}}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @not_transform(%arg0: !transform.any_op) {
    transform.yield
  }

  module attributes {transform.target_tag="payload"} {}
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @transform(%arg0: !transform.any_op) {
    transform.debug.emit_remark_at %arg0, "payload" : !transform.any_op
    transform.yield
  }

  // This will not be executed.
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.debug.emit_remark_at %arg0, "some other text that is not printed" : !transform.any_op
    transform.yield
  }

  module {
    module {}
    // expected-remark @below {{payload}}
    module attributes {transform.target_tag="payload"} {}
    module {}
  }
}
