// RUN: mlir-opt %s --pass-pipeline="builtin.module(test-transform-dialect-interpreter{debug-payload-root-tag=payload debug-transform-root-tag=transform})" \
// RUN:             --allow-unregistered-dialect --split-input-file --verify-diagnostics

// expected-error @below {{could not find the operation with transform.target_tag="payload" attribute}}
module {
  transform.sequence failures(suppress) {
  ^bb0(%arg0: !transform.any_op):
  }
}

// -----

// expected-error @below {{could not find the operation with transform.target_tag="transform" attribute}}
module {
  transform.sequence failures(suppress) {
  ^bb0(%arg0: !transform.any_op):
  }

  module attributes {transform.target_tag="payload"} {}
}

// -----

// expected-error @below {{more than one operation with transform.target_tag="transform" attribute}}
module {
  // expected-note @below {{first operation}}
  transform.sequence failures(propagate) attributes {transform.target_tag="transform"} {
  ^bb0(%arg0: !transform.any_op):
  }

  // expected-note @below {{other operation}}
  transform.sequence failures(propagate) attributes {transform.target_tag="transform"} {
  ^bb0(%arg0: !transform.any_op):
  }

  module attributes {transform.target_tag="payload"} {}
}

// -----

module {
  // expected-error @below {{expected the transform entry point to be a top-level transform op}}
  func.func private @foo() attributes {transform.target_tag="transform"}

  module attributes {transform.target_tag="payload"} {}
}

// -----

module {
  transform.sequence failures(suppress) attributes {transform.target_tag="transform"} {
  ^bb0(%arg0: !transform.any_op):
    transform.debug.emit_remark_at %arg0, "payload" : !transform.any_op
  }

  // This will not be executed because it's not tagged.
  transform.sequence failures(suppress)  {
  ^bb0(%arg0: !transform.any_op):
    transform.debug.emit_remark_at %arg0, "some other text that is not printed" : !transform.any_op
  }

  module {
    module {}
    // expected-remark @below {{payload}}
    module attributes {transform.target_tag="payload"} {}
    module {}
  }
}
