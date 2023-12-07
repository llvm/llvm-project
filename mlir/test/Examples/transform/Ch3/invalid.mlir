// RUN: transform-opt-ch3 %s --test-transform-dialect-interpreter --split-input-file --verify-diagnostics

// expected-note @below {{offending operation}}
module {
  transform.sequence failures(suppress) {
  // expected-error @below {{expected the payload operation to implement CallOpInterface}}
  ^bb0(%arg0: !transform.my.call_op_interface):
    yield
  }
}
