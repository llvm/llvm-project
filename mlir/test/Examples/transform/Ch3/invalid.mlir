// RUN: transform-opt-ch3 %s --transform-interpreter --split-input-file --verify-diagnostics

// expected-note @below {{offending operation}}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
  // expected-error @below {{expected the payload operation to implement CallOpInterface}}
  %arg0: !transform.my.call_op_interface) {
    transform.yield
  }
}
