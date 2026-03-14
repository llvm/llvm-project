// RUN: transform-opt-ch2 %s --transform-interpreter --split-input-file \
// RUN:                      --verify-diagnostics

// expected-note @below {{offending payload}}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-error @below {{only applies to func.call payloads}}
    transform.my.change_call_target %arg0, "updated" : !transform.any_op
    transform.yield
  }
}
