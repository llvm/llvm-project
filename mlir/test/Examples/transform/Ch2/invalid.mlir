// RUN: transform-opt-ch2 %s --test-transform-dialect-interpreter --split-input-file --verify-diagnostics

// expected-note @below {{offending payload}}
module {
  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    // expected-error @below {{only applies to func.call payloads}}
    transform.my.change_call_target %arg0, "updated" : !transform.any_op
    yield
  }
}
