// RUN: mlir-opt %s --pass-pipeline='builtin.module(transform-interpreter{\
// RUN:     debug-bind-trailing-args=^test.some_returning_op,^test.some_other_returning_op})' \
// RUN:             --split-input-file --verify-diagnostics

// Note that diagnostic checker will merge two diagnostics with the same message
// at the same location, so only check the remark once.
// 
// expected-remark @below {{first extra}}
// expected-note @below {{value handle points to an op result #0}}
// expected-note @below {{value handle points to an op result #1}}
%0:2 = "test.some_returning_op"() : () -> (i32, i64)

// expected-remark @below {{first extra}}
// expected-note @below {{value handle points to an op result #0}}
%1 = "test.some_returning_op"() : () -> index

// Note that diagnostic checker will merge two diagnostics with the same message
// at the same location, so only check the remark once.
// 
// expected-remark @below {{second extra}}
// expected-note @below {{value handle points to an op result #0}}
// expected-note @below {{value handle points to an op result #1}}
%2:2 = "test.some_other_returning_op"() : () -> (f32, f64)

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op, %arg1: !transform.any_value,
      %arg2: !transform.any_value) {
    transform.debug.emit_remark_at %arg1, "first extra" : !transform.any_value
    transform.debug.emit_remark_at %arg2, "second extra" : !transform.any_value
    transform.yield
  }
}

// -----

%0:2 = "test.some_returning_op"() : () -> (i32, i64)
%1 = "test.some_returning_op"() : () -> index

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      // expected-error @below {{wrong kind of value provided for top-level operation handle}}
      %arg0: !transform.any_op, %arg1: !transform.any_op, %arg2: !transform.any_value) {
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  // expected-error @below {{operation expects 1 extra value bindings, but 2 were provided to the interpreter}}
  transform.named_sequence @__transform_main(%arg0: !transform.any_op, %arg1: !transform.any_value) {
    transform.yield
  }
}
