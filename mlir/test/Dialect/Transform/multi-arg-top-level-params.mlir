// RUN: mlir-opt %s --pass-pipeline='builtin.module(transform-interpreter{\
// RUN:                             debug-bind-trailing-args=#1;2;3,#42;45})' \
// RUN:          --split-input-file --verify-diagnostics

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op, %arg1: !transform.param<i64>,
      %arg2: !transform.param<i64>) {
    // expected-remark @below {{1 : i64, 2 : i64, 3 : i64}}
    transform.debug.emit_param_as_remark %arg1 : !transform.param<i64>
    // expected-remark @below {{42 : i64, 45 : i64}}
    transform.debug.emit_param_as_remark %arg2 : !transform.param<i64>
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op, %arg1: !transform.any_op,
      // expected-error @above {{wrong kind of value provided for top-level operation handle}}
      %arg2: !transform.param<i64>) {
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  // expected-error @below {{operation expects 3 extra value bindings, but 2 were provided to the interpreter}}
  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op, %arg1: !transform.param<i64>,
      %arg2: !transform.param<i64>, %arg3: !transform.param<i64>) {
    transform.yield
  }
}
