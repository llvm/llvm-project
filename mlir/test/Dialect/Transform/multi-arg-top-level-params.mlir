// RUN: mlir-opt %s --pass-pipeline='builtin.module(test-transform-dialect-interpreter{bind-first-extra-to-params=1,2,3 bind-second-extra-to-params=42,45})' \
// RUN:          --split-input-file --verify-diagnostics

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation, %arg1: !transform.param<i64>, %arg2: !transform.param<i64>):
  // expected-remark @below {{1 : i64, 2 : i64, 3 : i64}}
  transform.test_print_param %arg1 : !transform.param<i64>
  // expected-remark @below {{42 : i64, 45 : i64}}
  transform.test_print_param %arg2 : !transform.param<i64>
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation, %arg1: !transform.any_op, %arg2: !transform.param<i64>):
  // expected-error @above {{wrong kind of value provided for top-level operation handle}}
}

// -----

// expected-error @below {{operation expects 3 extra value bindings, but 2 were provided to the interpreter}}
transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation, %arg1: !transform.param<i64>, %arg2: !transform.param<i64>, %arg3: !transform.param<i64>):
}
