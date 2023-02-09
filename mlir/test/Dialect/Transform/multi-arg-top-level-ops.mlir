// RUN: mlir-opt %s --pass-pipeline='builtin.module(test-transform-dialect-interpreter{bind-first-extra-to-ops=func.func bind-second-extra-to-ops=func.return})' \
// RUN:             --split-input-file --verify-diagnostics

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op, %arg1: !transform.any_op, %arg2: !transform.any_op):
  transform.test_print_remark_at_operand %arg1, "first extra" : !transform.any_op
  transform.test_print_remark_at_operand %arg2, "second extra" : !transform.any_op
}

// expected-remark @below {{first extra}}
func.func @foo() {
  // expected-remark @below {{second extra}}
  return
}

// expected-remark @below {{first extra}}
func.func @bar(%arg0: i1) {
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  // expected-remark @below {{second extra}}
  return
^bb2:
  // expected-remark @below {{second extra}}
  return
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op, %arg1: !transform.any_op, %arg2: !transform.param<i64>):
  // expected-error @above {{wrong kind of value provided for top-level parameter}}
}

func.func @foo() {
  return
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op, %arg1: !transform.any_op, %arg2: !transform.any_value):
  // expected-error @above {{wrong kind of value provided for the top-level value handle}}
}

func.func @foo() {
  return
}

// -----

// expected-error @below {{operation expects 1 extra value bindings, but 2 were provided to the interpreter}}
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op, %arg1: !transform.any_op):
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op, %arg1: !transform.any_op, %arg2: !transform.any_op):
  transform.sequence %arg0, %arg1, %arg2 : !transform.any_op, !transform.any_op, !transform.any_op failures(propagate) {
  ^bb0(%arg3: !transform.any_op, %arg4: !transform.any_op, %arg5: !transform.any_op):
    transform.test_print_remark_at_operand %arg4, "first extra" : !transform.any_op
    transform.test_print_remark_at_operand %arg5, "second extra" : !transform.any_op
  }
}

// expected-remark @below {{first extra}}
func.func @foo() {
  // expected-remark @below {{second extra}}
  return
}

// expected-remark @below {{first extra}}
func.func @bar(%arg0: i1) {
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  // expected-remark @below {{second extra}}
  return
^bb2:
  // expected-remark @below {{second extra}}
  return
}
