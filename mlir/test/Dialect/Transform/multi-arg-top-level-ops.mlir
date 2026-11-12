// RUN: mlir-opt %s --pass-pipeline="builtin.module(transform-interpreter{\
// RUN:       debug-bind-trailing-args=func.func,func.return})" \
// RUN:   --split-input-file --verify-diagnostics

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op, %arg1: !transform.any_op,
      %arg2: !transform.any_op) {
    transform.debug.emit_remark_at %arg1, "first extra" : !transform.any_op
    transform.debug.emit_remark_at %arg2, "second extra" : !transform.any_op
    transform.yield
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

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op, %arg1: !transform.any_op,
      %arg2: !transform.param<i64>) {
    // expected-error @above {{wrong kind of value provided for top-level parameter}}
    transform.yield
  }
}

func.func @foo() {
  return
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op, %arg1: !transform.any_op,
      %arg2: !transform.any_value) {
    // expected-error @above {{wrong kind of value provided for the top-level value handle}}
    transform.yield
  }
}

func.func @foo() {
  return
}

// -----


module attributes {transform.with_named_sequence} {
  // expected-error @below {{operation expects 1 extra value bindings, but 2 were provided to the interpreter}}
  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op, %arg1: !transform.any_op) {
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op, %arg1: !transform.any_op,
      %arg2: !transform.any_op) {
    transform.sequence %arg0, %arg1, %arg2 : !transform.any_op, !transform.any_op, !transform.any_op failures(propagate) {
    ^bb0(%arg3: !transform.any_op, %arg4: !transform.any_op, %arg5: !transform.any_op):
      transform.debug.emit_remark_at %arg4, "first extra" : !transform.any_op
      transform.debug.emit_remark_at %arg5, "second extra" : !transform.any_op
    }
    transform.yield
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
