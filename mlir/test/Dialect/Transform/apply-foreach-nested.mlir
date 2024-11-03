// RUN: mlir-opt %s --split-input-file --verify-diagnostics \
// RUN:          --pass-pipeline="builtin.module(test-transform-dialect-interpreter{enable-expensive-checks=1 bind-first-extra-to-ops=scf.for})"

func.func private @bar()

func.func @foo() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  // expected-note @below {{ancestor payload op}}
  scf.for %i = %c0 to %c1 step %c10 {
    // expected-note @below {{descendant payload op}}
    scf.for %j = %c0 to %c1 step %c10 {
      func.call @bar() : () -> ()
    }
  }
  return
}

transform.sequence failures(suppress) {
^bb0(%arg0: !transform.any_op, %arg1: !transform.op<"scf.for">):
  %1 = transform.test_reverse_payload_ops %arg1 : (!transform.op<"scf.for">) -> !transform.op<"scf.for">
  // expected-error @below {{transform operation consumes a handle pointing to an ancestor payload operation before its descendant}}
  // expected-note @below {{the ancestor is likely erased or rewritten before the descendant is accessed, leading to undefined behavior}}
  transform.test_consume_operand_each %1 : !transform.op<"scf.for">
}

// -----

func.func private @bar()

func.func @foo() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  scf.for %i = %c0 to %c1 step %c10 {
    scf.for %j = %c0 to %c1 step %c10 {
      func.call @bar() : () -> ()
    }
  }
  return
}

// No error here, processing ancestors before descendants.
transform.sequence failures(suppress) {
^bb0(%arg0: !transform.any_op, %arg1: !transform.op<"scf.for">):
  transform.test_consume_operand_each %arg1 : !transform.op<"scf.for">
}
