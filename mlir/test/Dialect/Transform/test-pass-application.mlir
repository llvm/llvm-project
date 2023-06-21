// RUN: mlir-opt %s --test-transform-dialect-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @successful_pass_application(
//       CHECK:   %[[c5:.*]] = arith.constant 5 : index
//       CHECK:   return %[[c5]]
func.func @successful_pass_application(%t: tensor<5xf32>) -> index {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %t, %c0 : tensor<5xf32>
  return %dim : index
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  transform.apply_registered_pass "canonicalize" to %1 : (!transform.any_op) -> !transform.any_op
}

// -----

func.func @invalid_pass_name() {
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  // expected-error @below {{unknown pass: non-existing-pass}}
  transform.apply_registered_pass "non-existing-pass" to %1 : (!transform.any_op) -> !transform.any_op
}

// -----

func.func @not_isolated_from_above(%t: tensor<5xf32>) -> index {
  %c0 = arith.constant 0 : index
  // expected-note @below {{target op}}
  // expected-error @below {{trying to schedule a pass on an operation not marked as 'IsolatedFromAbove'}}
  %dim = tensor.dim %t, %c0 : tensor<5xf32>
  return %dim : index
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %1 = transform.structured.match ops{["tensor.dim"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  // expected-error @below {{pass pipeline failed}}
  transform.apply_registered_pass "canonicalize" to %1 : (!transform.any_op) -> !transform.any_op
}

// -----

func.func @invalid_pass_option() {
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  // expected-error @below {{failed to add pass to pipeline: canonicalize}}
  transform.apply_registered_pass "canonicalize" to %1 {options = "invalid-option=1"} : (!transform.any_op) -> !transform.any_op
}

// -----

// CHECK-LABEL: func @valid_pass_option()
func.func @valid_pass_option() {
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  transform.apply_registered_pass "canonicalize" to %1 {options = "top-down=false"} : (!transform.any_op) -> !transform.any_op
}

// -----

module {
  // expected-error @below {{trying to schedule a pass on an unsupported operation}}
  // expected-note @below {{target op}}
  func.func @invalid_target_op_type() {
    return
  }

  transform.sequence failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op

    // func-bufferize can be applied only to ModuleOps.
    // expected-error @below {{pass pipeline failed}}
    transform.apply_registered_pass "func-bufferize" to %1 : (!transform.any_op) -> !transform.any_op
  }
}
