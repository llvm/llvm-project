// RUN: mlir-opt %s --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @successful_pass_application(
//       CHECK:   %[[c5:.*]] = arith.constant 5 : index
//       CHECK:   return %[[c5]]
func.func @successful_pass_application(%t: tensor<5xf32>) -> index {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %t, %c0 : tensor<5xf32>
  return %dim : index
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_registered_pass "canonicalize" to %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @pass_pipeline(
func.func @pass_pipeline() {
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // This pipeline does not do anything. Just make sure that the pipeline is
    // found and no error is produced.
    transform.apply_registered_pass "test-options-pass-pipeline" to %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @invalid_pass_name() {
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{unknown pass or pass pipeline: non-existing-pass}}
    transform.apply_registered_pass "non-existing-pass" to %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @not_isolated_from_above(%t: tensor<5xf32>) -> index {
  %c0 = arith.constant 0 : index
  // expected-note @below {{target op}}
  // expected-error @below {{trying to schedule a pass on an operation not marked as 'IsolatedFromAbove'}}
  %dim = tensor.dim %t, %c0 : tensor<5xf32>
  return %dim : index
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %1 = transform.structured.match ops{["tensor.dim"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{pass pipeline failed}}
    transform.apply_registered_pass "canonicalize" to %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @invalid_pass_option() {
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{failed to add pass or pass pipeline to pipeline: canonicalize}}
    // expected-error @below {{<Pass-Options-Parser>: no such option invalid-option}}
    transform.apply_registered_pass "canonicalize"
        with options = "invalid-option=1" to %1
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @valid_pass_option()
func.func @valid_pass_option() {
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_registered_pass "canonicalize"
        with options = "top-down=false" to %1
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @valid_pass_options()
func.func @valid_pass_options() {
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    //transform.apply_registered_pass "canonicalize" with options = "top-down=false,max-iterations=10" to %1 : (!transform.any_op) -> !transform.any_op
    transform.apply_registered_pass "canonicalize"
        with options = "top-down=false test-convergence=true" to %1
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @valid_pass_options_as_list()
func.func @valid_pass_options_as_list() {
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_registered_pass "canonicalize"
        with options = "top-down=false" "max-iterations=0" to %1
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @valid_dynamic_pass_options()
func.func @valid_dynamic_pass_options() {
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %max_iter = transform.param.constant "max-iterations=10" -> !transform.any_param
    %max_rewrites = transform.param.constant "max-num-rewrites=1" -> !transform.any_param
    %2 = transform.apply_registered_pass "canonicalize"
        with options = "top-down=false" %max_iter "test-convergence=true" %max_rewrites to %1
        : (!transform.any_param, !transform.any_param, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @invalid_dynamic_options_as_array() {
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %max_iter = transform.param.constant "max-iterations=10" -> !transform.any_param
    // expected-error @+2 {{expected at least one option (either a string or a param)}}
    %2 = transform.apply_registered_pass "canonicalize"
        with options = ["top-down=false" %max_iter] to %1
        : (!transform.any_param, !transform.any_param, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @invalid_options_as_pairs() {
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @+2 {{expected 'to'}}
    %2 = transform.apply_registered_pass "canonicalize"
        with options = "top-down=" false to %1
        : (!transform.any_param, !transform.any_param, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @invalid_pass_option_param() {
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %pass_options = transform.param.constant 42 -> !transform.any_param
    // expected-error @below {{options passed as a param must be a string, got 42}}
    transform.apply_registered_pass "canonicalize"
        with options = %pass_options to %1
        : (!transform.any_param, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @too_many_pass_option_params() {
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %x = transform.param.constant "x" -> !transform.any_param
    %y = transform.param.constant "y" -> !transform.any_param
    %pass_options = transform.merge_handles %x, %y : !transform.any_param
    // expected-error @below {{options passed as a param must have a single value associated, param 0 associates 2}}
    transform.apply_registered_pass "canonicalize"
        with options = %pass_options to %1
        : (!transform.any_param, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  // expected-error @below {{trying to schedule a pass on an unsupported operation}}
  // expected-note @below {{target op}}
  func.func @invalid_target_op_type() {
    return
  }

  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op

    // duplicate-function-elimination can be applied only to ModuleOps.
    // expected-error @below {{pass pipeline failed}}
    transform.apply_registered_pass "duplicate-function-elimination" to %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
