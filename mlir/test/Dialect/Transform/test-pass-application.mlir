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
        with options = { "invalid-option" = 1 } to %1
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
        with options = { "top-down" = false } to %1
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
        with options = { "top-down" = false, "test-convergence" =true } to %1
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
        with options = { "top-down" = false, "max-iterations" = 0 } to %1
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
    %max_iter = transform.param.constant 10 -> !transform.any_param
    %max_rewrites = transform.param.constant 1 -> !transform.any_param
    %2 = transform.apply_registered_pass
        "canonicalize"
        with options = { "top-down" = false,
                         "max-iterations" = %max_iter,
                         "test-convergence" = true,
                         "max-num-rewrites" =  %max_rewrites }
        to %1
        : (!transform.any_op, !transform.any_param, !transform.any_param) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @invalid_options_as_str() {
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @+2 {{expected '{' in options dictionary}}
    %2 = transform.apply_registered_pass "canonicalize"
        with options = "top-down=false" to %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @invalid_options_as_pairs_without_braces() {
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @+2 {{expected '{' in options dictionary}}
    %2 = transform.apply_registered_pass "canonicalize"
        with options = "top-down"=false to %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @invalid_options_due_to_reserved_attr() {
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @+2 {{the param_operand attribute is a marker reserved for indicating a value will be passed via params and is only used in the generic print format}}
    %2 = transform.apply_registered_pass "canonicalize"
        with options = { "top-down" = #transform.param_operand<index=0> } to %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @invalid_options_due_duplicated_key() {
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @+2 {{duplicate keys found in options dictionary}}
    %2 = transform.apply_registered_pass "canonicalize"
        with options = {"top-down"=false,"top-down"=true} to %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @invalid_options_due_invalid_key() {
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @+2 {{expected key to either be an identifier or a string}}
    %2 = transform.apply_registered_pass "canonicalize"
        with options = { @label = 0 } to %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @invalid_pass_option_bare_param() {
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %pass_options = transform.param.constant 42 -> !transform.any_param
    // expected-error @+2 {{expected '{' in options dictionary}}
    transform.apply_registered_pass "canonicalize"
        with options = %pass_options to %1
        : (!transform.any_op, !transform.any_param) -> !transform.any_op
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
    %x = transform.param.constant true -> !transform.any_param
    %y = transform.param.constant false -> !transform.any_param
    %topdown_options = transform.merge_handles %x, %y : !transform.any_param
    // expected-error @below {{options passed as a param must have a single value associated, param 0 associates 2}}
    transform.apply_registered_pass "canonicalize"
        with options = { "top-down" = %topdown_options } to %1
        : (!transform.any_op, !transform.any_param) -> !transform.any_op
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

// -----

/////////////////////////////////////////////////////////////////////
// Check that the following cases are caugh in the generic format. //
/////////////////////////////////////////////////////////////////////

// Invalid due to param_operand occurences in options dict not being
// one-to-one with the dynamic options provided as params:
//   param_operand_index out of bounds w.r.t. the number of options provided via params.

"builtin.module"() ({
  "transform.named_sequence"() <{function_type = (!transform.any_op) -> (), sym_name = "__transform_main"}> ({
  ^bb0(%arg0: !transform.any_op):
    %0 = "transform.structured.match"(%arg0) <{ops = ["func.func"]}> : (!transform.any_op) -> !transform.any_op
    %1 = "transform.param.constant"() <{value = 10 : i64}> : () -> !transform.any_param
    // expected-error @below {{dynamic option index 1 is out of bounds for the number of dynamic options: 1}}
    %2 = "transform.apply_registered_pass"(%0, %1) <{
      options = {"max-iterations" = #transform.param_operand<index=1 : i64>,
                 "test-convergence" = true,
                 "top-down" = false},
      pass_name = "canonicalize"}>
    : (!transform.any_op, !transform.any_param) -> !transform.any_op
    "transform.yield"() : () -> ()
  }) : () -> ()
}) {transform.with_named_sequence} : () -> ()

// -----

// Invalid due to param_operand occurences in options dict not being
// one-to-one with the dynamic options provided as params:
//   the first option-param is referred to twice and the second one not at all.
// (In the pretty-printed format, if you want to refer to a param SSA-value twice, it counts as two param arguments.)

"builtin.module"() ({
  "transform.named_sequence"() <{function_type = (!transform.any_op) -> (), sym_name = "__transform_main"}> ({
  ^bb0(%arg0: !transform.any_op):
    %0 = "transform.structured.match"(%arg0) <{ops = ["func.func"]}> : (!transform.any_op) -> !transform.any_op
    %1 = "transform.param.constant"() <{value = 10 : i64}> : () -> !transform.any_param
    %2 = "transform.param.constant"() <{value = 1 : i64}> : () -> !transform.any_param
    // expected-error @below {{dynamic option index 0 is already used in options}}
    %3 = "transform.apply_registered_pass"(%0, %1, %2) <{
      options = {"max-iterations" = #transform.param_operand<index=0 : i64>,
                 "max-num-rewrites" = #transform.param_operand<index=0 : i64>,
                 "test-convergence" = true,
                 "top-down" = false},
      pass_name = "canonicalize"}>
    : (!transform.any_op, !transform.any_param, !transform.any_param) -> !transform.any_op
    "transform.yield"() : () -> ()
  }) : () -> ()
}) {transform.with_named_sequence} : () -> ()

// -----

// Invalid due to param_operand occurences in options dict not being
// one-to-one with the dynamic options provided as params:
//   two option-params are provide though only the first one is referred to from the options-dict.

"builtin.module"() ({
  "transform.named_sequence"() <{function_type = (!transform.any_op) -> (), sym_name = "__transform_main"}> ({
  ^bb0(%arg0: !transform.any_op):
    %0 = "transform.structured.match"(%arg0) <{ops = ["func.func"]}> : (!transform.any_op) -> !transform.any_op
    %1 = "transform.param.constant"() <{value = 10 : i64}> : () -> !transform.any_param
    %2 = "transform.param.constant"() <{value = 1 : i64}> : () -> !transform.any_param
    // expected-error @below {{a param operand does not have a corresponding param_operand attr in the options dict}}
    %3 = "transform.apply_registered_pass"(%0, %1, %2) <{
      options = {"max-iterations" = #transform.param_operand<index=0 : i64>,
                 "test-convergence" = true,
                 "top-down" = false},
      pass_name = "canonicalize"}>
    : (!transform.any_op, !transform.any_param, !transform.any_param) -> !transform.any_op
    "transform.yield"() : () -> ()
  }) : () -> ()
}) {transform.with_named_sequence} : () -> ()
