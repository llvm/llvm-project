// RUN: mlir-opt %s --split-input-file --verify-diagnostics --transform-interpreter

// expected-error @below {{normal form test_single_block_normal_form requires payload operations to have a single region}}
func.func private @empty()

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.structured.match attributes {sym_name = "empty"} in %arg0 : (!transform.any_op) -> !transform.normalized_op<#transform.test_single_block_normal_form<nested false>>
    transform.yield
  }
}

// -----

// expected-remark @below {{found}}
func.func private @single() {
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match attributes {sym_name = "single"} in %arg0 : (!transform.any_op) -> !transform.normalized_op<#transform.test_single_block_normal_form<nested false>>
    transform.debug.emit_remark_at %0, "found" : !transform.normalized_op<#transform.test_single_block_normal_form<nested false>>
    transform.yield
  }
}

// -----

// expected-error @below {{normal form test_single_block_normal_form requires payload operations to have a single region}}
func.func private @branchy() {
  cf.br ^bb1
^bb1:
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.structured.match attributes {sym_name = "branchy"} in %arg0 : (!transform.any_op) -> !transform.normalized_op<#transform.test_single_block_normal_form<nested false>>
    transform.yield
  }
}

// -----

// expected-remark @below {{found}}
func.func private @nested() {
  scf.execute_region {
    cf.br ^bb1
  ^bb1:
    scf.yield
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match attributes {sym_name = "nested"} in %arg0 : (!transform.any_op) -> !transform.normalized_op<#transform.test_single_block_normal_form<nested false>>
    transform.debug.emit_remark_at %0, "found" : !transform.normalized_op<#transform.test_single_block_normal_form<nested false>>
    transform.yield
  }
}

// -----

func.func private @nested() {
  // expected-error @below {{normal form test_single_block_normal_form requires payload operations to have a single region}}
  scf.execute_region {
    cf.br ^bb1
  ^bb1:
    scf.yield
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.structured.match attributes {sym_name = "nested"} in %arg0 : (!transform.any_op) -> !transform.normalized_op<#transform.test_single_block_normal_form<nested true>>
    transform.yield
  }
}

