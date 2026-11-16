// RUN: mlir-opt %s --transform-interpreter --split-input-file --verify-diagnostics

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %module_op : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      // expected-error@below {{'gfx999' is not a valid AMDGPU architecture}}
      transform.apply_patterns.gpu.gpu_shuffle_to_amdgpu <arch = "gfx999">
    } : !transform.any_op
    transform.yield
  }
}

// -----

// Only xnack and sramecc are target-ID features, and only on a GPU that
// supports switching them.

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %module_op : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      // expected-error@below {{'gfx600:xnack+' is not a valid AMDGPU architecture}}
      transform.apply_patterns.gpu.gpu_shuffle_to_amdgpu <arch = "gfx600:xnack+">
    } : !transform.any_op
    transform.yield
  }
}

// -----

// A modifier without a +/- sign is not a target ID.

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %module_op : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      // expected-error@below {{'gfx908:xnack' is not a valid AMDGPU architecture}}
      transform.apply_patterns.gpu.gpu_shuffle_to_amdgpu <arch = "gfx908:xnack">
    } : !transform.any_op
    transform.yield
  }
}

// -----

// Wavefront size is not a target-ID feature, so it cannot ride on `arch`.

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %module_op : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      // expected-error@below {{'gfx1030:wavefrontsize64+' is not a valid AMDGPU architecture}}
      transform.apply_patterns.gpu.gpu_shuffle_to_amdgpu <arch = "gfx1030:wavefrontsize64+">
    } : !transform.any_op
    transform.yield
  }
}
