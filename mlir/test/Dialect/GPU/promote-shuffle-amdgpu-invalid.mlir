// RUN: mlir-opt %s --transform-interpreter --split-input-file --verify-diagnostics

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %module_op : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      // expected-error@below {{'gfx999' is not an AMDGCN triple or GPU name}}
      transform.apply_patterns.gpu.gpu_shuffle_to_amdgpu <triple = "gfx999">
    } : !transform.any_op
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %module_op : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      // expected-error@below {{'chip' and 'features' require a 'triple'}}
      transform.apply_patterns.gpu.gpu_shuffle_to_amdgpu <chip = "gfx950">
    } : !transform.any_op
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %module_op : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      // expected-error@below {{invalid target feature '+not-a-feature'}}
      transform.apply_patterns.gpu.gpu_shuffle_to_amdgpu <triple = "gfx950", features = "+not-a-feature">
    } : !transform.any_op
    transform.yield
  }
}
