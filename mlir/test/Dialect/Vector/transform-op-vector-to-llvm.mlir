// RUN: mlir-opt %s -transform-interpreter -verify-diagnostics -allow-unregistered-dialect -split-input-file | FileCheck %s

// CHECK-LABEL: func @lower_to_llvm
//   CHECK-NOT:   vector.bitcast
//       CHECK:   llvm.bitcast
func.func @lower_to_llvm(%input: vector<f32>) -> vector<i32> {
  %0 = vector.bitcast %input : vector<f32> to vector<i32>
  return %0 : vector<i32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_conversion_patterns to %0 {
      transform.apply_conversion_patterns.vector.vector_to_llvm
        force_32bit_vector_indices = false
        reassociate_fp_reductions = true use_vector_alignment = true
    } with type_converter {
      transform.apply_conversion_patterns.memref.memref_to_llvm_type_converter
    } <legal_dialects = ["func", "llvm"]> : !transform.any_op
    transform.yield
  }
}
