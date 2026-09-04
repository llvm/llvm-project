// RUN: mlir-opt %s | FileCheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.apply_conversion_patterns to %arg0 {
    } with type_converter {
      transform.apply_conversion_patterns.memref.memref_to_llvm_type_converter
        use_bare_ptr_call_conv = false data_layout = "e-p:64:64"
        index_bitwidth = 64 use_generic_functions = false
        use_aligned_alloc = false
    } : !transform.any_op
    transform.yield
  }
}

// CHECK: transform.apply_conversion_patterns.memref.memref_to_llvm_type_converter
// CHECK-SAME: use_aligned_alloc = false index_bitwidth = 64
// CHECK-SAME: use_generic_functions = false use_bare_ptr_call_conv = false
// CHECK-SAME: data_layout = "e-p:64:64"
