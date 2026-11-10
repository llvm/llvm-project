// RUN: mlir-opt %s -verify-diagnostics

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.apply_conversion_patterns to %arg0 {
    } with type_converter {
      // expected-error @+2 {{duplicate 'index_bitwidth' option}}
      transform.apply_conversion_patterns.memref.memref_to_llvm_type_converter
        index_bitwidth = 32 index_bitwidth = 64
    } : !transform.any_op
    transform.yield
  }
}
