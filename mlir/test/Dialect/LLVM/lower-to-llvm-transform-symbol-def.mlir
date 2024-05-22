// RUN: mlir-opt %s

/// Schedule to lower to LLVM.
module @lower_module_to_cpu attributes { transform.with_named_sequence } {

transform.named_sequence @lower_to_cpu(
    %module: !transform.any_op {transform.consumed}) -> !transform.any_op {

  %func = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.any_op
  %f = transform.apply_registered_pass "convert-vector-to-scf" to %func : (!transform.any_op) -> !transform.any_op
  %f2 = transform.apply_registered_pass "convert-linalg-to-loops" to %f : (!transform.any_op) -> !transform.any_op
  %f3 = transform.apply_registered_pass "convert-scf-to-cf" to %f2 : (!transform.any_op) -> !transform.any_op
  %f4 = transform.apply_registered_pass "expand-strided-metadata" to %f3 : (!transform.any_op) -> !transform.any_op
  %f5 = transform.apply_registered_pass "lower-affine" to %f4 : (!transform.any_op) -> !transform.any_op

  transform.apply_conversion_patterns to %f5 {
    transform.apply_conversion_patterns.dialect_to_llvm "math"
    transform.apply_conversion_patterns.vector.vector_to_llvm
    transform.apply_conversion_patterns.dialect_to_llvm "memref"
    transform.apply_conversion_patterns.func.func_to_llvm
    transform.apply_conversion_patterns.dialect_to_llvm "index"
    transform.apply_conversion_patterns.dialect_to_llvm "arith"
    transform.apply_conversion_patterns.dialect_to_llvm "cf"
  } with type_converter {
    transform.apply_conversion_patterns.memref.memref_to_llvm_type_converter
      {index_bitwidth = 64,
       use_bare_ptr = false,
       use_bare_ptr_memref_call_conv = false,
       use_opaque_pointers = true}
  } {
    legal_dialects = ["llvm"],
    partial_conversion
  } : !transform.any_op

  %m2 = transform.apply_registered_pass "reconcile-unrealized-casts" to %module : (!transform.any_op) -> !transform.any_op
  transform.yield %m2 : !transform.any_op
}

} // transform module
