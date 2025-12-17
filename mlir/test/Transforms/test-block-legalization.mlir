// RUN: mlir-opt %s -transform-interpreter | FileCheck %s

// CHECK-LABEL: func @complex_block_signature_conversion(
//       CHECK:   %[[cst:.*]] = complex.constant
//       CHECK:   %[[complex_llvm:.*]] = builtin.unrealized_conversion_cast %[[cst]] : complex<f64> to !llvm.struct<(f64, f64)>
// Note: Some blocks are omitted.
//       CHECK:   llvm.br ^[[block1:.*]](%[[complex_llvm]]
//       CHECK: ^[[block1]](%[[arg:.*]]: !llvm.struct<(f64, f64)>):
//       CHECK:   %[[cast:.*]] = builtin.unrealized_conversion_cast %[[arg]] : !llvm.struct<(f64, f64)> to complex<f64>
//       CHECK:   llvm.br ^[[block2:.*]]
//       CHECK: ^[[block2]]:
//       CHECK:   "test.consumer_of_complex"(%[[cast]]) : (complex<f64>) -> ()
func.func @complex_block_signature_conversion() {
  %cst = complex.constant [0.000000e+00, 0.000000e+00] : complex<f64>
  %true = arith.constant true
  %0 = scf.if %true -> complex<f64> {
    scf.yield %cst : complex<f64>
  } else {
    scf.yield %cst : complex<f64>
  }

  // Regression test to ensure that the a source materialization is inserted.
  // The operand of "test.consumer_of_complex" must not change.
  "test.consumer_of_complex"(%0) : (complex<f64>) -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%toplevel_module: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %toplevel_module
      : (!transform.any_op) -> !transform.any_op
    transform.apply_conversion_patterns to %func {
      transform.apply_conversion_patterns.dialect_to_llvm "cf"
      transform.apply_conversion_patterns.func.func_to_llvm
      transform.apply_conversion_patterns.scf.scf_to_control_flow
    } with type_converter {
      transform.apply_conversion_patterns.memref.memref_to_llvm_type_converter
    } {
      legal_dialects = ["llvm"], 
      partial_conversion
    } : !transform.any_op
    transform.yield
  }
}
