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
    } with type_converter {
      transform.apply_conversion_patterns.memref.memref_to_llvm_type_converter
    } {legal_dialects = ["func", "llvm"]} : !transform.any_op
    transform.yield
  }
}

// -----

// RUN: mlir-opt %s -transform-interpreter -verify-diagnostics
// Regression test for bug #204100.
// Assertion idx < size() in SmallVector.h used to happen here.
// CHECK-LABEL: func @add_dynamic
module {
  func.func @add_dynamic(%arg0: memref<?x?xbf16>, %arg1: memref<?x?xbf16>, %arg2: memref<?x?xbf16>) {
    linalg.add ins(%arg0, %arg1 : memref<?x?xbf16>, memref<?x?xbf16>) outs(%arg2 : memref<?x?xbf16>)
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.add"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      // This combination was crashing (static/dynamic mismatch)
      transform.structured.vectorize %0 vector_sizes [8, [16], 4] : !transform.any_op
      // expected-error @above {{Attempted to vectorize, but failed}}
      transform.yield
    }
  }
}




