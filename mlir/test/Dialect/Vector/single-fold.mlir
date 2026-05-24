// RUN: mlir-opt %s -split-input-file -test-single-fold | FileCheck %s

// The tests in this file verify that fold() methods can handle complex
// optimization scenarios without requiring multiple folding iterations.
// This is important because:
//
// 1. OpBuilder::createOrFold() only calls fold() once, so operations must
//    be fully optimized in that single call
// 2. Multiple rounds of folding would incur higher performance costs,
//    so it's more efficient to complete all optimizations in one pass
//
// These tests ensure that folding implementations are robust and complete,
// avoiding situations where operations are left in intermediate states
// that could be further optimized.

// CHECK-LABEL: fold_extract_in_single_pass
// CHECK-SAME: (%{{.*}}: vector<4xf16>, %[[ARG1:.+]]: f16)
func.func @fold_extract_in_single_pass(%arg0: vector<4xf16>, %arg1: f16) -> f16 {
  %0 = vector.insert %arg1, %arg0 [1] : f16 into vector<4xf16>
  %c1 = arith.constant 1 : index
  // Verify that the fold is finished in a single pass even if the index is dynamic.
  %1 = vector.extract %0[%c1] : f16 from vector<4xf16>
  // CHECK: return %[[ARG1]] : f16
  return %1 : f16
}

// -----

// CHECK-LABEL: fold_insert_in_single_pass
func.func @fold_insert_in_single_pass() -> vector<2xf16> {
  %cst = arith.constant dense<0.000000e+00> : vector<2xf16>
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2.5 : f16
  // Verify that the fold is finished in a single pass even if the index is dynamic.
  // CHECK: arith.constant dense<[0.000000e+00, 2.500000e+00]> : vector<2xf16>
  %0 = vector.insert %c2, %cst [%c1] : f16 into vector<2xf16>
  return %0 : vector<2xf16>
} 