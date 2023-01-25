// RUN: mlir-opt %s -split-input-file -test-constant-fold | FileCheck %s

// CHECK-LABEL: fold_extract_transpose_negative
func.func @fold_extract_transpose_negative(%arg0: vector<4x4xf16>) -> vector<4x4xf16> {
  %cst = arith.constant dense<0.000000e+00> : vector<1x4x4xf16>
  %0 = vector.insert %arg0, %cst [0] : vector<4x4xf16> into vector<1x4x4xf16>
  // Verify that the transpose didn't get dropped.
  // CHECK: %[[T:.+]] = vector.transpose
  %1 = vector.transpose %0, [0, 2, 1] : vector<1x4x4xf16> to vector<1x4x4xf16>
  // CHECK: vector.extract %[[T]][0]
  %2 = vector.extract %1[0] : vector<1x4x4xf16>
  return %2 : vector<4x4xf16>
}
