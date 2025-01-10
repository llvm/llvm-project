// RUN: mlir-opt %s -split-input-file -test-constant-fold | FileCheck %s

// CHECK-LABEL: fold_extract_transpose_negative
func.func @fold_extract_transpose_negative(%arg0: vector<4x4xf16>) -> vector<4x4xf16> {
  %cst = arith.constant dense<0.000000e+00> : vector<1x4x4xf16>
  %0 = vector.insert %arg0, %cst [0] : vector<4x4xf16> into vector<1x4x4xf16>
  // Verify that the transpose didn't get dropped.
  // CHECK: %[[T:.+]] = vector.transpose
  %1 = vector.transpose %0, [0, 2, 1] : vector<1x4x4xf16> to vector<1x4x4xf16>
  // CHECK: vector.extract %[[T]][0]
  %2 = vector.extract %1[0] : vector<4x4xf16> from vector<1x4x4xf16>
  return %2 : vector<4x4xf16>
}

// CHECK-LABEL: fold_multid_reduction_f32_add
func.func @fold_multid_reduction_f32_add() -> vector<1xf32> {
  %cst_0 = arith.constant dense<0.000000e+00> : vector<1xf32>
  %0 = arith.constant dense<1.000000e+00> : vector<1x128x128xf32>
  // CHECK: %{{.*}} = arith.constant dense<1.638400e+04> : vector<1xf32>
  %1 = vector.multi_reduction <add>, %0, %cst_0 [1, 2] : vector<1x128x128xf32> to vector<1xf32>
  return %1 : vector<1xf32>
}

// CHECK-LABEL: fold_multid_reduction_f32_mul
func.func @fold_multid_reduction_f32_mul() -> vector<1xf32> {
  %cst_0 = arith.constant dense<1.000000e+00> : vector<1xf32>
  %0 = arith.constant dense<2.000000e+00> : vector<1x2x2xf32>
  // CHECK: %{{.*}} = arith.constant dense<1.600000e+01> : vector<1xf32>
  %1 = vector.multi_reduction <mul>, %0, %cst_0 [1, 2] : vector<1x2x2xf32> to vector<1xf32>
  return %1 : vector<1xf32>
}

// CHECK-LABEL: fold_multid_reduction_i32_add
func.func @fold_multid_reduction_i32_add() -> vector<1xi32> {
  %cst_1 = arith.constant dense<1> : vector<1xi32>
  %0 = arith.constant dense<1> : vector<1x128x128xi32>
  // CHECK: %{{.*}} = arith.constant dense<16385> : vector<1xi32>
  %1 = vector.multi_reduction <add>, %0, %cst_1 [1, 2] : vector<1x128x128xi32> to vector<1xi32>
  return %1 : vector<1xi32>
}

// CHECK-LABEL: fold_multid_reduction_i32_xor_odd
func.func @fold_multid_reduction_i32_xor_odd() -> vector<1xi32> {
  %cst_1 = arith.constant dense<0xFF> : vector<1xi32>
  %0 = arith.constant dense<0xA0A> : vector<1x3xi32>
  // CHECK: %{{.*}} = arith.constant dense<2805> : vector<1xi32>
  %1 = vector.multi_reduction <xor>, %0, %cst_1 [1] : vector<1x3xi32> to vector<1xi32>
  return %1 : vector<1xi32>
}

// CHECK-LABEL: fold_multid_reduction_i32_xor_even
func.func @fold_multid_reduction_i32_xor_even() -> vector<1xi32> {
  %cst_1 = arith.constant dense<0xFF> : vector<1xi32>
  %0 = arith.constant dense<0xA0A> : vector<1x4xi32>
  // CHECK: %{{.*}} = arith.constant dense<255> : vector<1xi32>
  %1 = vector.multi_reduction <xor>, %0, %cst_1 [1] : vector<1x4xi32> to vector<1xi32>
  return %1 : vector<1xi32>
}

// CHECK-LABEL: fold_multid_reduction_i64_add
func.func @fold_multid_reduction_i64_add() -> vector<1xi64> {
  %cst_1 = arith.constant dense<1> : vector<1xi64>
  %0 = arith.constant dense<1> : vector<1x128x128xi64>
  // CHECK: %{{.*}} = arith.constant dense<16385> : vector<1xi64>
  %1 = vector.multi_reduction <add>, %0, %cst_1 [1, 2] : vector<1x128x128xi64> to vector<1xi64>
  return %1 : vector<1xi64>
}