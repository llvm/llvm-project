// RUN: mlir-opt %s -convert-vector-to-arm-sme -split-input-file -allow-unregistered-dialect | FileCheck %s

// =============================================================================
// arith.constant dense<0> to arm_sme.zero
// =============================================================================

// -----

// CHECK-LABEL: @arith_constant_dense_2d_zero_i8
// CHECK: %[[ZERO:.*]] = arm_sme.zero : vector<[16]x[16]xi8>
func.func @arith_constant_dense_2d_zero_i8() {
  %zero = arith.constant dense<0> : vector<[16]x[16]xi8>
  "prevent.dce"(%zero) : (vector<[16]x[16]xi8>) -> ()
  return
}

// -----

// CHECK-LABEL: @arith_constant_dense_2d_zero_i16
// CHECK: %[[ZERO:.*]] = arm_sme.zero : vector<[8]x[8]xi16>
func.func @arith_constant_dense_2d_zero_i16() {
  %zero = arith.constant dense<0> : vector<[8]x[8]xi16>
  "prevent.dce"(%zero) : (vector<[8]x[8]xi16>) -> ()
  return
}

// -----

// CHECK-LABEL: @arith_constant_dense_2d_zero_i32
// CHECK: %[[ZERO:.*]] = arm_sme.zero : vector<[4]x[4]xi32>
func.func @arith_constant_dense_2d_zero_i32() {
  %zero = arith.constant dense<0> : vector<[4]x[4]xi32>
  "prevent.dce"(%zero) : (vector<[4]x[4]xi32>) -> ()
  return
}

// -----

// CHECK-LABEL: @arith_constant_dense_2d_zero_i64
// CHECK: %[[ZERO:.*]] = arm_sme.zero : vector<[2]x[2]xi64>
func.func @arith_constant_dense_2d_zero_i64() {
  %zero = arith.constant dense<0> : vector<[2]x[2]xi64>
  "prevent.dce"(%zero) : (vector<[2]x[2]xi64>) -> ()
  return
}

// -----

// CHECK-LABEL: @arith_constant_dense_2d_zero_f16
// CHECK: %[[ZERO:.*]] = arm_sme.zero : vector<[8]x[8]xf16>
func.func @arith_constant_dense_2d_zero_f16() {
  %zero = arith.constant dense<0.0> : vector<[8]x[8]xf16>
  "prevent.dce"(%zero) : (vector<[8]x[8]xf16>) -> ()
  return
}

// -----

// CHECK-LABEL: @arith_constant_dense_2d_zero_bf16
// CHECK: %[[ZERO:.*]] = arm_sme.zero : vector<[8]x[8]xbf16>
func.func @arith_constant_dense_2d_zero_bf16() {
  %zero = arith.constant dense<0.0> : vector<[8]x[8]xbf16>
  "prevent.dce"(%zero) : (vector<[8]x[8]xbf16>) -> ()
  return
}

// -----

// CHECK-LABEL: @arith_constant_dense_2d_zero_f32
// CHECK: %[[ZERO:.*]] = arm_sme.zero : vector<[4]x[4]xf32>
func.func @arith_constant_dense_2d_zero_f32() {
  %zero = arith.constant dense<0.0> : vector<[4]x[4]xf32>
  "prevent.dce"(%zero) : (vector<[4]x[4]xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: @arith_constant_dense_2d_zero_f64
// CHECK: %[[ZERO:.*]] = arm_sme.zero : vector<[2]x[2]xf64>
func.func @arith_constant_dense_2d_zero_f64() {
  %zero = arith.constant dense<0.0> : vector<[2]x[2]xf64>
  "prevent.dce"(%zero) : (vector<[2]x[2]xf64>) -> ()
  return
}
