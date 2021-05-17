// RUN: mlir-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL: func @real_of_const(
func @real_of_const() -> f32 {
  // CHECK: %[[CST:.*]] = constant 1.000000e+00 : f32
  // CHECK-NEXT: return %[[CST]] : f32
  %complex = constant [1.0 : f32, 0.0 : f32] : complex<f32>
  %1 = complex.re %complex : complex<f32>
  return %1 : f32
}

// CHECK-LABEL: func @imag_of_const(
func @imag_of_const() -> f32 {
  // CHECK: %[[CST:.*]] = constant 0.000000e+00 : f32
  // CHECK-NEXT: return %[[CST]] : f32
  %complex = constant [1.0 : f32, 0.0 : f32] : complex<f32>
  %1 = complex.im %complex : complex<f32>
  return %1 : f32
}
