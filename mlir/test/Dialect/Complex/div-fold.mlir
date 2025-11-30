// RUN: mlir-opt %s -split-input-file -test-single-fold | FileCheck %s

// CHECK-LABEL: div_op_fold
func.func @div_op_fold() -> complex<f32> {
  %a = complex.constant [2.0 : f32, 1.0 : f32]: complex<f32>
  %b = complex.constant [1.0: f32, 0.0 : f32]: complex<f32>
  %div = complex.div %a, %b : complex<f32>
  // CHECK: %[[DIV:.*]] = complex.constant [2.000000e+00 : f32, 1.000000e+00 : f32] : complex<f32>
  // CHECK: return %[[DIV]] : complex<f32>
  return %div : complex<f32>
}

// CHECK-LABEL: div_op_not_fold_if_rhs_has_nan
func.func @div_op_not_fold_if_rhs_has_nan() -> complex<f32> {
  %a = complex.constant [0x7fffffff : f32, 1.0 : f32]: complex<f32>
  %b = complex.constant [1.0: f32, 0.0 : f32]: complex<f32>
  %div = complex.div %a, %b : complex<f32>
  // CHECK: %[[A:.*]] = complex.constant [0x7FFFFFFF : f32, 1.000000e+00 : f32] : complex<f32>
  // CHECK: %[[B:.*]] = complex.constant [1.000000e+00 : f32, 0.000000e+00 : f32] : complex<f32>
  // CHECK: %[[DIV:.*]] = complex.div %[[A]], %[[B]] : complex<f32>
  // CHECK: return %[[DIV]] : complex<f32>
  return %div : complex<f32>
}


