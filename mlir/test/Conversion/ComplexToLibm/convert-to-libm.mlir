// RUN: mlir-opt %s -convert-complex-to-libm -canonicalize | FileCheck %s

// CHECK-DAG: @cpowf(complex<f32>, complex<f32>) -> complex<f32>
// CHECK-DAG: @cpow(complex<f64>, complex<f64>) -> complex<f64>
// CHECK-DAG: @csqrtf(complex<f32>) -> complex<f32>
// CHECK-DAG: @csqrt(complex<f64>) -> complex<f64>
// CHECK-DAG: @ctanhf(complex<f32>) -> complex<f32>
// CHECK-DAG: @ctanh(complex<f64>) -> complex<f64>

// CHECK-LABEL: func @cpow_caller
// CHECK-SAME: %[[FLOAT:.*]]: complex<f32>
// CHECK-SAME: %[[DOUBLE:.*]]: complex<f64>
func.func @cpow_caller(%float: complex<f32>, %double: complex<f64>) -> (complex<f32>, complex<f64>)  {
  // CHECK-DAG: %[[FLOAT_RESULT:.*]] = call @cpowf(%[[FLOAT]], %[[FLOAT]]) : (complex<f32>, complex<f32>) -> complex<f32>
  %float_result = complex.pow %float, %float : complex<f32>
  // CHECK-DAG: %[[DOUBLE_RESULT:.*]] = call @cpow(%[[DOUBLE]], %[[DOUBLE]]) : (complex<f64>, complex<f64>) -> complex<f64>
  %double_result = complex.pow %double, %double : complex<f64>
  // CHECK: return %[[FLOAT_RESULT]], %[[DOUBLE_RESULT]]
  return %float_result, %double_result : complex<f32>, complex<f64>
}

// CHECK-LABEL: func @csqrt_caller
// CHECK-SAME: %[[FLOAT:.*]]: complex<f32>
// CHECK-SAME: %[[DOUBLE:.*]]: complex<f64>
func.func @csqrt_caller(%float: complex<f32>, %double: complex<f64>) -> (complex<f32>, complex<f64>)  {
  // CHECK-DAG: %[[FLOAT_RESULT:.*]] = call @csqrtf(%[[FLOAT]]) : (complex<f32>) -> complex<f32>
  %float_result = complex.sqrt %float : complex<f32>
  // CHECK-DAG: %[[DOUBLE_RESULT:.*]] = call @csqrt(%[[DOUBLE]]) : (complex<f64>) -> complex<f64>
  %double_result = complex.sqrt %double : complex<f64>
  // CHECK: return %[[FLOAT_RESULT]], %[[DOUBLE_RESULT]]
  return %float_result, %double_result : complex<f32>, complex<f64>
}

// CHECK-LABEL: func @ctanh_caller
// CHECK-SAME: %[[FLOAT:.*]]: complex<f32>
// CHECK-SAME: %[[DOUBLE:.*]]: complex<f64>
func.func @ctanh_caller(%float: complex<f32>, %double: complex<f64>) -> (complex<f32>, complex<f64>)  {
  // CHECK-DAG: %[[FLOAT_RESULT:.*]] = call @ctanhf(%[[FLOAT]]) : (complex<f32>) -> complex<f32>
  %float_result = complex.tanh %float : complex<f32>
  // CHECK-DAG: %[[DOUBLE_RESULT:.*]] = call @ctanh(%[[DOUBLE]]) : (complex<f64>) -> complex<f64>
  %double_result = complex.tanh %double : complex<f64>
  // CHECK: return %[[FLOAT_RESULT]], %[[DOUBLE_RESULT]]
  return %float_result, %double_result : complex<f32>, complex<f64>
}
