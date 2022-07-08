// RUN: mlir-opt %s -convert-complex-to-libm -canonicalize | FileCheck %s

// CHECK-DAG: @cpowf(complex<f32>, complex<f32>) -> complex<f32>
// CHECK-DAG: @cpow(complex<f64>, complex<f64>) -> complex<f64>
// CHECK-DAG: @csqrtf(complex<f32>) -> complex<f32>
// CHECK-DAG: @csqrt(complex<f64>) -> complex<f64>
// CHECK-DAG: @ctanhf(complex<f32>) -> complex<f32>
// CHECK-DAG: @ctanh(complex<f64>) -> complex<f64>
// CHECK-DAG: @ccos(complex<f64>) -> complex<f64>
// CHECK-DAG: @csin(complex<f64>) -> complex<f64>
// CHECK-DAG: @conj(complex<f64>) -> complex<f64>
// CHECK-DAG: @cabs(complex<f64>) -> f64

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

// CHECK-LABEL: func @ccos_caller
// CHECK-SAME: %[[FLOAT:.*]]: complex<f32>
// CHECK-SAME: %[[DOUBLE:.*]]: complex<f64>
func.func @ccos_caller(%float: complex<f32>, %double: complex<f64>) -> (complex<f32>, complex<f64>)  {
  // CHECK: %[[FLOAT_RESULT:.*]] = call @ccosf(%[[FLOAT]])
  %float_result = complex.cos %float : complex<f32>
  // CHECK: %[[DOUBLE_RESULT:.*]] = call @ccos(%[[DOUBLE]])
  %double_result = complex.cos %double : complex<f64>
  // CHECK: return %[[FLOAT_RESULT]], %[[DOUBLE_RESULT]]
  return %float_result, %double_result : complex<f32>, complex<f64>
}

// CHECK-LABEL: func @csin_caller
// CHECK-SAME: %[[FLOAT:.*]]: complex<f32>
// CHECK-SAME: %[[DOUBLE:.*]]: complex<f64>
func.func @csin_caller(%float: complex<f32>, %double: complex<f64>) -> (complex<f32>, complex<f64>)  {
  // CHECK: %[[FLOAT_RESULT:.*]] = call @csinf(%[[FLOAT]])
  %float_result = complex.sin %float : complex<f32>
  // CHECK: %[[DOUBLE_RESULT:.*]] = call @csin(%[[DOUBLE]])
  %double_result = complex.sin %double : complex<f64>
  // CHECK: return %[[FLOAT_RESULT]], %[[DOUBLE_RESULT]]
  return %float_result, %double_result : complex<f32>, complex<f64>
}

// CHECK-LABEL: func @conj_caller
// CHECK-SAME: %[[FLOAT:.*]]: complex<f32>
// CHECK-SAME: %[[DOUBLE:.*]]: complex<f64>
func.func @conj_caller(%float: complex<f32>, %double: complex<f64>) -> (complex<f32>, complex<f64>)  {
  // CHECK: %[[FLOAT_RESULT:.*]] = call @conjf(%[[FLOAT]])
  %float_result = complex.conj %float : complex<f32>
  // CHECK: %[[DOUBLE_RESULT:.*]] = call @conj(%[[DOUBLE]])
  %double_result = complex.conj %double : complex<f64>
  // CHECK: return %[[FLOAT_RESULT]], %[[DOUBLE_RESULT]]
  return %float_result, %double_result : complex<f32>, complex<f64>
}

// CHECK-LABEL: func @cabs_caller
// CHECK-SAME: %[[FLOAT:.*]]: complex<f32>
// CHECK-SAME: %[[DOUBLE:.*]]: complex<f64>
func.func @cabs_caller(%float: complex<f32>, %double: complex<f64>) -> (f32, f64)  {
  // CHECK: %[[FLOAT_RESULT:.*]] = call @cabsf(%[[FLOAT]])
  %float_result = complex.abs %float : complex<f32>
  // CHECK: %[[DOUBLE_RESULT:.*]] = call @cabs(%[[DOUBLE]])
  %double_result = complex.abs %double : complex<f64>
  // CHECK: return %[[FLOAT_RESULT]], %[[DOUBLE_RESULT]]
  return %float_result, %double_result : f32, f64
}