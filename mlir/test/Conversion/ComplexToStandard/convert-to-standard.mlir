// RUN: mlir-opt %s -convert-complex-to-standard | FileCheck %s

// CHECK-LABEL: func @complex_abs
// CHECK-SAME: %[[ARG:.*]]: complex<f32>
func @complex_abs(%arg: complex<f32>) -> f32 {
  %abs = complex.abs %arg: complex<f32>
  return %abs : f32
}
// CHECK: %[[REAL:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK: %[[IMAG:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK-DAG: %[[REAL_SQ:.*]] = mulf %[[REAL]], %[[REAL]] : f32
// CHECK-DAG: %[[IMAG_SQ:.*]] = mulf %[[IMAG]], %[[IMAG]] : f32
// CHECK: %[[SQ_NORM:.*]] = addf %[[REAL_SQ]], %[[IMAG_SQ]] : f32
// CHECK: %[[NORM:.*]] = math.sqrt %[[SQ_NORM]] : f32
// CHECK: return %[[NORM]] : f32

