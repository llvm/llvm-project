// RUN: mlir-opt %s -convert-complex-to-rocdl -canonicalize | FileCheck %s

// CHECK-DAG: @__ocml_cabs_f32(complex<f32>) -> f32
// CHECK-DAG: @__ocml_cabs_f64(complex<f64>) -> f64

func.func @abs_caller(%f: complex<f32>, %d: complex<f64>) -> (f32, f64) {
  // CHECK: %[[RF:.*]] = call @__ocml_cabs_f32(%[[F:.*]])
  %rf = complex.abs %f : complex<f32>
  // CHECK: %[[RD:.*]] = call @__ocml_cabs_f64(%[[D:.*]])
  %rd = complex.abs %d : complex<f64>
  // CHECK: return %[[RF]], %[[RD]]
  return %rf, %rd : f32, f64
}
