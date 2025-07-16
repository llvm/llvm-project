// RUN: mlir-opt %s -convert-complex-to-rocdl-library-calls | FileCheck %s

// CHECK-DAG: @__ocml_cabs_f32(complex<f32>) -> f32
// CHECK-DAG: @__ocml_cabs_f64(complex<f64>) -> f64
// CHECK-DAG: @__ocml_cexp_f32(complex<f32>) -> complex<f32>
// CHECK-DAG: @__ocml_cexp_f64(complex<f64>) -> complex<f64>

//CHECK-LABEL: @abs_caller
func.func @abs_caller(%f: complex<f32>, %d: complex<f64>) -> (f32, f64) {
  // CHECK: %[[RF:.*]] = call @__ocml_cabs_f32(%{{.*}})
  %rf = complex.abs %f : complex<f32>
  // CHECK: %[[RD:.*]] = call @__ocml_cabs_f64(%{{.*}})
  %rd = complex.abs %d : complex<f64>
  // CHECK: return %[[RF]], %[[RD]]
  return %rf, %rd : f32, f64
}

//CHECK-LABEL: @exp_caller
func.func @exp_caller(%f: complex<f32>, %d: complex<f64>) -> (complex<f32>, complex<f64>) {
  // CHECK: %[[EF:.*]] = call @__ocml_cexp_f32(%{{.*}})
  %ef = complex.exp %f : complex<f32>
  // CHECK: %[[ED:.*]] = call @__ocml_cexp_f64(%{{.*}})
  %ed = complex.exp %d : complex<f64>
  // CHECK: return %[[EF]], %[[ED]]
  return %ef, %ed : complex<f32>, complex<f64>
}
