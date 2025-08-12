// RUN: mlir-opt %s -convert-complex-to-rocdl-library-calls | FileCheck %s

// CHECK-DAG: @__ocml_cabs_f32(complex<f32>) -> f32
// CHECK-DAG: @__ocml_cabs_f64(complex<f64>) -> f64
// CHECK-DAG: @__ocml_carg_f32(complex<f32>) -> f32
// CHECK-DAG: @__ocml_carg_f64(complex<f64>) -> f64
// CHECK-DAG: @__ocml_ccos_f32(complex<f32>) -> complex<f32>
// CHECK-DAG: @__ocml_ccos_f64(complex<f64>) -> complex<f64>
// CHECK-DAG: @__ocml_cexp_f32(complex<f32>) -> complex<f32>
// CHECK-DAG: @__ocml_cexp_f64(complex<f64>) -> complex<f64>
// CHECK-DAG: @__ocml_clog_f32(complex<f32>) -> complex<f32>
// CHECK-DAG: @__ocml_clog_f64(complex<f64>) -> complex<f64>
// CHECK-DAG: @__ocml_conj_f32(complex<f32>) -> complex<f32>
// CHECK-DAG: @__ocml_conj_f64(complex<f64>) -> complex<f64>
// CHECK-DAG: @__ocml_cpow_f32(complex<f32>, complex<f32>) -> complex<f32>
// CHECK-DAG: @__ocml_cpow_f64(complex<f64>, complex<f64>) -> complex<f64>
// CHECK-DAG: @__ocml_csin_f32(complex<f32>) -> complex<f32>
// CHECK-DAG: @__ocml_csin_f64(complex<f64>) -> complex<f64>
// CHECK-DAG: @__ocml_csqrt_f32(complex<f32>) -> complex<f32>
// CHECK-DAG: @__ocml_csqrt_f64(complex<f64>) -> complex<f64>
// CHECK-DAG: @__ocml_ctan_f32(complex<f32>) -> complex<f32>
// CHECK-DAG: @__ocml_ctan_f64(complex<f64>) -> complex<f64>
// CHECK-DAG: @__ocml_ctanh_f32(complex<f32>) -> complex<f32>
// CHECK-DAG: @__ocml_ctanh_f64(complex<f64>) -> complex<f64>

//CHECK-LABEL: @abs_caller
func.func @abs_caller(%f: complex<f32>, %d: complex<f64>) -> (f32, f64) {
  // CHECK: %[[RF:.*]] = call @__ocml_cabs_f32(%{{.*}})
  %rf = complex.abs %f : complex<f32>
  // CHECK: %[[RD:.*]] = call @__ocml_cabs_f64(%{{.*}})
  %rd = complex.abs %d : complex<f64>
  // CHECK: return %[[RF]], %[[RD]]
  return %rf, %rd : f32, f64
}

//CHECK-LABEL: @cos_caller
func.func @cos_caller(%f: complex<f32>, %d: complex<f64>) -> (complex<f32>, complex<f64>) {
  // CHECK: %[[CF:.*]] = call @__ocml_ccos_f32(%{{.*}})
  %cf = complex.cos %f : complex<f32>
  // CHECK: %[[CD:.*]] = call @__ocml_ccos_f64(%{{.*}})
  %cd = complex.cos %d : complex<f64>
  // CHECK: return %[[CF]], %[[CD]]
  return %cf, %cd : complex<f32>, complex<f64>
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

//CHECK-LABEL: @log_caller
func.func @log_caller(%f: complex<f32>, %d: complex<f64>) -> (complex<f32>, complex<f64>) {
  // CHECK: %[[LF:.*]] = call @__ocml_clog_f32(%{{.*}})
  %lf = complex.log %f : complex<f32>
  // CHECK: %[[LD:.*]] = call @__ocml_clog_f64(%{{.*}})
  %ld = complex.log %d : complex<f64>
  // CHECK: return %[[LF]], %[[LD]]
  return %lf, %ld : complex<f32>, complex<f64>
}

//CHECK-LABEL: @sin_caller
func.func @sin_caller(%f: complex<f32>, %d: complex<f64>) -> (complex<f32>, complex<f64>) {
  // CHECK: %[[SF:.*]] = call @__ocml_csin_f32(%{{.*}})
  %sf2 = complex.sin %f : complex<f32>
  // CHECK: %[[SD:.*]] = call @__ocml_csin_f64(%{{.*}})
  %sd2 = complex.sin %d : complex<f64>
  // CHECK: return %[[SF]], %[[SD]]
  return %sf2, %sd2 : complex<f32>, complex<f64>
}

//CHECK-LABEL: @sqrt_caller
func.func @sqrt_caller(%f: complex<f32>, %d: complex<f64>) -> (complex<f32>, complex<f64>) {
  // CHECK: %[[SF:.*]] = call @__ocml_csqrt_f32(%{{.*}})
  %sf = complex.sqrt %f : complex<f32>
  // CHECK: %[[SD:.*]] = call @__ocml_csqrt_f64(%{{.*}})
  %sd = complex.sqrt %d : complex<f64>
  // CHECK: return %[[SF]], %[[SD]]
  return %sf, %sd : complex<f32>, complex<f64>
}

//CHECK-LABEL: @tan_caller
func.func @tan_caller(%f: complex<f32>, %d: complex<f64>) -> (complex<f32>, complex<f64>) {
  // CHECK: %[[TF:.*]] = call @__ocml_ctan_f32(%{{.*}})
  %tf2 = complex.tan %f : complex<f32>
  // CHECK: %[[TD:.*]] = call @__ocml_ctan_f64(%{{.*}})
  %td2 = complex.tan %d : complex<f64>
  // CHECK: return %[[TF]], %[[TD]]
  return %tf2, %td2 : complex<f32>, complex<f64>
}

//CHECK-LABEL: @tanh_caller
func.func @tanh_caller(%f: complex<f32>, %d: complex<f64>) -> (complex<f32>, complex<f64>) {
  // CHECK: %[[TF:.*]] = call @__ocml_ctanh_f32(%{{.*}})
  %tf = complex.tanh %f : complex<f32>
  // CHECK: %[[TD:.*]] = call @__ocml_ctanh_f64(%{{.*}})
  %td = complex.tanh %d : complex<f64>
  // CHECK: return %[[TF]], %[[TD]]
  return %tf, %td : complex<f32>, complex<f64>
}
