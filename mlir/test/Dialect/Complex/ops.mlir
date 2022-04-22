// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s


// CHECK-LABEL: func @ops(
// CHECK-SAME:            %[[F:.*]]: f32) {
func.func @ops(%f: f32) {
  // CHECK: complex.constant [1.{{.*}}, -1.{{.*}}] : complex<f64>
  %cst_f64 = complex.constant [0.1, -1.0] : complex<f64>

  // CHECK: complex.constant [1.{{.*}} : f32, -1.{{.*}} : f32] : complex<f32>
  %cst_f32 = complex.constant [0.1 : f32, -1.0 : f32] : complex<f32>

  // CHECK: %[[C:.*]] = complex.create %[[F]], %[[F]] : complex<f32>
  %complex = complex.create %f, %f : complex<f32>

  // CHECK: complex.re %[[C]] : complex<f32>
  %real = complex.re %complex : complex<f32>

  // CHECK: complex.im %[[C]] : complex<f32>
  %imag = complex.im %complex : complex<f32>

  // CHECK: complex.abs %[[C]] : complex<f32>
  %abs = complex.abs %complex : complex<f32>

  // CHECK: complex.add %[[C]], %[[C]] : complex<f32>
  %sum = complex.add %complex, %complex : complex<f32>

  // CHECK: complex.div %[[C]], %[[C]] : complex<f32>
  %div = complex.div %complex, %complex : complex<f32>

  // CHECK: complex.eq %[[C]], %[[C]] : complex<f32>
  %eq = complex.eq %complex, %complex : complex<f32>

  // CHECK: complex.exp %[[C]] : complex<f32>
  %exp = complex.exp %complex : complex<f32>

  // CHECK: complex.log %[[C]] : complex<f32>
  %log = complex.log %complex : complex<f32>

  // CHECK: complex.log1p %[[C]] : complex<f32>
  %log1p = complex.log1p %complex : complex<f32>

  // CHECK: complex.mul %[[C]], %[[C]] : complex<f32>
  %prod = complex.mul %complex, %complex : complex<f32>

  // CHECK: complex.neg %[[C]] : complex<f32>
  %neg = complex.neg %complex : complex<f32>

  // CHECK: complex.neq %[[C]], %[[C]] : complex<f32>
  %neq = complex.neq %complex, %complex : complex<f32>

  // CHECK: complex.sign %[[C]] : complex<f32>
  %sign = complex.sign %complex : complex<f32>

  // CHECK: complex.sub %[[C]], %[[C]] : complex<f32>
  %diff = complex.sub %complex, %complex : complex<f32>
  return
}
