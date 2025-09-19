// RUN: mlir-opt %s -test-math-algebraic-simplification | FileCheck %s

func.func @pow3(%arg0: complex<f32>) -> complex<f32> {
  %c3 = arith.constant 3 : i32
  %0 = complex.powi %arg0, %c3 : complex<f32>, i32
  return %0 : complex<f32>
}
// CHECK-LABEL: func.func @pow3(
// CHECK-NOT: complex.powi
// CHECK: %[[M0:.+]] = complex.mul %{{.*}}, %{{.*}} : complex<f32>
// CHECK: %[[M1:.+]] = complex.mul %[[M0]], %{{.*}} : complex<f32>
// CHECK: return %[[M1]] : complex<f32>

func.func @pow9(%arg0: complex<f32>) -> complex<f32> {
  %c9 = arith.constant 9 : i32
  %0 = complex.powi %arg0, %c9 : complex<f32>, i32
  return %0 : complex<f32>
}
// CHECK-LABEL: func.func @pow9(
// CHECK: complex.powi %{{.*}}, %{{.*}} : complex<f32>, i32
