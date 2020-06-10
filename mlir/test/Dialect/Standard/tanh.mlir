// RUN: mlir-opt -tanh-lowering %s | FileCheck %s

// CHECK-LABEL: @tanh_f64
func @tanh_f64() {
  %cst = constant 1.0 : f64
  // CHECK: call @tanh(%{{.*}}) : (f64) -> f64
  %1 = tanh %cst : f64
  return
}

// CHECK-LABEL: @tanh_f32
func @tanh_f32() {
  %cst = constant 1.0 : f32
  // CHECK: call @tanhf(%{{.*}}) : (f32) -> f32
  %1 = tanh %cst : f32
  return
}
