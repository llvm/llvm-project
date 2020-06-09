// RUN: mlir-opt -tanh-lower -convert-std-to-llvm %s | pmlc-jit | FileCheck %s

module {
  // CHECK-LABEL: @tanh_lower_f64
  func @tanh_lower_f64() {
    %c0 = constant 1.0 : f64
    // CHECK: %[[.*]] = call @tanh%[[.*]]) : (f64) -> f64
    %1 = tanh %c0 : f64
    return
  }

  // CHECK-LABEL: @tanh_lower_f32
  func @tanh_lower_f32() {
    %c0 = constant 1.0 : f32
    // CHECK: %[[.*]] = call @tanhf(%[[.*]]) : (f32) -> f32
    %1 = tanh %c0 : f32
    return
  }
}