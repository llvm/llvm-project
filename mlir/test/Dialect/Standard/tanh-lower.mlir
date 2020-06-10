// RUN: mlir-opt -tanh-lower -convert-std-to-llvm %s | FileCheck %s

module {
  // CHECK-LABEL: @tanh_lower_f64
  func @tanh_lower_f64() {
    %c0 = constant 1.0 : f64
    // CHECK: %[[A:.*]] = llvm.call @tanh(%[[B:.*]]) : (!llvm.double) -> !llvm.double
    %1 = tanh %c0 : f64
    return
  }

  // CHECK-LABEL: @tanh_lower_f32
  func @tanh_lower_f32() {
    %c0 = constant 1.0 : f32
    // CHECK: %[[A:.*]] = llvm.call @tanhf(%[[B:.*]]) : (!llvm.float) -> !llvm.float
    %1 = tanh %c0 : f32
    return
  }
}