// RUN: mlir-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @nvvm_tanh_f32
func.func @nvvm_tanh_f32(%arg0: f32) -> f32 {
  // CHECK: nvvm.tanh {{.*}} : f32
  %0 = nvvm.tanh %arg0 : f32
  return %0 : f32
}
