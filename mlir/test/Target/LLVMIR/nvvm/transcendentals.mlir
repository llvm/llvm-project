// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: @nvvm_tanh
llvm.func @nvvm_tanh(%arg0: f32) -> f32 {
  // CHECK: call afn float @llvm.tanh.f32(float %{{.*}})
  %0 = nvvm.tanh %arg0 : f32
  llvm.return %0 : f32
}
