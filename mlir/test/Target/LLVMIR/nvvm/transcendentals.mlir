// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: @nvvm_sin
llvm.func @nvvm_sin(%arg0: f32) -> f32 {
  // CHECK: call float @llvm.nvvm.sin.approx.f(float %{{.*}})
  %0 = nvvm.sin %arg0 : f32
  llvm.return %0 : f32
}

// CHECK-LABEL: @nvvm_sin_ftz
llvm.func @nvvm_sin_ftz(%arg0: f32) -> f32 {
  // CHECK: call float @llvm.nvvm.sin.approx.ftz.f(float %{{.*}})
  %0 = nvvm.sin %arg0 {ftz = true} : f32
  llvm.return %0 : f32
}
