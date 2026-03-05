// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: define float @nvvm_log2_approx_f(float %0)
// CHECK: call float @llvm.nvvm.lg2.approx.f(float %0)
llvm.func @nvvm_log2_approx_f(%arg0: f32) -> f32 {
  %0 = nvvm.log2.approx.f %arg0 : f32
  llvm.return %0 : f32
}

// CHECK-LABEL: define float @nvvm_log2_approx_ftz_f(float %0)
// CHECK: call float @llvm.nvvm.lg2.approx.ftz.f(float %0)
llvm.func @nvvm_log2_approx_ftz_f(%arg0: f32) -> f32 {
  %0 = nvvm.log2.approx.f %arg0 {ftz = true} : f32
  llvm.return %0 : f32
}
