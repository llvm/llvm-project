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

// CHECK-LABEL: @nvvm_cos
llvm.func @nvvm_cos(%arg0: f32) -> f32 {
  // CHECK: call float @llvm.nvvm.cos.approx.f(float %{{.*}})
  %0 = nvvm.cos %arg0 : f32
  llvm.return %0 : f32
}

// CHECK-LABEL: @nvvm_cos_ftz
llvm.func @nvvm_cos_ftz(%arg0: f32) -> f32 {
  // CHECK: call float @llvm.nvvm.cos.approx.ftz.f(float %{{.*}})
  %0 = nvvm.cos %arg0 {ftz = true} : f32
  llvm.return %0 : f32
}

// CHECK-LABEL: @nvvm_ex2
llvm.func @nvvm_ex2(%arg0: f32) -> f32 {
  // CHECK: call float @llvm.nvvm.ex2.approx.f32(float %{{.*}})
  %0 = nvvm.ex2 %arg0 : f32
  llvm.return %0 : f32
}

// CHECK-LABEL: @nvvm_ex2_ftz
llvm.func @nvvm_ex2_ftz(%arg0: f32) -> f32 {
  // CHECK: call float @llvm.nvvm.ex2.approx.ftz.f32(float %{{.*}})
  %0 = nvvm.ex2 %arg0 {ftz = true} : f32
  llvm.return %0 : f32
}
