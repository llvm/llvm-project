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
  %0 = nvvm.sin %arg0 ftz = true : f32
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
  %0 = nvvm.cos %arg0 ftz = true : f32
  llvm.return %0 : f32
}

// CHECK-LABEL: @nvvm_lg2
llvm.func @nvvm_lg2(%arg0: f32) -> f32 {
  // CHECK: call float @llvm.nvvm.lg2.approx.f(float %{{.*}})
  %0 = nvvm.log2 %arg0 : f32
  llvm.return %0 : f32
}

// CHECK-LABEL: @nvvm_lg2_ftz
llvm.func @nvvm_lg2_ftz(%arg0: f32) -> f32 {
  // CHECK: call float @llvm.nvvm.lg2.approx.ftz.f(float %{{.*}})
  %0 = nvvm.log2 %arg0 ftz = true : f32
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
  %0 = nvvm.ex2 %arg0 ftz = true : f32
  llvm.return %0 : f32
}

// CHECK-LABEL: @nvvm_ex2_f16x2
llvm.func @nvvm_ex2_f16x2(%arg0: vector<2xf16>) -> vector<2xf16> {
  // CHECK: call <2 x half> @llvm.nvvm.ex2.approx.v2f16(<2 x half> %{{.*}})
  %0 = nvvm.ex2 %arg0 : vector<2xf16>
  llvm.return %0 : vector<2xf16>
}

// CHECK-LABEL: @nvvm_ex2_bf16x2
llvm.func @nvvm_ex2_bf16x2(%arg0: vector<2xbf16>) -> vector<2xbf16> {
  // CHECK: call <2 x bfloat> @llvm.nvvm.ex2.approx.ftz.v2bf16(<2 x bfloat> %{{.*}})
  %0 = nvvm.ex2 %arg0 ftz = true : vector<2xbf16>
  llvm.return %0 : vector<2xbf16>
}
