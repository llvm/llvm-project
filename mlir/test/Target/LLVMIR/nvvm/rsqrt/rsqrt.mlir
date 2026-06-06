// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// f32 rsqrt -- non-ftz and ftz forms.
llvm.func @rsqrt_f32(%a : f32) -> f32 {
  // CHECK-LABEL: define float @rsqrt_f32(float %0) {
  // CHECK: call float @llvm.nvvm.rsqrt.approx.f(float %{{.*}})
  // CHECK: call float @llvm.nvvm.rsqrt.approx.ftz.f(float %{{.*}})
  %r1 = nvvm.rsqrt %a : f32
  %r2 = nvvm.rsqrt %r1 {ftz = true} : f32
  llvm.return %r2 : f32
}

// f64 rsqrt -- non-ftz and ftz forms.
llvm.func @rsqrt_f64(%a : f64) -> f64 {
  // CHECK-LABEL: define double @rsqrt_f64(double %0) {
  // CHECK: call double @llvm.nvvm.rsqrt.approx.d(double %{{.*}})
  // CHECK: call double @llvm.nvvm.rsqrt.approx.ftz.d(double %{{.*}})
  %r1 = nvvm.rsqrt %a : f64
  %r2 = nvvm.rsqrt %r1 {ftz = true} : f64
  llvm.return %r2 : f64
}
