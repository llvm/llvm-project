// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// f32 sqrt — all 8 forms (4 rounding modes × 2 ftz states).
llvm.func @sqrt_f32(%a : f32) -> f32 {
  // CHECK-LABEL: define float @sqrt_f32(float %0) {
  // CHECK: call float @llvm.nvvm.sqrt.rn.f(float %{{.*}})
  // CHECK: call float @llvm.nvvm.sqrt.rz.f(float %{{.*}})
  // CHECK: call float @llvm.nvvm.sqrt.rm.f(float %{{.*}})
  // CHECK: call float @llvm.nvvm.sqrt.rp.f(float %{{.*}})
  // CHECK: call float @llvm.nvvm.sqrt.rn.ftz.f(float %{{.*}})
  // CHECK: call float @llvm.nvvm.sqrt.rz.ftz.f(float %{{.*}})
  // CHECK: call float @llvm.nvvm.sqrt.rm.ftz.f(float %{{.*}})
  // CHECK: call float @llvm.nvvm.sqrt.rp.ftz.f(float %{{.*}})
  %r1 = nvvm.sqrt %a  {rnd = #nvvm.fp_rnd_mode<rn>} : f32
  %r2 = nvvm.sqrt %r1 {rnd = #nvvm.fp_rnd_mode<rz>} : f32
  %r3 = nvvm.sqrt %r2 {rnd = #nvvm.fp_rnd_mode<rm>} : f32
  %r4 = nvvm.sqrt %r3 {rnd = #nvvm.fp_rnd_mode<rp>} : f32
  %r5 = nvvm.sqrt %r4 {rnd = #nvvm.fp_rnd_mode<rn>, ftz = true} : f32
  %r6 = nvvm.sqrt %r5 {rnd = #nvvm.fp_rnd_mode<rz>, ftz = true} : f32
  %r7 = nvvm.sqrt %r6 {rnd = #nvvm.fp_rnd_mode<rm>, ftz = true} : f32
  %r8 = nvvm.sqrt %r7 {rnd = #nvvm.fp_rnd_mode<rp>, ftz = true} : f32
  llvm.return %r8 : f32
}

// f64 sqrt — all 4 forms (4 rounding modes, no ftz).
llvm.func @sqrt_f64(%a : f64) -> f64 {
  // CHECK-LABEL: define double @sqrt_f64(double %0) {
  // CHECK: call double @llvm.nvvm.sqrt.rn.d(double %{{.*}})
  // CHECK: call double @llvm.nvvm.sqrt.rz.d(double %{{.*}})
  // CHECK: call double @llvm.nvvm.sqrt.rm.d(double %{{.*}})
  // CHECK: call double @llvm.nvvm.sqrt.rp.d(double %{{.*}})
  %r1 = nvvm.sqrt %a  {rnd = #nvvm.fp_rnd_mode<rn>} : f64
  %r2 = nvvm.sqrt %r1 {rnd = #nvvm.fp_rnd_mode<rz>} : f64
  %r3 = nvvm.sqrt %r2 {rnd = #nvvm.fp_rnd_mode<rm>} : f64
  %r4 = nvvm.sqrt %r3 {rnd = #nvvm.fp_rnd_mode<rp>} : f64
  llvm.return %r4 : f64
}

// sqrt.approx — 2 forms.
llvm.func @sqrt_approx(%a : f32) -> f32 {
  // CHECK-LABEL: define float @sqrt_approx(float %0) {
  // CHECK: call float @llvm.nvvm.sqrt.approx.f(float %{{.*}})
  // CHECK: call float @llvm.nvvm.sqrt.approx.ftz.f(float %{{.*}})
  %r1 = nvvm.sqrt.approx %a  : f32
  %r2 = nvvm.sqrt.approx %r1 {ftz = true} : f32
  llvm.return %r2 : f32
}
