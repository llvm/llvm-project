// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// f32 divf — rounded, all 8 forms (4 rounding modes × 2 ftz states).
llvm.func @divf_f32(%a : f32, %b : f32) -> f32 {
  // CHECK-LABEL: define float @divf_f32(float %0, float %1) {
  // CHECK: call float @llvm.nvvm.div.rn.f(float %{{.*}}, float %{{.*}})
  // CHECK: call float @llvm.nvvm.div.rz.f(float %{{.*}}, float %{{.*}})
  // CHECK: call float @llvm.nvvm.div.rm.f(float %{{.*}}, float %{{.*}})
  // CHECK: call float @llvm.nvvm.div.rp.f(float %{{.*}}, float %{{.*}})
  // CHECK: call float @llvm.nvvm.div.rn.ftz.f(float %{{.*}}, float %{{.*}})
  // CHECK: call float @llvm.nvvm.div.rz.ftz.f(float %{{.*}}, float %{{.*}})
  // CHECK: call float @llvm.nvvm.div.rm.ftz.f(float %{{.*}}, float %{{.*}})
  // CHECK: call float @llvm.nvvm.div.rp.ftz.f(float %{{.*}}, float %{{.*}})
  %r1 = nvvm.divf %a,  %b  {rnd = #nvvm.fp_rnd_mode<rn>} : f32
  %r2 = nvvm.divf %r1, %r1 {rnd = #nvvm.fp_rnd_mode<rz>} : f32
  %r3 = nvvm.divf %r2, %r2 {rnd = #nvvm.fp_rnd_mode<rm>} : f32
  %r4 = nvvm.divf %r3, %r3 {rnd = #nvvm.fp_rnd_mode<rp>} : f32
  %r5 = nvvm.divf %r4, %r4 {rnd = #nvvm.fp_rnd_mode<rn>, ftz = true} : f32
  %r6 = nvvm.divf %r5, %r5 {rnd = #nvvm.fp_rnd_mode<rz>, ftz = true} : f32
  %r7 = nvvm.divf %r6, %r6 {rnd = #nvvm.fp_rnd_mode<rm>, ftz = true} : f32
  %r8 = nvvm.divf %r7, %r7 {rnd = #nvvm.fp_rnd_mode<rp>, ftz = true} : f32
  llvm.return %r8 : f32
}

// f64 divf — rounded, all 4 forms (no ftz).
llvm.func @divf_f64(%a : f64, %b : f64) -> f64 {
  // CHECK-LABEL: define double @divf_f64(double %0, double %1) {
  // CHECK: call double @llvm.nvvm.div.rn.d(double %{{.*}}, double %{{.*}})
  // CHECK: call double @llvm.nvvm.div.rz.d(double %{{.*}}, double %{{.*}})
  // CHECK: call double @llvm.nvvm.div.rm.d(double %{{.*}}, double %{{.*}})
  // CHECK: call double @llvm.nvvm.div.rp.d(double %{{.*}}, double %{{.*}})
  %r1 = nvvm.divf %a,  %b  {rnd = #nvvm.fp_rnd_mode<rn>} : f64
  %r2 = nvvm.divf %r1, %r1 {rnd = #nvvm.fp_rnd_mode<rz>} : f64
  %r3 = nvvm.divf %r2, %r2 {rnd = #nvvm.fp_rnd_mode<rm>} : f64
  %r4 = nvvm.divf %r3, %r3 {rnd = #nvvm.fp_rnd_mode<rp>} : f64
  llvm.return %r4 : f64
}

// divf.approx — 2 forms.
llvm.func @divf_approx(%a : f32, %b : f32) -> f32 {
  // CHECK-LABEL: define float @divf_approx(float %0, float %1) {
  // CHECK: call float @llvm.nvvm.div.approx.f(float %{{.*}}, float %{{.*}})
  // CHECK: call float @llvm.nvvm.div.approx.ftz.f(float %{{.*}}, float %{{.*}})
  %r1 = nvvm.divf %a,  %b  {approx = true} : f32
  %r2 = nvvm.divf %r1, %r1 {approx = true, ftz = true} : f32
  llvm.return %r2 : f32
}

// divf.full — 2 forms.
llvm.func @divf_full(%a : f32, %b : f32) -> f32 {
  // CHECK-LABEL: define float @divf_full(float %0, float %1) {
  // CHECK: call float @llvm.nvvm.div.full(float %{{.*}}, float %{{.*}})
  // CHECK: call float @llvm.nvvm.div.full.ftz(float %{{.*}}, float %{{.*}})
  %r1 = nvvm.divf %a,  %b  {full = true} : f32
  %r2 = nvvm.divf %r1, %r1 {full = true, ftz = true} : f32
  llvm.return %r2 : f32
}
