// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @fma_f16(%a: f16, %b: f16, %c: f16) -> f16 {
  // CHECK-LABEL: define half @fma_f16(half %0, half %1, half %2) {
  // CHECK-NEXT: %4 = call half @llvm.nvvm.fma.rn.f16(half %0, half %1, half %2)
  // CHECK-NEXT: %5 = call half @llvm.nvvm.fma.rn.ftz.f16(half %0, half %1, half %4)
  // CHECK-NEXT: %6 = call half @llvm.nvvm.fma.rn.sat.f16(half %0, half %1, half %5)
  // CHECK-NEXT: %7 = call half @llvm.nvvm.fma.rn.ftz.sat.f16(half %0, half %1, half %6)
  // CHECK-NEXT: %8 = call half @llvm.nvvm.fma.rn.relu.f16(half %0, half %1, half %7)
  // CHECK-NEXT: %9 = call half @llvm.nvvm.fma.rn.ftz.relu.f16(half %0, half %1, half %8)
  // CHECK-NEXT: %10 = call half @llvm.nvvm.fma.rn.oob.f16(half %0, half %1, half %9)
  // CHECK-NEXT: %11 = call half @llvm.nvvm.fma.rn.oob.relu.f16(half %0, half %1, half %10)
  // CHECK-NEXT: ret half %11
  // CHECK-NEXT: }
  %f0 = nvvm.fma %a, %b, %c {rnd = #nvvm.fp_rnd_mode<rn>} : f16
  %f1 = nvvm.fma %a, %b, %f0 {rnd = #nvvm.fp_rnd_mode<rn>, ftz = true} : f16
  %f2 = nvvm.fma %a, %b, %f1 {rnd = #nvvm.fp_rnd_mode<rn>, sat = #nvvm.sat_mode<sat>} : f16
  %f3 = nvvm.fma %a, %b, %f2 {rnd = #nvvm.fp_rnd_mode<rn>, sat = #nvvm.sat_mode<sat>, ftz = true} : f16
  %f4 = nvvm.fma %a, %b, %f3 {rnd = #nvvm.fp_rnd_mode<rn>, relu = true} : f16
  %f5 = nvvm.fma %a, %b, %f4 {rnd = #nvvm.fp_rnd_mode<rn>, relu = true, ftz = true} : f16
  %f6 = nvvm.fma %a, %b, %f5 {rnd = #nvvm.fp_rnd_mode<rn>, oob = true} : f16
  %f7 = nvvm.fma %a, %b, %f6 {rnd = #nvvm.fp_rnd_mode<rn>, oob = true, relu = true} : f16
  llvm.return %f7 : f16
}

llvm.func @fma_bf16(%a: bf16, %b: bf16, %c: bf16) -> bf16 {
  // CHECK-LABEL: define bfloat @fma_bf16(bfloat %0, bfloat %1, bfloat %2) {
  // CHECK-NEXT: %4 = call bfloat @llvm.nvvm.fma.rn.bf16(bfloat %0, bfloat %1, bfloat %2)
  // CHECK-NEXT: %5 = call bfloat @llvm.nvvm.fma.rn.relu.bf16(bfloat %0, bfloat %1, bfloat %4)
  // CHECK-NEXT: %6 = call bfloat @llvm.nvvm.fma.rn.oob.bf16(bfloat %0, bfloat %1, bfloat %5)
  // CHECK-NEXT: %7 = call bfloat @llvm.nvvm.fma.rn.oob.relu.bf16(bfloat %0, bfloat %1, bfloat %6)
  // CHECK-NEXT: ret bfloat %7
  // CHECK-NEXT: }
  %f0 = nvvm.fma %a, %b, %c {rnd = #nvvm.fp_rnd_mode<rn>} : bf16
  %f1 = nvvm.fma %a, %b, %f0 {rnd = #nvvm.fp_rnd_mode<rn>, relu = true} : bf16
  %f2 = nvvm.fma %a, %b, %f1 {rnd = #nvvm.fp_rnd_mode<rn>, oob = true} : bf16
  %f3 = nvvm.fma %a, %b, %f2 {rnd = #nvvm.fp_rnd_mode<rn>, oob = true, relu = true} : bf16
  llvm.return %f3 : bf16
}

llvm.func @fma_f32_rn(%a: f32, %b: f32, %c: f32) -> f32 {
  // CHECK-LABEL: define float @fma_f32_rn(float %0, float %1, float %2) {
  // CHECK-NEXT: %4 = call float @llvm.nvvm.fma.rn.f(float %0, float %1, float %2)
  // CHECK-NEXT: %5 = call float @llvm.nvvm.fma.rn.ftz.f(float %0, float %1, float %4)
  // CHECK-NEXT: %6 = call float @llvm.nvvm.fma.rn.sat.f(float %0, float %1, float %5)
  // CHECK-NEXT: %7 = call float @llvm.nvvm.fma.rn.ftz.sat.f(float %0, float %1, float %6)
  // CHECK-NEXT: ret float %7
  // CHECK-NEXT: }
  %f0 = nvvm.fma %a, %b, %c {rnd = #nvvm.fp_rnd_mode<rn>} : f32
  %f1 = nvvm.fma %a, %b, %f0 {rnd = #nvvm.fp_rnd_mode<rn>, ftz = true} : f32
  %f2 = nvvm.fma %a, %b, %f1 {rnd = #nvvm.fp_rnd_mode<rn>, sat = #nvvm.sat_mode<sat>} : f32
  %f3 = nvvm.fma %a, %b, %f2 {rnd = #nvvm.fp_rnd_mode<rn>, sat = #nvvm.sat_mode<sat>, ftz = true} : f32
  llvm.return %f3 : f32
}

llvm.func @fma_f32_rm(%a: f32, %b: f32, %c: f32) -> f32 {
  // CHECK-LABEL: define float @fma_f32_rm(float %0, float %1, float %2) {
  // CHECK-NEXT: %4 = call float @llvm.nvvm.fma.rm.f(float %0, float %1, float %2)
  // CHECK-NEXT: %5 = call float @llvm.nvvm.fma.rm.ftz.f(float %0, float %1, float %4)
  // CHECK-NEXT: %6 = call float @llvm.nvvm.fma.rm.sat.f(float %0, float %1, float %5)
  // CHECK-NEXT: %7 = call float @llvm.nvvm.fma.rm.ftz.sat.f(float %0, float %1, float %6)
  // CHECK-NEXT: ret float %7
  // CHECK-NEXT: }
  %f0 = nvvm.fma %a, %b, %c {rnd = #nvvm.fp_rnd_mode<rm>} : f32
  %f1 = nvvm.fma %a, %b, %f0 {rnd = #nvvm.fp_rnd_mode<rm>, ftz = true} : f32
  %f2 = nvvm.fma %a, %b, %f1 {rnd = #nvvm.fp_rnd_mode<rm>, sat = #nvvm.sat_mode<sat>} : f32
  %f3 = nvvm.fma %a, %b, %f2 {rnd = #nvvm.fp_rnd_mode<rm>, sat = #nvvm.sat_mode<sat>, ftz = true} : f32
  llvm.return %f3 : f32
}

llvm.func @fma_f32_rp(%a: f32, %b: f32, %c: f32) -> f32 {
  // CHECK-LABEL: define float @fma_f32_rp(float %0, float %1, float %2) {
  // CHECK-NEXT: %4 = call float @llvm.nvvm.fma.rp.f(float %0, float %1, float %2)
  // CHECK-NEXT: %5 = call float @llvm.nvvm.fma.rp.ftz.f(float %0, float %1, float %4)
  // CHECK-NEXT: %6 = call float @llvm.nvvm.fma.rp.sat.f(float %0, float %1, float %5)
  // CHECK-NEXT: %7 = call float @llvm.nvvm.fma.rp.ftz.sat.f(float %0, float %1, float %6)
  // CHECK-NEXT: ret float %7
  // CHECK-NEXT: }
  %f0 = nvvm.fma %a, %b, %c {rnd = #nvvm.fp_rnd_mode<rp>} : f32
  %f1 = nvvm.fma %a, %b, %f0 {rnd = #nvvm.fp_rnd_mode<rp>, ftz = true} : f32
  %f2 = nvvm.fma %a, %b, %f1 {rnd = #nvvm.fp_rnd_mode<rp>, sat = #nvvm.sat_mode<sat>} : f32
  %f3 = nvvm.fma %a, %b, %f2 {rnd = #nvvm.fp_rnd_mode<rp>, sat = #nvvm.sat_mode<sat>, ftz = true} : f32
  llvm.return %f3 : f32
}

llvm.func @fma_f32_rz(%a: f32, %b: f32, %c: f32) -> f32 {
  // CHECK-LABEL: define float @fma_f32_rz(float %0, float %1, float %2) {
  // CHECK-NEXT: %4 = call float @llvm.nvvm.fma.rz.f(float %0, float %1, float %2)
  // CHECK-NEXT: %5 = call float @llvm.nvvm.fma.rz.ftz.f(float %0, float %1, float %4)
  // CHECK-NEXT: %6 = call float @llvm.nvvm.fma.rz.sat.f(float %0, float %1, float %5)
  // CHECK-NEXT: %7 = call float @llvm.nvvm.fma.rz.ftz.sat.f(float %0, float %1, float %6)
  // CHECK-NEXT: ret float %7
  // CHECK-NEXT: }
  %f0 = nvvm.fma %a, %b, %c {rnd = #nvvm.fp_rnd_mode<rz>} : f32
  %f1 = nvvm.fma %a, %b, %f0 {rnd = #nvvm.fp_rnd_mode<rz>, ftz = true} : f32
  %f2 = nvvm.fma %a, %b, %f1 {rnd = #nvvm.fp_rnd_mode<rz>, sat = #nvvm.sat_mode<sat>} : f32
  %f3 = nvvm.fma %a, %b, %f2 {rnd = #nvvm.fp_rnd_mode<rz>, sat = #nvvm.sat_mode<sat>, ftz = true} : f32
  llvm.return %f3 : f32
}

llvm.func @fma_f64(%a: f64, %b: f64, %c: f64) -> f64 {
  // CHECK-LABEL: define double @fma_f64(double %0, double %1, double %2) {
  // CHECK-NEXT: %4 = call double @llvm.nvvm.fma.rn.d(double %0, double %1, double %2)
  // CHECK-NEXT: %5 = call double @llvm.nvvm.fma.rm.d(double %0, double %1, double %4)
  // CHECK-NEXT: %6 = call double @llvm.nvvm.fma.rp.d(double %0, double %1, double %5)
  // CHECK-NEXT: %7 = call double @llvm.nvvm.fma.rz.d(double %0, double %1, double %6)
  // CHECK-NEXT: ret double %7
  // CHECK-NEXT: }
  %f0 = nvvm.fma %a, %b, %c {rnd = #nvvm.fp_rnd_mode<rn>} : f64
  %f1 = nvvm.fma %a, %b, %f0 {rnd = #nvvm.fp_rnd_mode<rm>} : f64
  %f2 = nvvm.fma %a, %b, %f1 {rnd = #nvvm.fp_rnd_mode<rp>} : f64
  %f3 = nvvm.fma %a, %b, %f2 {rnd = #nvvm.fp_rnd_mode<rz>} : f64
  llvm.return %f3 : f64
}
