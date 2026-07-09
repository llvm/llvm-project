// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// f16 + f16 -> f16
llvm.func @fadd_f16_f16(%a : f16, %b : f16) -> f16 {
  // CHECK-LABEL: define half @fadd_f16_f16(half %0, half %1) {
  // CHECK-NEXT: %3 = fadd half %0, %1
  // CHECK-NEXT: %4 = fadd half %3, %3
  // CHECK-NEXT: %5 = call half @llvm.nvvm.add.rn.sat.f16(half %4, half %4)
  // CHECK-NEXT: %6 = call half @llvm.nvvm.add.rn.ftz.sat.f16(half %5, half %5)
  // CHECK-NEXT: ret half %6
  // CHECK-NEXT: }
  %f1 = nvvm.addf %a, %b : f16
  %f2 = nvvm.addf %f1, %f1 {rnd = #nvvm.fp_rnd_mode<rn>} : f16
  %f3 = nvvm.addf %f2, %f2 {rnd = #nvvm.fp_rnd_mode<rn>, sat = #nvvm.sat_mode<sat>} : f16
  %f4 = nvvm.addf %f3, %f3 {rnd = #nvvm.fp_rnd_mode<rn>, sat = #nvvm.sat_mode<sat>, ftz=true} : f16
  llvm.return %f4 : f16
}

// bf16 + bf16 -> bf16
llvm.func @fadd_bf16_bf16(%a : bf16, %b : bf16) -> bf16 {
  // CHECK-LABEL: define bfloat @fadd_bf16_bf16(bfloat %0, bfloat %1) {
  // CHECK-NEXT: %3 = fadd bfloat %0, %1
  // CHECK-NEXT: %4 = fadd bfloat %3, %3
  // CHECK-NEXT: ret bfloat %4
  // CHECK-NEXT: }
  %f1 = nvvm.addf %a, %b : bf16
  %f2 = nvvm.addf %f1, %f1 {rnd = #nvvm.fp_rnd_mode<rn>} : bf16
  llvm.return %f2 : bf16
}

// f32 + f32 -> f32
llvm.func @fadd_f32_f32(%a : f32, %b : f32) -> f32 {
  // CHECK-LABEL: define float @fadd_f32_f32(float %0, float %1) {
  // CHECK-NEXT: %3 = call float @llvm.nvvm.add.rn.f(float %0, float %1)
  // CHECK-NEXT: %4 = call float @llvm.nvvm.add.rn.f(float %3, float %3)
  // CHECK-NEXT: %5 = call float @llvm.nvvm.add.rn.sat.f(float %4, float %4)
  // CHECK-NEXT: %6 = call float @llvm.nvvm.add.rn.ftz.f(float %5, float %5)
  // CHECK-NEXT: %7 = call float @llvm.nvvm.add.rn.ftz.sat.f(float %6, float %6)
  // CHECK-NEXT: %8 = call float @llvm.nvvm.add.rm.f(float %7, float %7)
  // CHECK-NEXT: %9 = call float @llvm.nvvm.add.rm.sat.f(float %8, float %8)
  // CHECK-NEXT: %10 = call float @llvm.nvvm.add.rm.ftz.f(float %9, float %9)
  // CHECK-NEXT: %11 = call float @llvm.nvvm.add.rm.ftz.sat.f(float %10, float %10)
  // CHECK-NEXT: %12 = call float @llvm.nvvm.add.rp.f(float %11, float %11)
  // CHECK-NEXT: %13 = call float @llvm.nvvm.add.rp.sat.f(float %12, float %12)
  // CHECK-NEXT: %14 = call float @llvm.nvvm.add.rp.ftz.f(float %13, float %13)
  // CHECK-NEXT: %15 = call float @llvm.nvvm.add.rp.ftz.sat.f(float %14, float %14)
  // CHECK-NEXT: %16 = call float @llvm.nvvm.add.rz.f(float %15, float %15)
  // CHECK-NEXT: %17 = call float @llvm.nvvm.add.rz.sat.f(float %16, float %16)
  // CHECK-NEXT: %18 = call float @llvm.nvvm.add.rz.ftz.f(float %17, float %17)
  // CHECK-NEXT: %19 = call float @llvm.nvvm.add.rz.ftz.sat.f(float %18, float %18)
  // CHECK-NEXT: ret float %19
  // CHECK-NEXT: }
  %f1 = nvvm.addf %a, %b : f32
  %f2 = nvvm.addf %f1, %f1 {rnd = #nvvm.fp_rnd_mode<rn>} : f32
  %f3 = nvvm.addf %f2, %f2 {rnd = #nvvm.fp_rnd_mode<rn>, sat = #nvvm.sat_mode<sat>} : f32
  %f4 = nvvm.addf %f3, %f3 {rnd = #nvvm.fp_rnd_mode<rn>, ftz=true} : f32
  %f5 = nvvm.addf %f4, %f4 {rnd = #nvvm.fp_rnd_mode<rn>, sat = #nvvm.sat_mode<sat>, ftz=true} : f32
  %f6 = nvvm.addf %f5, %f5 {rnd = #nvvm.fp_rnd_mode<rm>} : f32
  %f7 = nvvm.addf %f6, %f6 {rnd = #nvvm.fp_rnd_mode<rm>, sat = #nvvm.sat_mode<sat>} : f32
  %f8 = nvvm.addf %f7, %f7 {rnd = #nvvm.fp_rnd_mode<rm>, ftz=true} : f32
  %f9 = nvvm.addf %f8, %f8 {rnd = #nvvm.fp_rnd_mode<rm>, sat = #nvvm.sat_mode<sat>, ftz=true} : f32
  %f10 = nvvm.addf %f9, %f9 {rnd = #nvvm.fp_rnd_mode<rp>} : f32
  %f11 = nvvm.addf %f10, %f10 {rnd = #nvvm.fp_rnd_mode<rp>, sat = #nvvm.sat_mode<sat>} : f32
  %f12 = nvvm.addf %f11, %f11 {rnd = #nvvm.fp_rnd_mode<rp>, ftz=true} : f32
  %f13 = nvvm.addf %f12, %f12 {rnd = #nvvm.fp_rnd_mode<rp>, sat = #nvvm.sat_mode<sat>, ftz=true} : f32
  %f14 = nvvm.addf %f13, %f13 {rnd = #nvvm.fp_rnd_mode<rz>} : f32
  %f15 = nvvm.addf %f14, %f14 {rnd = #nvvm.fp_rnd_mode<rz>, sat = #nvvm.sat_mode<sat>} : f32
  %f16 = nvvm.addf %f15, %f15 {rnd = #nvvm.fp_rnd_mode<rz>, ftz=true} : f32
  %f17 = nvvm.addf %f16, %f16 {rnd = #nvvm.fp_rnd_mode<rz>, sat = #nvvm.sat_mode<sat>, ftz=true} : f32
  llvm.return %f17 : f32
}

// f64 + f64 -> f64
llvm.func @fadd_f64_f64(%a : f64, %b : f64) -> f64 {
  // CHECK-LABEL: define double @fadd_f64_f64(double %0, double %1) {
  // CHECK-NEXT: %3 = call double @llvm.nvvm.add.rn.d(double %0, double %1)
  // CHECK-NEXT: %4 = call double @llvm.nvvm.add.rn.d(double %3, double %3)
  // CHECK-NEXT: %5 = call double @llvm.nvvm.add.rm.d(double %4, double %4)
  // CHECK-NEXT: %6 = call double @llvm.nvvm.add.rp.d(double %5, double %5)
  // CHECK-NEXT: %7 = call double @llvm.nvvm.add.rz.d(double %6, double %6)
  // CHECK-NEXT: ret double %7
  // CHECK-NEXT: }
  %f1 = nvvm.addf %a, %b : f64
  %f2 = nvvm.addf %f1, %f1 {rnd = #nvvm.fp_rnd_mode<rn>} : f64
  %f3 = nvvm.addf %f2, %f2 {rnd = #nvvm.fp_rnd_mode<rm>} : f64
  %f4 = nvvm.addf %f3, %f3 {rnd = #nvvm.fp_rnd_mode<rp>} : f64
  %f5 = nvvm.addf %f4, %f4 {rnd = #nvvm.fp_rnd_mode<rz>} : f64
  llvm.return %f5 : f64
}
