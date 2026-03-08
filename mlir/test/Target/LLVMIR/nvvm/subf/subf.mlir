// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// f16 - f16 -> f16
llvm.func @fsub_f16_f16(%a : f16, %b : f16) -> f16 {
  // CHECK-LABEL: define half @fsub_f16_f16(half %0, half %1) {
  // CHECK-NEXT: %3 = fneg half %1
  // CHECK-NEXT: %4 = fadd half %0, %3
  // CHECK-NEXT: %5 = fneg half %4
  // CHECK-NEXT: %6 = fadd half %4, %5
  // CHECK-NEXT: %7 = fneg half %6
  // CHECK-NEXT: %8 = call half @llvm.nvvm.add.rn.sat.f16(half %6, half %7)
  // CHECK-NEXT: %9 = fneg half %8
  // CHECK-NEXT: %10 = call half @llvm.nvvm.add.rn.ftz.sat.f16(half %8, half %9)
  // CHECK-NEXT: ret half %10
  // CHECK-NEXT: }
  %f1 = nvvm.subf %a, %b : f16
  %f2 = nvvm.subf %f1, %f1 {rnd = #nvvm.fp_rnd_mode<rn>} : f16
  %f3 = nvvm.subf %f2, %f2 {rnd = #nvvm.fp_rnd_mode<rn>, sat = #nvvm.sat_mode<sat>} : f16
  %f4 = nvvm.subf %f3, %f3 {rnd = #nvvm.fp_rnd_mode<rn>, sat = #nvvm.sat_mode<sat>, ftz=true} : f16
  llvm.return %f4 : f16
}

// bf16 - bf16 -> bf16
llvm.func @fsub_bf16_bf16(%a : bf16, %b : bf16) -> bf16 {
  // CHECK-LABEL: define bfloat @fsub_bf16_bf16(bfloat %0, bfloat %1) {
  // CHECK-NEXT: %3 = fneg bfloat %1
  // CHECK-NEXT: %4 = fadd bfloat %0, %3
  // CHECK-NEXT: %5 = fneg bfloat %4
  // CHECK-NEXT: %6 = fadd bfloat %4, %5
  // CHECK-NEXT: ret bfloat %6
  // CHECK-NEXT: }
  %f1 = nvvm.subf %a, %b : bf16
  %f2 = nvvm.subf %f1, %f1 {rnd = #nvvm.fp_rnd_mode<rn>} : bf16
  llvm.return %f2 : bf16
}

// f32 - f32 -> f32
llvm.func @fsub_f32_f32(%a : f32, %b : f32) -> f32 {
  // CHECK-LABEL: define float @fsub_f32_f32(float %0, float %1) {
  // CHECK-NEXT: %3 = fneg float %1
  // CHECK-NEXT: %4 = call float @llvm.nvvm.add.rn.f(float %0, float %3)
  // CHECK-NEXT: %5 = fneg float %4
  // CHECK-NEXT: %6 = call float @llvm.nvvm.add.rn.f(float %4, float %5)
  // CHECK-NEXT: %7 = fneg float %6
  // CHECK-NEXT: %8 = call float @llvm.nvvm.add.rn.sat.f(float %6, float %7)
  // CHECK-NEXT: %9 = fneg float %8
  // CHECK-NEXT: %10 = call float @llvm.nvvm.add.rn.ftz.f(float %8, float %9)
  // CHECK-NEXT: %11 = fneg float %10
  // CHECK-NEXT: %12 = call float @llvm.nvvm.add.rn.ftz.sat.f(float %10, float %11)
  // CHECK-NEXT: %13 = fneg float %12
  // CHECK-NEXT: %14 = call float @llvm.nvvm.add.rm.f(float %12, float %13)
  // CHECK-NEXT: %15 = fneg float %14
  // CHECK-NEXT: %16 = call float @llvm.nvvm.add.rm.sat.f(float %14, float %15)
  // CHECK-NEXT: %17 = fneg float %16
  // CHECK-NEXT: %18 = call float @llvm.nvvm.add.rm.ftz.f(float %16, float %17)
  // CHECK-NEXT: %19 = fneg float %18
  // CHECK-NEXT: %20 = call float @llvm.nvvm.add.rm.ftz.sat.f(float %18, float %19)
  // CHECK-NEXT: %21 = fneg float %20
  // CHECK-NEXT: %22 = call float @llvm.nvvm.add.rp.f(float %20, float %21)
  // CHECK-NEXT: %23 = fneg float %22
  // CHECK-NEXT: %24 = call float @llvm.nvvm.add.rp.sat.f(float %22, float %23)
  // CHECK-NEXT: %25 = fneg float %24
  // CHECK-NEXT: %26 = call float @llvm.nvvm.add.rp.ftz.f(float %24, float %25)
  // CHECK-NEXT: %27 = fneg float %26
  // CHECK-NEXT: %28 = call float @llvm.nvvm.add.rp.ftz.sat.f(float %26, float %27)
  // CHECK-NEXT: %29 = fneg float %28
  // CHECK-NEXT: %30 = call float @llvm.nvvm.add.rz.f(float %28, float %29)
  // CHECK-NEXT: %31 = fneg float %30
  // CHECK-NEXT: %32 = call float @llvm.nvvm.add.rz.sat.f(float %30, float %31)
  // CHECK-NEXT: %33 = fneg float %32
  // CHECK-NEXT: %34 = call float @llvm.nvvm.add.rz.ftz.f(float %32, float %33)
  // CHECK-NEXT: %35 = fneg float %34
  // CHECK-NEXT: %36 = call float @llvm.nvvm.add.rz.ftz.sat.f(float %34, float %35)
  // CHECK-NEXT: ret float %36
  // CHECK-NEXT: }
  %f1 = nvvm.subf %a, %b : f32
  %f2 = nvvm.subf %f1, %f1 {rnd = #nvvm.fp_rnd_mode<rn>} : f32
  %f3 = nvvm.subf %f2, %f2 {rnd = #nvvm.fp_rnd_mode<rn>, sat = #nvvm.sat_mode<sat>} : f32
  %f4 = nvvm.subf %f3, %f3 {rnd = #nvvm.fp_rnd_mode<rn>, ftz=true} : f32
  %f5 = nvvm.subf %f4, %f4 {rnd = #nvvm.fp_rnd_mode<rn>, sat = #nvvm.sat_mode<sat>, ftz=true} : f32
  %f6 = nvvm.subf %f5, %f5 {rnd = #nvvm.fp_rnd_mode<rm>} : f32
  %f7 = nvvm.subf %f6, %f6 {rnd = #nvvm.fp_rnd_mode<rm>, sat = #nvvm.sat_mode<sat>} : f32
  %f8 = nvvm.subf %f7, %f7 {rnd = #nvvm.fp_rnd_mode<rm>, ftz=true} : f32
  %f9 = nvvm.subf %f8, %f8 {rnd = #nvvm.fp_rnd_mode<rm>, sat = #nvvm.sat_mode<sat>, ftz=true} : f32
  %f10 = nvvm.subf %f9, %f9 {rnd = #nvvm.fp_rnd_mode<rp>} : f32
  %f11 = nvvm.subf %f10, %f10 {rnd = #nvvm.fp_rnd_mode<rp>, sat = #nvvm.sat_mode<sat>} : f32
  %f12 = nvvm.subf %f11, %f11 {rnd = #nvvm.fp_rnd_mode<rp>, ftz=true} : f32
  %f13 = nvvm.subf %f12, %f12 {rnd = #nvvm.fp_rnd_mode<rp>, sat = #nvvm.sat_mode<sat>, ftz=true} : f32
  %f14 = nvvm.subf %f13, %f13 {rnd = #nvvm.fp_rnd_mode<rz>} : f32
  %f15 = nvvm.subf %f14, %f14 {rnd = #nvvm.fp_rnd_mode<rz>, sat = #nvvm.sat_mode<sat>} : f32
  %f16 = nvvm.subf %f15, %f15 {rnd = #nvvm.fp_rnd_mode<rz>, ftz=true} : f32
  %f17 = nvvm.subf %f16, %f16 {rnd = #nvvm.fp_rnd_mode<rz>, sat = #nvvm.sat_mode<sat>, ftz=true} : f32
  llvm.return %f17 : f32
}

// f64 - f64 -> f64
llvm.func @fsub_f64_f64(%a : f64, %b : f64) -> f64 {
  // CHECK-LABEL: define double @fsub_f64_f64(double %0, double %1) {
  // CHECK-NEXT: %3 = fneg double %1
  // CHECK-NEXT: %4 = call double @llvm.nvvm.add.rn.d(double %0, double %3)
  // CHECK-NEXT: %5 = fneg double %4
  // CHECK-NEXT: %6 = call double @llvm.nvvm.add.rn.d(double %4, double %5)
  // CHECK-NEXT: %7 = fneg double %6
  // CHECK-NEXT: %8 = call double @llvm.nvvm.add.rm.d(double %6, double %7)
  // CHECK-NEXT: %9 = fneg double %8
  // CHECK-NEXT: %10 = call double @llvm.nvvm.add.rp.d(double %8, double %9)
  // CHECK-NEXT: %11 = fneg double %10
  // CHECK-NEXT: %12 = call double @llvm.nvvm.add.rz.d(double %10, double %11)
  // CHECK-NEXT: ret double %12
  // CHECK-NEXT: }
  %f1 = nvvm.subf %a, %b : f64
  %f2 = nvvm.subf %f1, %f1 {rnd = #nvvm.fp_rnd_mode<rn>} : f64
  %f3 = nvvm.subf %f2, %f2 {rnd = #nvvm.fp_rnd_mode<rm>} : f64
  %f4 = nvvm.subf %f3, %f3 {rnd = #nvvm.fp_rnd_mode<rp>} : f64
  %f5 = nvvm.subf %f4, %f4 {rnd = #nvvm.fp_rnd_mode<rz>} : f64
  llvm.return %f5 : f64
}
