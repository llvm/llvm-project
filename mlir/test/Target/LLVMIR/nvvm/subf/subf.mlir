// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// f16 - f16 -> f16
llvm.func @fsub_f16_f16(%a : f16, %b : f16) -> f16 {
  // CHECK-LABEL: define half @fsub_f16_f16(half %0, half %1) {
  // CHECK-NEXT: %3 = fneg half %1
  // CHECK-NEXT: %4 = call half @llvm.nvvm.fadd.f16(half %0, half %3, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %5 = fneg half %4
  // CHECK-NEXT: %6 = call half @llvm.nvvm.fadd.f16(half %4, half %5, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %7 = fneg half %6
  // CHECK-NEXT: %8 = call half @llvm.nvvm.fadd.ftz.f16(half %6, half %7, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %9 = fneg half %8
  // CHECK-NEXT: %10 = call half @llvm.nvvm.fadd.sat.f16(half %8, half %9, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %11 = fneg half %10
  // CHECK-NEXT: %12 = call half @llvm.nvvm.fadd.ftz.sat.f16(half %10, half %11, /* rnd=rn */ i32 1)
  // CHECK-NEXT: ret half %12
  // CHECK-NEXT: }
  %f1 = nvvm.subf %a, %b : f16
  %f2 = nvvm.subf %f1, %f1 rnd = <rn> : f16
  %f3 = nvvm.subf %f2, %f2 rnd = <rn> ftz = true : f16
  %f4 = nvvm.subf %f3, %f3 rnd = <rn> sat = <sat> : f16
  %f5 = nvvm.subf %f4, %f4 rnd = <rn> sat = <sat> ftz = true : f16
  llvm.return %f5 : f16
}

// bf16 - bf16 -> bf16
llvm.func @fsub_bf16_bf16(%a : bf16, %b : bf16) -> bf16 {
  // CHECK-LABEL: define bfloat @fsub_bf16_bf16(bfloat %0, bfloat %1) {
  // CHECK-NEXT: %3 = fneg bfloat %1
  // CHECK-NEXT: %4 = call bfloat @llvm.nvvm.fadd.bf16(bfloat %0, bfloat %3, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %5 = fneg bfloat %4
  // CHECK-NEXT: %6 = call bfloat @llvm.nvvm.fadd.bf16(bfloat %4, bfloat %5, /* rnd=rn */ i32 1)
  // CHECK-NEXT: ret bfloat %6
  // CHECK-NEXT: }
  %f1 = nvvm.subf %a, %b : bf16
  %f2 = nvvm.subf %f1, %f1 rnd = <rn> : bf16
  llvm.return %f2 : bf16
}

// f32 - f32 -> f32
llvm.func @fsub_f32_f32(%a : f32, %b : f32) -> f32 {
  // CHECK-LABEL: define float @fsub_f32_f32(float %0, float %1) {
  // CHECK-NEXT: %3 = fneg float %1
  // CHECK-NEXT: %4 = call float @llvm.nvvm.fadd.f32(float %0, float %3, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %5 = fneg float %4
  // CHECK-NEXT: %6 = call float @llvm.nvvm.fadd.f32(float %4, float %5, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %7 = fneg float %6
  // CHECK-NEXT: %8 = call float @llvm.nvvm.fadd.sat.f32(float %6, float %7, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %9 = fneg float %8
  // CHECK-NEXT: %10 = call float @llvm.nvvm.fadd.ftz.f32(float %8, float %9, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %11 = fneg float %10
  // CHECK-NEXT: %12 = call float @llvm.nvvm.fadd.ftz.sat.f32(float %10, float %11, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %13 = fneg float %12
  // CHECK-NEXT: %14 = call float @llvm.nvvm.fadd.f32(float %12, float %13, /* rnd=rm */ i32 3)
  // CHECK-NEXT: %15 = fneg float %14
  // CHECK-NEXT: %16 = call float @llvm.nvvm.fadd.sat.f32(float %14, float %15, /* rnd=rm */ i32 3)
  // CHECK-NEXT: %17 = fneg float %16
  // CHECK-NEXT: %18 = call float @llvm.nvvm.fadd.ftz.f32(float %16, float %17, /* rnd=rm */ i32 3)
  // CHECK-NEXT: %19 = fneg float %18
  // CHECK-NEXT: %20 = call float @llvm.nvvm.fadd.ftz.sat.f32(float %18, float %19, /* rnd=rm */ i32 3)
  // CHECK-NEXT: %21 = fneg float %20
  // CHECK-NEXT: %22 = call float @llvm.nvvm.fadd.f32(float %20, float %21, /* rnd=rp */ i32 2)
  // CHECK-NEXT: %23 = fneg float %22
  // CHECK-NEXT: %24 = call float @llvm.nvvm.fadd.sat.f32(float %22, float %23, /* rnd=rp */ i32 2)
  // CHECK-NEXT: %25 = fneg float %24
  // CHECK-NEXT: %26 = call float @llvm.nvvm.fadd.ftz.f32(float %24, float %25, /* rnd=rp */ i32 2)
  // CHECK-NEXT: %27 = fneg float %26
  // CHECK-NEXT: %28 = call float @llvm.nvvm.fadd.ftz.sat.f32(float %26, float %27, /* rnd=rp */ i32 2)
  // CHECK-NEXT: %29 = fneg float %28
  // CHECK-NEXT: %30 = call float @llvm.nvvm.fadd.f32(float %28, float %29, /* rnd=rz */ i32 0)
  // CHECK-NEXT: %31 = fneg float %30
  // CHECK-NEXT: %32 = call float @llvm.nvvm.fadd.sat.f32(float %30, float %31, /* rnd=rz */ i32 0)
  // CHECK-NEXT: %33 = fneg float %32
  // CHECK-NEXT: %34 = call float @llvm.nvvm.fadd.ftz.f32(float %32, float %33, /* rnd=rz */ i32 0)
  // CHECK-NEXT: %35 = fneg float %34
  // CHECK-NEXT: %36 = call float @llvm.nvvm.fadd.ftz.sat.f32(float %34, float %35, /* rnd=rz */ i32 0)
  // CHECK-NEXT: ret float %36
  // CHECK-NEXT: }
  %f1 = nvvm.subf %a, %b : f32
  %f2 = nvvm.subf %f1, %f1 rnd = <rn> : f32
  %f3 = nvvm.subf %f2, %f2 rnd = <rn> sat = <sat> : f32
  %f4 = nvvm.subf %f3, %f3 rnd = <rn> ftz = true : f32
  %f5 = nvvm.subf %f4, %f4 rnd = <rn> sat = <sat> ftz = true : f32
  %f6 = nvvm.subf %f5, %f5 rnd = <rm> : f32
  %f7 = nvvm.subf %f6, %f6 rnd = <rm> sat = <sat> : f32
  %f8 = nvvm.subf %f7, %f7 rnd = <rm> ftz = true : f32
  %f9 = nvvm.subf %f8, %f8 rnd = <rm> sat = <sat> ftz = true : f32
  %f10 = nvvm.subf %f9, %f9 rnd = <rp> : f32
  %f11 = nvvm.subf %f10, %f10 rnd = <rp> sat = <sat> : f32
  %f12 = nvvm.subf %f11, %f11 rnd = <rp> ftz = true : f32
  %f13 = nvvm.subf %f12, %f12 rnd = <rp> sat = <sat> ftz = true : f32
  %f14 = nvvm.subf %f13, %f13 rnd = <rz> : f32
  %f15 = nvvm.subf %f14, %f14 rnd = <rz> sat = <sat> : f32
  %f16 = nvvm.subf %f15, %f15 rnd = <rz> ftz = true : f32
  %f17 = nvvm.subf %f16, %f16 rnd = <rz> sat = <sat> ftz = true : f32
  llvm.return %f17 : f32
}

// f64 - f64 -> f64
llvm.func @fsub_f64_f64(%a : f64, %b : f64) -> f64 {
  // CHECK-LABEL: define double @fsub_f64_f64(double %0, double %1) {
  // CHECK-NEXT: %3 = fneg double %1
  // CHECK-NEXT: %4 = call double @llvm.nvvm.fadd.f64(double %0, double %3, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %5 = fneg double %4
  // CHECK-NEXT: %6 = call double @llvm.nvvm.fadd.f64(double %4, double %5, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %7 = fneg double %6
  // CHECK-NEXT: %8 = call double @llvm.nvvm.fadd.f64(double %6, double %7, /* rnd=rm */ i32 3)
  // CHECK-NEXT: %9 = fneg double %8
  // CHECK-NEXT: %10 = call double @llvm.nvvm.fadd.f64(double %8, double %9, /* rnd=rp */ i32 2)
  // CHECK-NEXT: %11 = fneg double %10
  // CHECK-NEXT: %12 = call double @llvm.nvvm.fadd.f64(double %10, double %11, /* rnd=rz */ i32 0)
  // CHECK-NEXT: ret double %12
  // CHECK-NEXT: }
  %f1 = nvvm.subf %a, %b : f64
  %f2 = nvvm.subf %f1, %f1 rnd = <rn> : f64
  %f3 = nvvm.subf %f2, %f2 rnd = <rm> : f64
  %f4 = nvvm.subf %f3, %f3 rnd = <rp> : f64
  %f5 = nvvm.subf %f4, %f4 rnd = <rz> : f64
  llvm.return %f5 : f64
}
