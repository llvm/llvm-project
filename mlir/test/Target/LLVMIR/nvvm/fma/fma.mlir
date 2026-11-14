// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @fma_f16(%a: f16, %b: f16, %c: f16) -> f16 {
  // CHECK-LABEL: define half @fma_f16(half %0, half %1, half %2) {
  // CHECK-NEXT: %4 = call half @llvm.nvvm.fma.f16(half %0, half %1, half %2, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %5 = call half @llvm.nvvm.fma.ftz.f16(half %0, half %1, half %4, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %6 = call half @llvm.nvvm.fma.sat.f16(half %0, half %1, half %5, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %7 = call half @llvm.nvvm.fma.ftz.sat.f16(half %0, half %1, half %6, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %8 = call half @llvm.nvvm.fma.relu.f16(half %0, half %1, half %7, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %9 = call half @llvm.nvvm.fma.ftz.relu.f16(half %0, half %1, half %8, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %10 = call half @llvm.nvvm.fma.rn.oob.f16(half %0, half %1, half %9)
  // CHECK-NEXT: %11 = call half @llvm.nvvm.fma.rn.oob.relu.f16(half %0, half %1, half %10)
  // CHECK-NEXT: ret half %11
  // CHECK-NEXT: }
  %f0 = nvvm.fma %a, %b, %c, rnd = <rn> : f16
  %f1 = nvvm.fma %a, %b, %f0, rnd = <rn> ftz = true : f16
  %f2 = nvvm.fma %a, %b, %f1, rnd = <rn> sat = <sat> : f16
  %f3 = nvvm.fma %a, %b, %f2, rnd = <rn> sat = <sat> ftz = true : f16
  %f4 = nvvm.fma %a, %b, %f3, rnd = <rn> relu = true : f16
  %f5 = nvvm.fma %a, %b, %f4, rnd = <rn> ftz = true relu = true : f16
  %f6 = nvvm.fma %a, %b, %f5, rnd = <rn> oob = true : f16
  %f7 = nvvm.fma %a, %b, %f6, rnd = <rn> relu = true oob = true : f16
  llvm.return %f7 : f16
}

llvm.func @fma_bf16(%a: bf16, %b: bf16, %c: bf16) -> bf16 {
  // CHECK-LABEL: define bfloat @fma_bf16(bfloat %0, bfloat %1, bfloat %2) {
  // CHECK-NEXT: %4 = call bfloat @llvm.nvvm.fma.bf16(bfloat %0, bfloat %1, bfloat %2, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %5 = call bfloat @llvm.nvvm.fma.relu.bf16(bfloat %0, bfloat %1, bfloat %4, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %6 = call bfloat @llvm.nvvm.fma.rn.oob.bf16(bfloat %0, bfloat %1, bfloat %5)
  // CHECK-NEXT: %7 = call bfloat @llvm.nvvm.fma.rn.oob.relu.bf16(bfloat %0, bfloat %1, bfloat %6)
  // CHECK-NEXT: ret bfloat %7
  // CHECK-NEXT: }
  %f0 = nvvm.fma %a, %b, %c, rnd = <rn> : bf16
  %f1 = nvvm.fma %a, %b, %f0, rnd = <rn> relu = true : bf16
  %f2 = nvvm.fma %a, %b, %f1, rnd = <rn> oob = true : bf16
  %f3 = nvvm.fma %a, %b, %f2, rnd = <rn> relu = true oob = true : bf16
  llvm.return %f3 : bf16
}

llvm.func @fma_f32_rn(%a: f32, %b: f32, %c: f32) -> f32 {
  // CHECK-LABEL: define float @fma_f32_rn(float %0, float %1, float %2) {
  // CHECK-NEXT: %4 = call float @llvm.nvvm.fma.f32(float %0, float %1, float %2, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %5 = call float @llvm.nvvm.fma.ftz.f32(float %0, float %1, float %4, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %6 = call float @llvm.nvvm.fma.sat.f32(float %0, float %1, float %5, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %7 = call float @llvm.nvvm.fma.ftz.sat.f32(float %0, float %1, float %6, /* rnd=rn */ i32 1)
  // CHECK-NEXT: ret float %7
  // CHECK-NEXT: }
  %f0 = nvvm.fma %a, %b, %c, rnd = <rn> : f32
  %f1 = nvvm.fma %a, %b, %f0, rnd = <rn> ftz = true : f32
  %f2 = nvvm.fma %a, %b, %f1, rnd = <rn> sat = <sat> : f32
  %f3 = nvvm.fma %a, %b, %f2, rnd = <rn> sat = <sat> ftz = true : f32
  llvm.return %f3 : f32
}

llvm.func @fma_f32_rm(%a: f32, %b: f32, %c: f32) -> f32 {
  // CHECK-LABEL: define float @fma_f32_rm(float %0, float %1, float %2) {
  // CHECK-NEXT: %4 = call float @llvm.nvvm.fma.f32(float %0, float %1, float %2, /* rnd=rm */ i32 3)
  // CHECK-NEXT: %5 = call float @llvm.nvvm.fma.ftz.f32(float %0, float %1, float %4, /* rnd=rm */ i32 3)
  // CHECK-NEXT: %6 = call float @llvm.nvvm.fma.sat.f32(float %0, float %1, float %5, /* rnd=rm */ i32 3)
  // CHECK-NEXT: %7 = call float @llvm.nvvm.fma.ftz.sat.f32(float %0, float %1, float %6, /* rnd=rm */ i32 3)
  // CHECK-NEXT: ret float %7
  // CHECK-NEXT: }
  %f0 = nvvm.fma %a, %b, %c, rnd = <rm> : f32
  %f1 = nvvm.fma %a, %b, %f0, rnd = <rm> ftz = true : f32
  %f2 = nvvm.fma %a, %b, %f1, rnd = <rm> sat = <sat> : f32
  %f3 = nvvm.fma %a, %b, %f2, rnd = <rm> sat = <sat> ftz = true : f32
  llvm.return %f3 : f32
}

llvm.func @fma_f32_rp(%a: f32, %b: f32, %c: f32) -> f32 {
  // CHECK-LABEL: define float @fma_f32_rp(float %0, float %1, float %2) {
  // CHECK-NEXT: %4 = call float @llvm.nvvm.fma.f32(float %0, float %1, float %2, /* rnd=rp */ i32 2)
  // CHECK-NEXT: %5 = call float @llvm.nvvm.fma.ftz.f32(float %0, float %1, float %4, /* rnd=rp */ i32 2)
  // CHECK-NEXT: %6 = call float @llvm.nvvm.fma.sat.f32(float %0, float %1, float %5, /* rnd=rp */ i32 2)
  // CHECK-NEXT: %7 = call float @llvm.nvvm.fma.ftz.sat.f32(float %0, float %1, float %6, /* rnd=rp */ i32 2)
  // CHECK-NEXT: ret float %7
  // CHECK-NEXT: }
  %f0 = nvvm.fma %a, %b, %c, rnd = <rp> : f32
  %f1 = nvvm.fma %a, %b, %f0, rnd = <rp> ftz = true : f32
  %f2 = nvvm.fma %a, %b, %f1, rnd = <rp> sat = <sat> : f32
  %f3 = nvvm.fma %a, %b, %f2, rnd = <rp> sat = <sat> ftz = true : f32
  llvm.return %f3 : f32
}

llvm.func @fma_f32_rz(%a: f32, %b: f32, %c: f32) -> f32 {
  // CHECK-LABEL: define float @fma_f32_rz(float %0, float %1, float %2) {
  // CHECK-NEXT: %4 = call float @llvm.nvvm.fma.f32(float %0, float %1, float %2, /* rnd=rz */ i32 0)
  // CHECK-NEXT: %5 = call float @llvm.nvvm.fma.ftz.f32(float %0, float %1, float %4, /* rnd=rz */ i32 0)
  // CHECK-NEXT: %6 = call float @llvm.nvvm.fma.sat.f32(float %0, float %1, float %5, /* rnd=rz */ i32 0)
  // CHECK-NEXT: %7 = call float @llvm.nvvm.fma.ftz.sat.f32(float %0, float %1, float %6, /* rnd=rz */ i32 0)
  // CHECK-NEXT: ret float %7
  // CHECK-NEXT: }
  %f0 = nvvm.fma %a, %b, %c, rnd = <rz> : f32
  %f1 = nvvm.fma %a, %b, %f0, rnd = <rz> ftz = true : f32
  %f2 = nvvm.fma %a, %b, %f1, rnd = <rz> sat = <sat> : f32
  %f3 = nvvm.fma %a, %b, %f2, rnd = <rz> sat = <sat> ftz = true : f32
  llvm.return %f3 : f32
}

llvm.func @fma_f64(%a: f64, %b: f64, %c: f64) -> f64 {
  // CHECK-LABEL: define double @fma_f64(double %0, double %1, double %2) {
  // CHECK-NEXT: %4 = call double @llvm.nvvm.fma.f64(double %0, double %1, double %2, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %5 = call double @llvm.nvvm.fma.f64(double %0, double %1, double %4, /* rnd=rm */ i32 3)
  // CHECK-NEXT: %6 = call double @llvm.nvvm.fma.f64(double %0, double %1, double %5, /* rnd=rp */ i32 2)
  // CHECK-NEXT: %7 = call double @llvm.nvvm.fma.f64(double %0, double %1, double %6, /* rnd=rz */ i32 0)
  // CHECK-NEXT: ret double %7
  // CHECK-NEXT: }
  %f0 = nvvm.fma %a, %b, %c, rnd = <rn> : f64
  %f1 = nvvm.fma %a, %b, %f0, rnd = <rm> : f64
  %f2 = nvvm.fma %a, %b, %f1, rnd = <rp> : f64
  %f3 = nvvm.fma %a, %b, %f2, rnd = <rz> : f64
  llvm.return %f3 : f64
}
