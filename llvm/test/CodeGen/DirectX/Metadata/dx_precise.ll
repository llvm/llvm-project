; RUN: llc %s -enable-precise --filetype=asm -o - < %s 2>&1 | FileCheck %s --check-prefixes=ENABLED
; RUN: llc %s --filetype=asm -o - < %s 2>&1 | FileCheck %s --check-prefixes=DISABLED



target triple = "dxil-pc-shadermodel6.6-compute"

; ENABLED: call float @dx.op.unary.f32(i32 7,
; ENABLED-SAME: !dx.precise ![[SM:[0-9]+]]
; ENABLED-COUNT-26: !dx.precise ![[SM]]
; ENABLED-NOT: !dx.precise ![[SM]]
; ENABLED: ![[SM]] = !{i32 1}

; DISABLED-NOT: !dx.precise ![[SM:[0-9]+]]
define void @unary(float %p) {
entry:
  %1 = call float @llvm.dx.saturate.f32(float %p)
  %2 = call float @llvm.cos.f32(float %p)
  %3 = call float @llvm.sin.f32(float %p)
  %4 = call float @llvm.tan.f32(float %p)
  %5 = call float @llvm.acos.f32(float %p)
  %6 = call float @llvm.asin.f32(float %p)
  %7 = call float @llvm.atan.f32(float %p)
  %8 = call float @llvm.cosh.f32(float %p)
  %9 = call float @llvm.sinh.f32(float %p)
  %10 = call float @llvm.tanh.f32(float %p)
  %11 = call float @llvm.exp2.f32(float %p)
  %12 = call float @llvm.dx.frac.f32(float %p)
  %13 = call float @llvm.log2.f32(float %p)
  %14 = call float @llvm.log2.f32(float %p)
  %15 = call float @llvm.sqrt.f32(float %p)
  %16 = call float @llvm.roundeven.f32(float %p)
  %17 = call float @llvm.floor.f32(float %p)
  %18 = call float @llvm.ceil.f32(float %p)
  %19 = call float @llvm.trunc.f32(float %p)
  ret void
}

define void @binary(float %p1, float %p2) {
entry:
  %20 = call float @llvm.maxnum.f32(float %p1, float %p2)
  %21 = call float @llvm.minnum.f32(float %p1, float %p2)
  ret void
}

define void @tertiary(float %p1, float %p2, float %p3) {
entry:
  %22 = call float @llvm.fmuladd.f32(float %p1, float %p2, float %p3)
  ret void
}

define void @fma(double %p1, double %p2, double %p3) {
entry:
  %23 = call double @llvm.fma.f64(double %p1, double %p2, double %p3)
  ret void
}

define void @dot2(<2 x float> %a, <2 x float> %b) {
entry:
  %24 = call float @llvm.dx.fdot.v2f32(<2 x float> %a, <2 x float> %b)
  ret void
}

define void @dot3(<3 x float> %a, <3 x float> %b) {
entry:
  %25 = call float @llvm.dx.fdot.v3f32(<3 x float> %a, <3 x float> %b)
  ret void
}

define void @dot4(<4 x float> %a, <4 x float> %b) {
entry:
  %26 = call float @llvm.dx.fdot.v4f32(<4 x float> %a, <4 x float> %b)
  ret void
}

define void @wave_rla(float %expr, i32 %idx) {
entry:
  %27 = call float @llvm.dx.wave.readlane(float %expr, i32 %idx)
  ret void
}
