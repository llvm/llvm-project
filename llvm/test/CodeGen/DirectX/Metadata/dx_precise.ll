; RUN: llc %s --filetype=asm -o - 2>&1 | FileCheck %s
target triple = "dxil-pc-shadermodel6.6-compute"


; CHECK-LABEL: define void @unary_f32 
; CHECK: call float @dx.op.unary.f32(i32 7,
; CHECK-SAME: !dx.precise ![[SM:[0-9]+]]
; CHECK-COUNT-17: !dx.precise ![[SM]]
define void @unary_f32(float %p) {
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
  %14 = call float @llvm.sqrt.f32(float %p)
  %15 = call float @llvm.roundeven.f32(float %p)
  %16 = call float @llvm.floor.f32(float %p)
  %17 = call float @llvm.ceil.f32(float %p)
  %18 = call float @llvm.trunc.f32(float %p)
  ret void
}

; CHECK-LABEL: define void @unary_f16 
; CHECK: call half @dx.op.unary.f16(i32 7,
; CHECK-SAME: !dx.precise ![[SM]]
; CHECK-COUNT-17: !dx.precise ![[SM]]
define void @unary_f16(half %p) {
entry:
  %1 = call half @llvm.dx.saturate.f16(half %p)
  %2 = call half @llvm.cos.f16(half %p)
  %3 = call half @llvm.sin.f16(half %p)
  %4 = call half @llvm.tan.f16(half %p)
  %5 = call half @llvm.acos.f16(half %p)
  %6 = call half @llvm.asin.f16(half %p)
  %7 = call half @llvm.atan.f16(half %p)
  %8 = call half @llvm.cosh.f16(half %p)
  %9 = call half @llvm.sinh.f16(half %p)
  %10 = call half @llvm.tanh.f16(half %p)
  %11 = call half @llvm.exp2.f16(half %p)
  %12 = call half @llvm.dx.frac.f16(half %p)
  %13 = call half @llvm.log2.f16(half %p)
  %14 = call half @llvm.sqrt.f16(half %p)
  %15 = call half @llvm.roundeven.f16(half %p)
  %16 = call half @llvm.floor.f16(half %p)
  %17 = call half @llvm.ceil.f16(half %p)
  %18 = call half @llvm.trunc.f16(half %p)
  ret void
}

; CHECK-LABEL: define void @unary_f64 
; CHECK: call double @dx.op.unary.f64(i32 7,
; CHECK-SAME: !dx.precise ![[SM]]
define void @unary_f64(double %p) {
entry:
  %1 = call double @llvm.dx.saturate.f64(double %p)
  ret void
}

; CHECK-LABEL: define void @binary_f32 
; CHECK: call float @dx.op.binary.f32(i32 35,
; CHECK-SAME: !dx.precise ![[SM]]
; CHECK-COUNT-1: !dx.precise ![[SM]]
define void @binary_f32(float %p1, float %p2) {
entry:
  %20 = call float @llvm.maxnum.f32(float %p1, float %p2)
  %21 = call float @llvm.minnum.f32(float %p1, float %p2)
  ret void
}

; CHECK-LABEL: define void @binary_f16 
; CHECK: call half @dx.op.binary.f16(i32 35,
; CHECK-SAME: !dx.precise ![[SM]]
; CHECK-COUNT-1: !dx.precise ![[SM]]
define void @binary_f16(half %p1, half %p2) {
entry:
  %20 = call half @llvm.maxnum.f16(half %p1, half %p2)
  %21 = call half @llvm.minnum.f16(half %p1, half %p2)
  ret void
}

; CHECK-LABEL: define void @binary_f64 
; CHECK: call double @dx.op.binary.f64(i32 35,
; CHECK-SAME: !dx.precise ![[SM]]
; CHECK-COUNT-1: !dx.precise ![[SM]]
define void @binary_f64(double %p1, double %p2) {
entry:
  %20 = call double @llvm.maxnum.f64(double %p1, double %p2)
  %21 = call double @llvm.minnum.f64(double %p1, double %p2)
  ret void
}

; CHECK-LABEL: define void @tertiary 
; CHECK: call float @dx.op.tertiary.f32(i32 46,
; CHECK-SAME: !dx.precise ![[SM]]
define void @tertiary(float %p1, float %p2, float %p3) {
entry:
  %22 = call float @llvm.fmuladd.f32(float %p1, float %p2, float %p3)
  ret void
}

; CHECK-LABEL: define void @fma 
; CHECK: call double @dx.op.tertiary.f64(i32 47,
; CHECK-SAME: !dx.precise ![[SM]]
define void @fma(double %p1, double %p2, double %p3) {
entry:
  %23 = call double @llvm.fma.f64(double %p1, double %p2, double %p3)
  ret void
}

; CHECK-LABEL: define void @dot2_f32 
; CHECK: call float @dx.op.dot2.f32(i32 54,
; CHECK-SAME: !dx.precise ![[SM]]
define void @dot2_f32(<2 x float> %a, <2 x float> %b) {
entry:
  %24 = call float @llvm.dx.fdot.v2f32(<2 x float> %a, <2 x float> %b)
  ret void
}

; CHECK-LABEL: define void @dot2_f16 
; CHECK: call half @dx.op.dot2.f16(i32 54,
; CHECK-SAME: !dx.precise ![[SM]]
define void @dot2_f16(<2 x half> %a, <2 x half> %b) {
entry:
  %24 = call half @llvm.dx.fdot.v2f16(<2 x half> %a, <2 x half> %b)
  ret void
}

; CHECK-LABEL: define void @dot3_f32 
; CHECK: call float @dx.op.dot3.f32(i32 55,
; CHECK-SAME: !dx.precise ![[SM]]
define void @dot3_f32(<3 x float> %a, <3 x float> %b) {
entry:
  %25 = call float @llvm.dx.fdot.v3f32(<3 x float> %a, <3 x float> %b)
  ret void
}

; CHECK-LABEL: define void @dot3_f16 
; CHECK: call half @dx.op.dot3.f16(i32 55,
; CHECK-SAME: !dx.precise ![[SM]]
define void @dot3_f16(<3 x half> %a, <3 x half> %b) {
entry:
  %25 = call half @llvm.dx.fdot.v3f16(<3 x half> %a, <3 x half> %b)
  ret void
}

; CHECK-LABEL: define void @dot4_f32 
; CHECK: call float @dx.op.dot4.f32(i32 56,
; CHECK-SAME: !dx.precise ![[SM]]
define void @dot4_f32(<4 x float> %a, <4 x float> %b) {
entry:
  %26 = call float @llvm.dx.fdot.v4f32(<4 x float> %a, <4 x float> %b)
  ret void
}

; CHECK-LABEL: define void @dot4_f16 
; CHECK: call half @dx.op.dot4.f16(i32 56,
; CHECK-SAME: !dx.precise ![[SM]]
define void @dot4_f16(<4 x half> %a, <4 x half> %b) {
entry:
  %26 = call half @llvm.dx.fdot.v4f16(<4 x half> %a, <4 x half> %b)
  ret void
}

; CHECK-LABEL: define void @wave_rla_f32
; CHECK: call float @dx.op.waveReadLaneAt.f32(i32 117,
; CHECK-SAME: !dx.precise ![[SM]]
define void @wave_rla_f32(float %expr, i32 %idx) {
entry:
  %27 = call float @llvm.dx.wave.readlane(float %expr, i32 %idx)
  ret void
}

; CHECK-LABEL: define void @wave_rla_f16
; CHECK: call half @dx.op.waveReadLaneAt.f16(i32 117,
; CHECK-SAME: !dx.precise ![[SM]]
define void @wave_rla_f16(half %expr, i32 %idx) {
entry:
  %27 = call half @llvm.dx.wave.readlane(half %expr, i32 %idx)
  ret void
}

; CHECK-LABEL: define void @wave_rla_i32
; CHECK-NOT: @dx.op.waveReadLaneAt.i32(i32 117, i32 %expr, i32 %idx), !dx.precise ![[SM]]
define void @wave_rla_i32(i32 %expr, i32 %idx) {
entry:
  %27 = call i32 @llvm.dx.wave.readlane(i32 %expr, i32 %idx)
  ret void
}

; CHECK-LABEL: define void @not_fast 
; CHECK: call float @dx.op.unary.f32(i32 7,
; CHECK-SAME: !dx.precise ![[SM:[0-9]+]]
; CHECK-COUNT-17: !dx.precise ![[SM]]
define void @not_fast(float %p) {
entry:
  %p_mod = fadd float %p, 2.0
  %1 = call float @llvm.dx.saturate.f32(float %p_mod)
  %2 = call float @llvm.cos.f32(float %p_mod)
  %3 = call float @llvm.sin.f32(float %p_mod)
  %4 = call float @llvm.tan.f32(float %p_mod)
  %5 = call float @llvm.acos.f32(float %p_mod)
  %6 = call float @llvm.asin.f32(float %p_mod)
  %7 = call float @llvm.atan.f32(float %p_mod)
  %8 = call float @llvm.cosh.f32(float %p_mod)
  %9 = call float @llvm.sinh.f32(float %p_mod)
  %10 = call float @llvm.tanh.f32(float %p_mod)
  %11 = call float @llvm.exp2.f32(float %p_mod)
  %12 = call float @llvm.dx.frac.f32(float %p_mod)
  %13 = call float @llvm.log2.f32(float %p_mod)
  %14 = call float @llvm.sqrt.f32(float %p_mod)
  %15 = call float @llvm.roundeven.f32(float %p_mod)
  %16 = call float @llvm.floor.f32(float %p_mod)
  %17 = call float @llvm.ceil.f32(float %p_mod)
  %18 = call float @llvm.trunc.f32(float %p_mod)
  ret void
}

; By this point all !dx.precise anotation should be matched
; CHECK-NOT: !dx.precise ![[SM]]

define void @fast(float %p) {
entry:
  %1 = fadd float %p, 2.0
  %2 = call fast float @llvm.dx.saturate.f32(float %1)
  ret void
}
; CHECK: ![[SM]] = !{i32 1}
