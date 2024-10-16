; RUN: opt -S -dxil-intrinsic-expansion -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-compute %s | FileCheck %s

; Test that for scalar values, WaveReadLaneAt maps down to the DirectX op

define noundef half @wave_active_sum_half(half noundef %expr) {
entry:
; CHECK: call half @dx.op.waveActiveOp.f16(i32 119, half %expr, i8 0, i8 0)
  %ret = call half @llvm.dx.wave.active.sum.f16(half %expr)
  ret half %ret
}

define noundef float @wave_active_sum_float(float noundef %expr) {
entry:
; CHECK: call float @dx.op.waveActiveOp.f32(i32 119, float %expr, i8 0, i8 0)
  %ret = call float @llvm.dx.wave.active.sum(float %expr)
  ret float %ret
}

define noundef double @wave_active_sum_double(double noundef %expr) {
entry:
; CHECK: call double @dx.op.waveActiveOp.f64(i32 119, double %expr, i8 0, i8 0)
  %ret = call double @llvm.dx.wave.active.sum(double %expr)
  ret double %ret
}

define noundef i16 @wave_active_sum_i16(i16 noundef %expr) {
entry:
; CHECK: call i16 @dx.op.waveActiveOp.i16(i32 119, i16 %expr, i8 0, i8 0)
  %ret = call i16 @llvm.dx.wave.active.sum.i16(i16 %expr)
  ret i16 %ret
}

define noundef i32 @wave_active_sum_i32(i32 noundef %expr) {
entry:
; CHECK: call i32 @dx.op.waveActiveOp.i32(i32 119, i32 %expr, i8 0, i8 0)
  %ret = call i32 @llvm.dx.wave.active.sum.i32(i32 %expr)
  ret i32 %ret
}

define noundef i64 @wave_active_sum_i64(i64 noundef %expr) {
entry:
; CHECK: call i64 @dx.op.waveActiveOp.i64(i32 119, i64 %expr, i8 0, i8 0)
  %ret = call i64 @llvm.dx.wave.active.sum.i64(i64 %expr)
  ret i64 %ret
}

define noundef i16 @wave_active_usum_i16(i16 noundef %expr) {
entry:
; CHECK: call i16 @dx.op.waveActiveOp.i16(i32 119, i16 %expr, i8 0, i8 1)
  %ret = call i16 @llvm.dx.wave.active.usum.i16(i16 %expr)
  ret i16 %ret
}

define noundef i32 @wave_active_usum_i32(i32 noundef %expr) {
entry:
; CHECK: call i32 @dx.op.waveActiveOp.i32(i32 119, i32 %expr, i8 0, i8 1)
  %ret = call i32 @llvm.dx.wave.active.usum.i32(i32 %expr)
  ret i32 %ret
}

define noundef i64 @wave_active_usum_i64(i64 noundef %expr) {
entry:
; CHECK: call i64 @dx.op.waveActiveOp.i64(i32 119, i64 %expr, i8 0, i8 1)
  %ret = call i64 @llvm.dx.wave.active.usum.i64(i64 %expr)
  ret i64 %ret
}

declare half @llvm.dx.wave.active.sum.f16(half)
declare float @llvm.dx.wave.active.sum.f32(float)
declare double @llvm.dx.wave.active.sum.f64(double)

declare i16 @llvm.dx.wave.active.sum.i16(i16)
declare i32 @llvm.dx.wave.active.sum.i32(i32)
declare i64 @llvm.dx.wave.active.sum.i64(i64)

declare i16 @llvm.dx.wave.active.usum.i16(i16)
declare i32 @llvm.dx.wave.active.usum.i32(i32)
declare i64 @llvm.dx.wave.active.usum.i64(i64)
