; RUN: opt -S -scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-compute %s | FileCheck %s

; Test that WaveReadLaneFirst maps down to the DirectX op.

define noundef half @wave_readlane_first_half(half noundef %expr) {
entry:
; CHECK: call half @dx.op.waveReadLaneFirst.f16(i32 118, half %expr)
  %ret = call half @llvm.dx.wave.readlane.first.f16(half %expr)
  ret half %ret
}

define noundef float @wave_readlane_first_float(float noundef %expr) {
entry:
; CHECK: call float @dx.op.waveReadLaneFirst.f32(i32 118, float %expr)
  %ret = call float @llvm.dx.wave.readlane.first.f32(float %expr)
  ret float %ret
}

define noundef double @wave_readlane_first_double(double noundef %expr) {
entry:
; CHECK: call double @dx.op.waveReadLaneFirst.f64(i32 118, double %expr)
  %ret = call double @llvm.dx.wave.readlane.first.f64(double %expr)
  ret double %ret
}

define noundef i1 @wave_readlane_first_i1(i1 noundef %expr) {
entry:
; CHECK: call i1 @dx.op.waveReadLaneFirst.i1(i32 118, i1 %expr)
  %ret = call i1 @llvm.dx.wave.readlane.first.i1(i1 %expr)
  ret i1 %ret
}

define noundef i16 @wave_readlane_first_i16(i16 noundef %expr) {
entry:
; CHECK: call i16 @dx.op.waveReadLaneFirst.i16(i32 118, i16 %expr)
  %ret = call i16 @llvm.dx.wave.readlane.first.i16(i16 %expr)
  ret i16 %ret
}

define noundef i32 @wave_readlane_first_i32(i32 noundef %expr) {
entry:
; CHECK: call i32 @dx.op.waveReadLaneFirst.i32(i32 118, i32 %expr)
  %ret = call i32 @llvm.dx.wave.readlane.first.i32(i32 %expr)
  ret i32 %ret
}

define noundef i64 @wave_readlane_first_i64(i64 noundef %expr) {
entry:
; CHECK: call i64 @dx.op.waveReadLaneFirst.i64(i32 118, i64 %expr)
  %ret = call i64 @llvm.dx.wave.readlane.first.i64(i64 %expr)
  ret i64 %ret
}

define noundef <2 x half> @wave_readlane_first_v2half(
    <2 x half> noundef %expr) {
entry:
; CHECK: call half @dx.op.waveReadLaneFirst.f16(i32 118, half %expr.i0)
; CHECK: call half @dx.op.waveReadLaneFirst.f16(i32 118, half %expr.i1)
  %ret = call <2 x half> @llvm.dx.wave.readlane.first.v2f16(
      <2 x half> %expr)
  ret <2 x half> %ret
}

define noundef <3 x i32> @wave_readlane_first_v3i32(
    <3 x i32> noundef %expr) {
entry:
; CHECK: call i32 @dx.op.waveReadLaneFirst.i32(i32 118, i32 %expr.i0)
; CHECK: call i32 @dx.op.waveReadLaneFirst.i32(i32 118, i32 %expr.i1)
; CHECK: call i32 @dx.op.waveReadLaneFirst.i32(i32 118, i32 %expr.i2)
  %ret = call <3 x i32> @llvm.dx.wave.readlane.first.v3i32(
      <3 x i32> %expr)
  ret <3 x i32> %ret
}

define noundef <4 x float> @wave_readlane_first_v4float(
    <4 x float> noundef %expr) {
entry:
; CHECK: call float @dx.op.waveReadLaneFirst.f32(i32 118, float %expr.i0)
; CHECK: call float @dx.op.waveReadLaneFirst.f32(i32 118, float %expr.i1)
; CHECK: call float @dx.op.waveReadLaneFirst.f32(i32 118, float %expr.i2)
; CHECK: call float @dx.op.waveReadLaneFirst.f32(i32 118, float %expr.i3)
  %ret = call <4 x float> @llvm.dx.wave.readlane.first.v4f32(
      <4 x float> %expr)
  ret <4 x float> %ret
}

declare half @llvm.dx.wave.readlane.first.f16(half)
declare float @llvm.dx.wave.readlane.first.f32(float)
declare double @llvm.dx.wave.readlane.first.f64(double)
declare i1 @llvm.dx.wave.readlane.first.i1(i1)
declare i16 @llvm.dx.wave.readlane.first.i16(i16)
declare i32 @llvm.dx.wave.readlane.first.i32(i32)
declare i64 @llvm.dx.wave.readlane.first.i64(i64)
declare <2 x half> @llvm.dx.wave.readlane.first.v2f16(<2 x half>)
declare <3 x i32> @llvm.dx.wave.readlane.first.v3i32(<3 x i32>)
declare <4 x float> @llvm.dx.wave.readlane.first.v4f32(<4 x float>)
