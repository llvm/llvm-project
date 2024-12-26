; RUN: opt -S -scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-compute %s | FileCheck %s

; Test that for vector values, WaveReadLaneAt scalarizes and maps down to the
; DirectX op

define noundef <2 x half> @wave_read_lane_v2half(<2 x half> noundef %expr, i32 %idx) {
entry:
; CHECK: call half @dx.op.waveReadLaneAt.f16(i32 117, half %expr.i0, i32 %idx)
; CHECK: call half @dx.op.waveReadLaneAt.f16(i32 117, half %expr.i1, i32 %idx)
  %ret = call <2 x half> @llvm.dx.wave.readlane.f16(<2 x half> %expr, i32 %idx)
  ret <2 x half> %ret
}

define noundef <3 x i32> @wave_read_lane_v3i32(<3 x i32> noundef %expr, i32 %idx) {
entry:
; CHECK: call i32 @dx.op.waveReadLaneAt.i32(i32 117, i32 %expr.i0, i32 %idx)
; CHECK: call i32 @dx.op.waveReadLaneAt.i32(i32 117, i32 %expr.i1, i32 %idx)
; CHECK: call i32 @dx.op.waveReadLaneAt.i32(i32 117, i32 %expr.i2, i32 %idx)
  %ret = call <3 x i32> @llvm.dx.wave.readlane(<3 x i32> %expr, i32 %idx)
  ret <3 x i32> %ret
}

define noundef <4 x double> @wave_read_lane_v4f64(<4 x double> noundef %expr, i32 %idx) {
entry:
; CHECK: call double @dx.op.waveReadLaneAt.f64(i32 117, double %expr.i0, i32 %idx)
; CHECK: call double @dx.op.waveReadLaneAt.f64(i32 117, double %expr.i1, i32 %idx)
; CHECK: call double @dx.op.waveReadLaneAt.f64(i32 117, double %expr.i2, i32 %idx)
; CHECK: call double @dx.op.waveReadLaneAt.f64(i32 117, double %expr.i3, i32 %idx)
  %ret = call <4 x double> @llvm.dx.wave.readlane(<4 x double> %expr, i32 %idx)
  ret <4 x double> %ret
}

declare <2 x half> @llvm.dx.wave.readlane.v2f16(<2 x half>, i32)
declare <3 x i32> @llvm.dx.wave.readlane.v3i32(<3 x i32>, i32)
declare <4 x double> @llvm.dx.wave.readlane.v4f64(<4 x double>, i32)
