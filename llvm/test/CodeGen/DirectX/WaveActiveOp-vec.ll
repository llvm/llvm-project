; RUN: opt -S -scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-compute %s | FileCheck %s

; Test that for scalar values, WaveReadLaneAt maps down to the DirectX op

define noundef <2 x half> @wave_active_op_v2half(<2 x half> noundef %expr) {
entry:
; CHECK: call half @dx.op.waveActiveOp.f16(i32 119, half %expr.i0, i8 0, i8 0)
; CHECK: call half @dx.op.waveActiveOp.f16(i32 119, half %expr.i1, i8 0, i8 0)
  %ret = call <2 x half> @llvm.dx.wave.active.op.f16(<2 x half> %expr, i8 0, i8 0)
  ret <2 x half> %ret
}

define noundef <3 x i32> @wave_active_op_v3i32(<3 x i32> noundef %expr) {
entry:
; CHECK: call i32 @dx.op.waveActiveOp.i32(i32 119, i32 %expr.i0, i8 1, i8 1)
; CHECK: call i32 @dx.op.waveActiveOp.i32(i32 119, i32 %expr.i1, i8 1, i8 1)
; CHECK: call i32 @dx.op.waveActiveOp.i32(i32 119, i32 %expr.i2, i8 1, i8 1)
  %ret = call <3 x i32> @llvm.dx.wave.active.op(<3 x i32> %expr, i8 1, i8 1)
  ret <3 x i32> %ret
}

define noundef <4 x double> @wave_active_op_v4f64(<4 x double> noundef %expr) {
entry:
; CHECK: call double @dx.op.waveActiveOp.f64(i32 119, double %expr.i0, i8 2, i8 0)
; CHECK: call double @dx.op.waveActiveOp.f64(i32 119, double %expr.i1, i8 2, i8 0)
; CHECK: call double @dx.op.waveActiveOp.f64(i32 119, double %expr.i2, i8 2, i8 0)
; CHECK: call double @dx.op.waveActiveOp.f64(i32 119, double %expr.i3, i8 2, i8 0)
  %ret = call <4 x double> @llvm.dx.wave.active.op(<4 x double> %expr, i8 2, i8 0)
  ret <4 x double> %ret
}

declare <2 x half> @llvm.dx.wave.active.op.v2f16(<2 x half>, i8, i8)
declare <3 x i32> @llvm.dx.wave.active.op.v3i32(<3 x i32>, i8, i8)
declare <4 x double> @llvm.dx.wave.active.op.v4f64(<4 x double>, i8, i8)
