; RUN: opt -S -scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library < %s | FileCheck %s

; Test that for scalar values, WaveActiveMax maps down to the DirectX op

define noundef half @wave_active_max_half(half noundef %expr) {
entry:
; CHECK: call half @dx.op.waveActiveOp.f16(i32 119, half %expr, i8 3, i8 0){{$}}
  %ret = call half @llvm.dx.wave.reduce.max.f16(half %expr)
  ret half %ret
}

define noundef float @wave_active_max_float(float noundef %expr) {
entry:
; CHECK: call float @dx.op.waveActiveOp.f32(i32 119, float %expr, i8 3, i8 0){{$}}
  %ret = call float @llvm.dx.wave.reduce.max.f32(float %expr)
  ret float %ret
}

define noundef double @wave_active_max_double(double noundef %expr) {
entry:
; CHECK: call double @dx.op.waveActiveOp.f64(i32 119, double %expr, i8 3, i8 0){{$}}
  %ret = call double @llvm.dx.wave.reduce.max.f64(double %expr)
  ret double %ret
}

define noundef i16 @wave_active_max_i16(i16 noundef %expr) {
entry:
; CHECK: call i16 @dx.op.waveActiveOp.i16(i32 119, i16 %expr, i8 3, i8 0){{$}}
  %ret = call i16 @llvm.dx.wave.reduce.max.i16(i16 %expr)
  ret i16 %ret
}

define noundef i32 @wave_active_max_i32(i32 noundef %expr) {
entry:
; CHECK: call i32 @dx.op.waveActiveOp.i32(i32 119, i32 %expr, i8 3, i8 0){{$}}
  %ret = call i32 @llvm.dx.wave.reduce.max.i32(i32 %expr)
  ret i32 %ret
}

define noundef i64 @wave_active_max_i64(i64 noundef %expr) {
entry:
; CHECK: call i64 @dx.op.waveActiveOp.i64(i32 119, i64 %expr, i8 3, i8 0){{$}}
  %ret = call i64 @llvm.dx.wave.reduce.max.i64(i64 %expr)
  ret i64 %ret
}

define noundef i16 @wave_active_umax_i16(i16 noundef %expr) {
entry:
; CHECK: call i16 @dx.op.waveActiveOp.i16(i32 119, i16 %expr, i8 3, i8 1){{$}}
  %ret = call i16 @llvm.dx.wave.reduce.umax.i16(i16 %expr)
  ret i16 %ret
}

define noundef i32 @wave_active_umax_i32(i32 noundef %expr) {
entry:
; CHECK: call i32 @dx.op.waveActiveOp.i32(i32 119, i32 %expr, i8 3, i8 1){{$}}
  %ret = call i32 @llvm.dx.wave.reduce.umax.i32(i32 %expr)
  ret i32 %ret
}

define noundef i64 @wave_active_umax_i64(i64 noundef %expr) {
entry:
; CHECK: call i64 @dx.op.waveActiveOp.i64(i32 119, i64 %expr, i8 3, i8 1){{$}}
  %ret = call i64 @llvm.dx.wave.reduce.umax.i64(i64 %expr)
  ret i64 %ret
}

declare half @llvm.dx.wave.reduce.max.f16(half)
declare float @llvm.dx.wave.reduce.max.f32(float)
declare double @llvm.dx.wave.reduce.max.f64(double)

declare i16 @llvm.dx.wave.reduce.max.i16(i16)
declare i32 @llvm.dx.wave.reduce.max.i32(i32)
declare i64 @llvm.dx.wave.reduce.max.i64(i64)

declare i16 @llvm.dx.wave.reduce.umax.i16(i16)
declare i32 @llvm.dx.wave.reduce.umax.i32(i32)
declare i64 @llvm.dx.wave.reduce.umax.i64(i64)

; Test that for vector values, WaveActiveMax scalarizes and maps down to the
; DirectX op

define noundef <2 x half> @wave_active_max_v2half(<2 x half> noundef %expr) {
entry:
; CHECK: call half @dx.op.waveActiveOp.f16(i32 119, half %expr.i0, i8 3, i8 0){{$}}
; CHECK: call half @dx.op.waveActiveOp.f16(i32 119, half %expr.i1, i8 3, i8 0){{$}}
  %ret = call <2 x half> @llvm.dx.wave.reduce.max.v2f16(<2 x half> %expr)
  ret <2 x half> %ret
}

define noundef <3 x i32> @wave_active_max_v3i32(<3 x i32> noundef %expr) {
entry:
; CHECK: call i32 @dx.op.waveActiveOp.i32(i32 119, i32 %expr.i0, i8 3, i8 0){{$}}
; CHECK: call i32 @dx.op.waveActiveOp.i32(i32 119, i32 %expr.i1, i8 3, i8 0){{$}}
; CHECK: call i32 @dx.op.waveActiveOp.i32(i32 119, i32 %expr.i2, i8 3, i8 0){{$}}
  %ret = call <3 x i32> @llvm.dx.wave.reduce.max.v3i32(<3 x i32> %expr)
  ret <3 x i32> %ret
}

define noundef <4 x double> @wave_active_max_v4f64(<4 x double> noundef %expr) {
entry:
; CHECK: call double @dx.op.waveActiveOp.f64(i32 119, double %expr.i0, i8 3, i8 0){{$}}
; CHECK: call double @dx.op.waveActiveOp.f64(i32 119, double %expr.i1, i8 3, i8 0){{$}}
; CHECK: call double @dx.op.waveActiveOp.f64(i32 119, double %expr.i2, i8 3, i8 0){{$}}
; CHECK: call double @dx.op.waveActiveOp.f64(i32 119, double %expr.i3, i8 3, i8 0){{$}}
  %ret = call <4 x double> @llvm.dx.wave.reduce.max.v4f64(<4 x double> %expr)
  ret <4 x double> %ret
}

declare <2 x half> @llvm.dx.wave.reduce.max.v2f16(<2 x half>)
declare <3 x i32> @llvm.dx.wave.reduce.max.v3i32(<3 x i32>)
declare <4 x double> @llvm.dx.wave.reduce.max.v4f64(<4 x double>)

define noundef <2 x i16> @wave_active_umax_v2i16(<2 x i16> noundef %expr) {
entry:
; CHECK: call i16 @dx.op.waveActiveOp.i16(i32 119, i16 %expr.i0, i8 3, i8 1){{$}}
; CHECK: call i16 @dx.op.waveActiveOp.i16(i32 119, i16 %expr.i1, i8 3, i8 1){{$}}
  %ret = call <2 x i16> @llvm.dx.wave.reduce.umax.v2f16(<2 x i16> %expr)
  ret <2 x i16> %ret
}

define noundef <3 x i32> @wave_active_umax_v3i32(<3 x i32> noundef %expr) {
entry:
; CHECK: call i32 @dx.op.waveActiveOp.i32(i32 119, i32 %expr.i0, i8 3, i8 1){{$}}
; CHECK: call i32 @dx.op.waveActiveOp.i32(i32 119, i32 %expr.i1, i8 3, i8 1){{$}}
; CHECK: call i32 @dx.op.waveActiveOp.i32(i32 119, i32 %expr.i2, i8 3, i8 1){{$}}
  %ret = call <3 x i32> @llvm.dx.wave.reduce.umax.v3i32(<3 x i32> %expr)
  ret <3 x i32> %ret
}

define noundef <4 x i64> @wave_active_umax_v4f64(<4 x i64> noundef %expr) {
entry:
; CHECK: call i64 @dx.op.waveActiveOp.i64(i32 119, i64 %expr.i0, i8 3, i8 1){{$}}
; CHECK: call i64 @dx.op.waveActiveOp.i64(i32 119, i64 %expr.i1, i8 3, i8 1){{$}}
; CHECK: call i64 @dx.op.waveActiveOp.i64(i32 119, i64 %expr.i2, i8 3, i8 1){{$}}
; CHECK: call i64 @dx.op.waveActiveOp.i64(i32 119, i64 %expr.i3, i8 3, i8 1){{$}}
  %ret = call <4 x i64> @llvm.dx.wave.reduce.umax.v4f64(<4 x i64> %expr)
  ret <4 x i64> %ret
}

declare <2 x i16> @llvm.dx.wave.reduce.umax.v2f16(<2 x i16>)
declare <3 x i32> @llvm.dx.wave.reduce.umax.v3i32(<3 x i32>)
declare <4 x i64> @llvm.dx.wave.reduce.umax.v4f64(<4 x i64>)
