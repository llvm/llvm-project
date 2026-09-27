; RUN: opt -S -scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library < %s | FileCheck %s

; Test that for scalar values, QuadReadAcrossX maps down to the DirectX op

define noundef half @quad_read_across_x_half(half noundef %expr) {
entry:
; CHECK: call half @dx.op.quadOp.f16(i32 123, half %expr, i8 0)
  %ret = call half @llvm.dx.quad.read.across.x.f16(half %expr)
  ret half %ret
}

define noundef float @quad_read_across_x_float(float noundef %expr) {
entry:
; CHECK: call float @dx.op.quadOp.f32(i32 123, float %expr, i8 0)
  %ret = call float @llvm.dx.quad.read.across.x.f32(float %expr)
  ret float %ret
}

define noundef double @quad_read_across_x_double(double noundef %expr) {
entry:
; CHECK: call double @dx.op.quadOp.f64(i32 123, double %expr, i8 0)
  %ret = call double @llvm.dx.quad.read.across.x.f64(double %expr)
  ret double %ret
}

define noundef i16 @quad_read_across_x_i16(i16 noundef %expr) {
entry:
; CHECK: call i16 @dx.op.quadOp.i16(i32 123, i16 %expr, i8 0)
  %ret = call i16 @llvm.dx.quad.read.across.x.i16(i16 %expr)
  ret i16 %ret
}

define noundef i32 @quad_read_across_x_i32(i32 noundef %expr) {
entry:
; CHECK: call i32 @dx.op.quadOp.i32(i32 123, i32 %expr, i8 0)
  %ret = call i32 @llvm.dx.quad.read.across.x.i32(i32 %expr)
  ret i32 %ret
}

define noundef i64 @quad_read_across_x_i64(i64 noundef %expr) {
entry:
; CHECK: call i64 @dx.op.quadOp.i64(i32 123, i64 %expr, i8 0)
  %ret = call i64 @llvm.dx.quad.read.across.x.i64(i64 %expr)
  ret i64 %ret
}

declare half @llvm.dx.quad.read.across.x.f16(half)
declare float @llvm.dx.quad.read.across.x.f32(float)
declare double @llvm.dx.quad.read.across.x.f64(double)

declare i16 @llvm.dx.quad.read.across.x.i16(i16)
declare i32 @llvm.dx.quad.read.across.x.i32(i32)
declare i64 @llvm.dx.quad.read.across.x.i64(i64)

; Test that for vector values, QuadReadAcrossX scalarizes and maps down to the
; DirectX op

define noundef <2 x half> @quad_read_across_x_v2half(<2 x half> noundef %expr) {
entry:
; CHECK: call half @dx.op.quadOp.f16(i32 123, half %expr.i0, i8 0)
; CHECK: call half @dx.op.quadOp.f16(i32 123, half %expr.i1, i8 0)
  %ret = call <2 x half> @llvm.dx.quad.read.across.x.v2f16(<2 x half> %expr)
  ret <2 x half> %ret
}

define noundef <3 x i32> @quad_read_across_x_v3i32(<3 x i32> noundef %expr) {
entry:
; CHECK: call i32 @dx.op.quadOp.i32(i32 123, i32 %expr.i0, i8 0)
; CHECK: call i32 @dx.op.quadOp.i32(i32 123, i32 %expr.i1, i8 0)
; CHECK: call i32 @dx.op.quadOp.i32(i32 123, i32 %expr.i2, i8 0)
  %ret = call <3 x i32> @llvm.dx.quad.read.across.x.v3i32(<3 x i32> %expr)
  ret <3 x i32> %ret
}

define noundef <4 x double> @quad_read_across_x_v4f64(<4 x double> noundef %expr) {
entry:
; CHECK: call double @dx.op.quadOp.f64(i32 123, double %expr.i0, i8 0)
; CHECK: call double @dx.op.quadOp.f64(i32 123, double %expr.i1, i8 0)
; CHECK: call double @dx.op.quadOp.f64(i32 123, double %expr.i2, i8 0)
; CHECK: call double @dx.op.quadOp.f64(i32 123, double %expr.i3, i8 0)
  %ret = call <4 x double> @llvm.dx.quad.read.across.x.v464(<4 x double> %expr)
  ret <4 x double> %ret
}

declare <2 x half> @llvm.dx.quad.read.across.x.v2f16(<2 x half>)
declare <3 x i32> @llvm.dx.quad.read.across.x.v3i32(<3 x i32>)
declare <4 x double> @llvm.dx.quad.read.across.x.v4f64(<4 x double>)
