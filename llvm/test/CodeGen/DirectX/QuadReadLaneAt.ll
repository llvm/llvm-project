; RUN: opt -S -scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library < %s | FileCheck %s

; Test that for scalar values, QuadReadLaneAt maps down to the DirectX op

define noundef i1 @quad_read_lane_at_bool(i1 noundef %expr, i32 noundef %idx) {
entry:
; CHECK: call i1 @dx.op.quadReadLaneAt.i1(i32 122, i1 %expr, i32 %idx)
  %ret = call i1 @llvm.dx.quad.read.lane.at.i1(i1 %expr, i32 %idx)
  ret i1 %ret
}

define noundef half @quad_read_lane_at_half(half noundef %expr, i32 noundef %idx) {
entry:
; CHECK: call half @dx.op.quadReadLaneAt.f16(i32 122, half %expr, i32 %idx)
  %ret = call half @llvm.dx.quad.read.lane.at.f16(half %expr, i32 %idx)
  ret half %ret
}

define noundef float @quad_read_lane_at_float(float noundef %expr, i32 noundef %idx) {
entry:
; CHECK: call float @dx.op.quadReadLaneAt.f32(i32 122, float %expr, i32 %idx)
  %ret = call float @llvm.dx.quad.read.lane.at.f32(float %expr, i32 %idx)
  ret float %ret
}

define noundef double @quad_read_lane_at_double(double noundef %expr, i32 noundef %idx) {
entry:
; CHECK: call double @dx.op.quadReadLaneAt.f64(i32 122, double %expr, i32 %idx)
  %ret = call double @llvm.dx.quad.read.lane.at.f64(double %expr, i32 %idx)
  ret double %ret
}

define noundef i16 @quad_read_lane_at_i16(i16 noundef %expr, i32 noundef %idx) {
entry:
; CHECK: call i16 @dx.op.quadReadLaneAt.i16(i32 122, i16 %expr, i32 %idx)
  %ret = call i16 @llvm.dx.quad.read.lane.at.i16(i16 %expr, i32 %idx)
  ret i16 %ret
}

define noundef i32 @quad_read_lane_at_i32(i32 noundef %expr, i32 noundef %idx) {
entry:
; CHECK: call i32 @dx.op.quadReadLaneAt.i32(i32 122, i32 %expr, i32 %idx)
  %ret = call i32 @llvm.dx.quad.read.lane.at.i32(i32 %expr, i32 %idx)
  ret i32 %ret
}

define noundef i64 @quad_read_lane_at_i64(i64 noundef %expr, i32 noundef %idx) {
entry:
; CHECK: call i64 @dx.op.quadReadLaneAt.i64(i32 122, i64 %expr, i32 %idx)
  %ret = call i64 @llvm.dx.quad.read.lane.at.i64(i64 %expr, i32 %idx)
  ret i64 %ret
}

declare i1 @llvm.dx.quad.read.lane.at.i1(i1, i32)
declare half @llvm.dx.quad.read.lane.at.f16(half, i32)
declare float @llvm.dx.quad.read.lane.at.f32(float, i32)
declare double @llvm.dx.quad.read.lane.at.f64(double, i32)

declare i16 @llvm.dx.quad.read.lane.at.i16(i16, i32)
declare i32 @llvm.dx.quad.read.lane.at.i32(i32, i32)
declare i64 @llvm.dx.quad.read.lane.at.i64(i64, i32)

; Test that for vector values, QuadReadLaneAt scalarizes and maps down to the
; DirectX op

define noundef <2 x half> @quad_read_lane_at_v2half(<2 x half> noundef %expr, i32 noundef %idx) {
entry:
; CHECK: call half @dx.op.quadReadLaneAt.f16(i32 122, half %expr.i0, i32 %idx)
; CHECK: call half @dx.op.quadReadLaneAt.f16(i32 122, half %expr.i1, i32 %idx)
  %ret = call <2 x half> @llvm.dx.quad.read.lane.at.v2f16(<2 x half> %expr, i32 %idx)
  ret <2 x half> %ret
}

define noundef <3 x i32> @quad_read_lane_at_v3i32(<3 x i32> noundef %expr, i32 noundef %idx) {
entry:
; CHECK: call i32 @dx.op.quadReadLaneAt.i32(i32 122, i32 %expr.i0, i32 %idx)
; CHECK: call i32 @dx.op.quadReadLaneAt.i32(i32 122, i32 %expr.i1, i32 %idx)
; CHECK: call i32 @dx.op.quadReadLaneAt.i32(i32 122, i32 %expr.i2, i32 %idx)
  %ret = call <3 x i32> @llvm.dx.quad.read.lane.at.v3i32(<3 x i32> %expr, i32 %idx)
  ret <3 x i32> %ret
}

define noundef <4 x double> @quad_read_lane_at_v4f64(<4 x double> noundef %expr, i32 noundef %idx) {
entry:
; CHECK: call double @dx.op.quadReadLaneAt.f64(i32 122, double %expr.i0, i32 %idx)
; CHECK: call double @dx.op.quadReadLaneAt.f64(i32 122, double %expr.i1, i32 %idx)
; CHECK: call double @dx.op.quadReadLaneAt.f64(i32 122, double %expr.i2, i32 %idx)
; CHECK: call double @dx.op.quadReadLaneAt.f64(i32 122, double %expr.i3, i32 %idx)
  %ret = call <4 x double> @llvm.dx.quad.read.lane.at.v4f64(<4 x double> %expr, i32 %idx)
  ret <4 x double> %ret
}

declare <2 x half> @llvm.dx.quad.read.lane.at.v2f16(<2 x half>, i32)
declare <3 x i32> @llvm.dx.quad.read.lane.at.v3i32(<3 x i32>, i32)
declare <4 x double> @llvm.dx.quad.read.lane.at.v4f64(<4 x double>, i32)
