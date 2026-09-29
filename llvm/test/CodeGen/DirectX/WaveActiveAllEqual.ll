; RUN: opt -S -scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library < %s | FileCheck %s

; Test that for scalar values, WaveAcitveProduct maps down to the DirectX op

define noundef i1 @wave_active_all_equal_half(half noundef %expr) {
entry:
; CHECK: call i1 @dx.op.waveActiveAllEqual.f16(i32 115, half %expr)
  %ret = call i1 @llvm.dx.wave.all.equal.f16(half %expr)
  ret i1 %ret
}

define noundef i1 @wave_active_all_equal_float(float noundef %expr) {
entry:
; CHECK: call i1 @dx.op.waveActiveAllEqual.f32(i32 115, float %expr)
  %ret = call i1 @llvm.dx.wave.all.equal.f32(float %expr)
  ret i1 %ret
}

define noundef i1 @wave_active_all_equal_double(double noundef %expr) {
entry:
; CHECK: call i1 @dx.op.waveActiveAllEqual.f64(i32 115, double %expr)
  %ret = call i1 @llvm.dx.wave.all.equal.f64(double %expr)
  ret i1 %ret
}

define noundef i1 @wave_active_all_equal_i16(i16 noundef %expr) {
entry:
; CHECK: call i1 @dx.op.waveActiveAllEqual.i16(i32 115, i16 %expr)
  %ret = call i1 @llvm.dx.wave.all.equal.i16(i16 %expr)
  ret i1 %ret
}

define noundef i1 @wave_active_all_equal_i32(i32 noundef %expr) {
entry:
; CHECK: call i1 @dx.op.waveActiveAllEqual.i32(i32 115, i32 %expr)
  %ret = call i1 @llvm.dx.wave.all.equal.i32(i32 %expr)
  ret i1 %ret
}

define noundef i1 @wave_active_all_equal_i64(i64 noundef %expr) {
entry:
; CHECK: call i1 @dx.op.waveActiveAllEqual.i64(i32 115, i64 %expr)
  %ret = call i1 @llvm.dx.wave.all.equal.i64(i64 %expr)
  ret i1 %ret
}

declare i1 @llvm.dx.wave.all.equal.f16(half)
declare i1 @llvm.dx.wave.all.equal.f32(float)
declare i1 @llvm.dx.wave.all.equal.f64(double)

declare i1 @llvm.dx.wave.all.equal.i16(i16)
declare i1 @llvm.dx.wave.all.equal.i32(i32)
declare i1 @llvm.dx.wave.all.equal.i64(i64)

; Test that for vector values, WaveAcitveProduct scalarizes and maps down to the
; DirectX op

define noundef <2 x i1> @wave_active_all_equal_v2half(<2 x half> noundef %expr) {
entry:
; CHECK: %[[EXPR0:.*]] = extractelement <2 x half> %expr, i64 0
; CHECK: %[[RET0:.*]] = call i1 @dx.op.waveActiveAllEqual.f16(i32 115, half %[[EXPR0]])
; CHECK: %[[EXPR1:.*]] = extractelement <2 x half> %expr, i64 1
; CHECK: %[[RET1:.*]] = call i1 @dx.op.waveActiveAllEqual.f16(i32 115, half %[[EXPR1]])
; CHECK: %[[RETUPTO0:.*]] = insertelement <2 x i1> poison, i1 %[[RET0]], i64 0
; CHECK: %ret = insertelement <2 x i1> %[[RETUPTO0]], i1 %[[RET1]], i64 1
; CHECK: ret <2 x i1> %ret

  %ret = call <2 x i1> @llvm.dx.wave.all.equal.v2f16(<2 x half> %expr)
  ret <2 x i1> %ret
}

define noundef <3 x i1> @wave_active_all_equal_v3i32(<3 x i32> noundef %expr) {
entry:
; CHECK: %[[EXPR0:.*]] = extractelement <3 x i32> %expr, i64 0
; CHECK: %[[RET0:.*]] = call i1 @dx.op.waveActiveAllEqual.i32(i32 115, i32 %[[EXPR0]])
; CHECK: %[[EXPR1:.*]] = extractelement <3 x i32> %expr, i64 1
; CHECK: %[[RET1:.*]] = call i1 @dx.op.waveActiveAllEqual.i32(i32 115, i32 %[[EXPR1]])
; CHECK: %[[EXPR2:.*]] = extractelement <3 x i32> %expr, i64 2
; CHECK: %[[RET2:.*]] = call i1 @dx.op.waveActiveAllEqual.i32(i32 115, i32 %[[EXPR2]])
; CHECK: %[[RETUPTO0:.*]] = insertelement <3 x i1> poison, i1 %[[RET0]], i64 0
; CHECK: %[[RETUPTO1:.*]] = insertelement <3 x i1> %[[RETUPTO0]], i1 %[[RET1]], i64 1
; CHECK: %ret = insertelement <3 x i1> %[[RETUPTO1]], i1 %[[RET2]], i64 2

  %ret = call <3 x i1> @llvm.dx.wave.all.equal.v3i32(<3 x i32> %expr)
  ret <3 x i1> %ret
}

define noundef <4 x i1> @wave_active_all_equal_v4f64(<4 x double> noundef %expr) {
entry:
; CHECK: %[[EXPR0:.*]] = extractelement <4 x double> %expr, i64 0
; CHECK: %[[RET0:.*]] = call i1 @dx.op.waveActiveAllEqual.f64(i32 115, double %[[EXPR0]])
; CHECK: %[[EXPR1:.*]] = extractelement <4 x double> %expr, i64 1
; CHECK: %[[RET1:.*]] = call i1 @dx.op.waveActiveAllEqual.f64(i32 115, double %[[EXPR1]])
; CHECK: %[[EXPR2:.*]] = extractelement <4 x double> %expr, i64 2
; CHECK: %[[RET2:.*]] = call i1 @dx.op.waveActiveAllEqual.f64(i32 115, double %[[EXPR2]])
; CHECK: %[[EXPR3:.*]] = extractelement <4 x double> %expr, i64 3
; CHECK: %[[RET3:.*]] = call i1 @dx.op.waveActiveAllEqual.f64(i32 115, double %[[EXPR3]])
; CHECK: %[[RETUPTO0:.*]] = insertelement <4 x i1> poison, i1 %[[RET0]], i64 0
; CHECK: %[[RETUPTO1:.*]] = insertelement <4 x i1> %[[RETUPTO0]], i1 %[[RET1]], i64 1
; CHECK: %[[RETUPTO2:.*]] = insertelement <4 x i1> %[[RETUPTO1]], i1 %[[RET2]], i64 2
; CHECK: %ret = insertelement <4 x i1> %[[RETUPTO2]], i1 %[[RET3]], i64 3

  %ret = call <4 x i1> @llvm.dx.wave.all.equal.v464(<4 x double> %expr)
  ret <4 x i1> %ret
}

declare <2 x i1> @llvm.dx.wave.all.equal.v2f16(<2 x half>)
declare <3 x i1> @llvm.dx.wave.all.equal.v3i32(<3 x i32>)
declare <4 x i1> @llvm.dx.wave.all.equal.v4f64(<4 x double>)
