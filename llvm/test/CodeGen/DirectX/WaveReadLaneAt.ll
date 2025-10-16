; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-compute %s | FileCheck %s

; Test that for scalar values, WaveReadLaneAt maps down to the DirectX op

define noundef half @wave_rla_half(half noundef %expr, i32 noundef %idx) {
entry:
; CHECK: call half @dx.op.waveReadLaneAt.f16(i32 117, half %expr, i32 %idx){{$}}
  %ret = call half @llvm.dx.wave.readlane.f16(half %expr, i32 %idx)
  ret half %ret
}

define noundef float @wave_rla_float(float noundef %expr, i32 noundef %idx) {
entry:
; CHECK: call float @dx.op.waveReadLaneAt.f32(i32 117, float %expr, i32 %idx){{$}}
  %ret = call float @llvm.dx.wave.readlane(float %expr, i32 %idx)
  ret float %ret
}

define noundef double @wave_rla_double(double noundef %expr, i32 noundef %idx) {
entry:
; CHECK: call double @dx.op.waveReadLaneAt.f64(i32 117, double %expr, i32 %idx){{$}}
  %ret = call double @llvm.dx.wave.readlane(double %expr, i32 %idx)
  ret double %ret
}

define noundef i1 @wave_rla_i1(i1 noundef %expr, i32 noundef %idx) {
entry:
; CHECK: call i1 @dx.op.waveReadLaneAt.i1(i32 117, i1 %expr, i32 %idx){{$}}
  %ret = call i1 @llvm.dx.wave.readlane.i1(i1 %expr, i32 %idx)
  ret i1 %ret
}

define noundef i16 @wave_rla_i16(i16 noundef %expr, i32 noundef %idx) {
entry:
; CHECK: call i16 @dx.op.waveReadLaneAt.i16(i32 117, i16 %expr, i32 %idx){{$}}
  %ret = call i16 @llvm.dx.wave.readlane.i16(i16 %expr, i32 %idx)
  ret i16 %ret
}

define noundef i32 @wave_rla_i32(i32 noundef %expr, i32 noundef %idx) {
entry:
; CHECK: call i32 @dx.op.waveReadLaneAt.i32(i32 117, i32 %expr, i32 %idx){{$}}
  %ret = call i32 @llvm.dx.wave.readlane.i32(i32 %expr, i32 %idx)
  ret i32 %ret
}

define noundef i64 @wave_rla_i64(i64 noundef %expr, i32 noundef %idx) {
entry:
; CHECK: call i64 @dx.op.waveReadLaneAt.i64(i32 117, i64 %expr, i32 %idx){{$}}
  %ret = call i64 @llvm.dx.wave.readlane.i64(i64 %expr, i32 %idx)
  ret i64 %ret
}

; CHECK-NOT: attributes {{.*}} memory(none)

declare half @llvm.dx.wave.readlane.f16(half, i32)
declare float @llvm.dx.wave.readlane.f32(float, i32)
declare double @llvm.dx.wave.readlane.f64(double, i32)

declare i1 @llvm.dx.wave.readlane.i1(i1, i32)
declare i16 @llvm.dx.wave.readlane.i16(i16, i32)
declare i32 @llvm.dx.wave.readlane.i32(i32, i32)
declare i64 @llvm.dx.wave.readlane.i64(i64, i32)
