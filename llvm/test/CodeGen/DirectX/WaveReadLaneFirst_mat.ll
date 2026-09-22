; RUN: opt -S -scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-compute %s | FileCheck %s

; Test WaveReadLaneFirst scalarization for matrix values.

define noundef <6 x float> @wave_readlane_first_float2x3(
    <6 x float> noundef %expr) {
entry:
; CHECK-LABEL: define noundef <6 x float> @wave_readlane_first_float2x3(
; CHECK-COUNT-6: call float @dx.op.waveReadLaneFirst.f32(i32 118,
  %ret = call <6 x float> @llvm.dx.wave.readlane.first.v6f32(
      <6 x float> %expr)
  ret <6 x float> %ret
}

define noundef <12 x float> @wave_readlane_first_float3x4(
    <12 x float> noundef %expr) {
entry:
; CHECK-LABEL: define noundef <12 x float> @wave_readlane_first_float3x4(
; CHECK-COUNT-12: call float @dx.op.waveReadLaneFirst.f32(i32 118,
  %ret = call <12 x float> @llvm.dx.wave.readlane.first.v12f32(
      <12 x float> %expr)
  ret <12 x float> %ret
}

declare <6 x float> @llvm.dx.wave.readlane.first.v6f32(<6 x float>)
declare <12 x float> @llvm.dx.wave.readlane.first.v12f32(<12 x float>)
