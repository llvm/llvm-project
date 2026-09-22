; RUN: llc -mtriple=dxil-pc-shadermodel6.8-library -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-SCALAR
; RUN: llc -mtriple=dxil-pc-shadermodel6.9-library -stop-before=dxil-op-lower -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-VECTOR

; CHECK-LABEL: define <5 x float> @wave_readlane_first_v5float(
; CHECK-SCALAR-COUNT-5: call float @dx.op.waveReadLaneFirst.f32(i32 118,
; CHECK-VECTOR: call <5 x float> @llvm.dx.wave.readlane.first.v5f32
define <5 x float> @wave_readlane_first_v5float(<5 x float> %expr) {
  %ret = call <5 x float> @llvm.dx.wave.readlane.first.v5f32(
      <5 x float> %expr)
  ret <5 x float> %ret
}

declare <5 x float> @llvm.dx.wave.readlane.first.v5f32(<5 x float>)
