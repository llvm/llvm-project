; RUN: llc -mtriple=dxil-pc-shadermodel6.8-library -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-SCALAR
; RUN: llc -mtriple=dxil-pc-shadermodel6.9-library -stop-before=dxil-op-lower -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-VECTOR

; CHECK-LABEL: define <17 x i32> @test_wave_readlane(
; CHECK-SCALAR-COUNT-17: call i32 @dx.op.waveReadLaneAt.i32(i32 117, i32 {{.*}}, i32 {{.*}})
; CHECK-VECTOR: call <17 x i32> @llvm.dx.wave.readlane.v17i32
define <17 x i32> @test_wave_readlane(<17 x i32> %a, i32 %lane) {
  %result = call <17 x i32> @llvm.dx.wave.readlane.v17i32(<17 x i32> %a, i32 %lane)
  ret <17 x i32> %result
}
declare <17 x i32> @llvm.dx.wave.readlane.v17i32(<17 x i32>, i32)
