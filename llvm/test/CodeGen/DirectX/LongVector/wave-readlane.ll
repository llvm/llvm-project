; RUN: llc -mtriple=dxil-pc-shadermodel6.8-library -o - %s | FileCheck %s

; CHECK-LABEL: define <17 x i32> @test_wave_readlane(
; CHECK-COUNT-17: call i32 @dx.op.waveReadLaneAt.i32(i32 117, i32 {{.*}}, i32 {{.*}})
define <17 x i32> @test_wave_readlane(<17 x i32> %a, i32 %lane) {
  %result = call <17 x i32> @llvm.dx.wave.readlane.v17i32(<17 x i32> %a, i32 %lane)
  ret <17 x i32> %result
}
declare <17 x i32> @llvm.dx.wave.readlane.v17i32(<17 x i32>, i32)
