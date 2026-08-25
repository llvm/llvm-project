; RUN: llc -mtriple=dxil-pc-shadermodel6.8-library -o - %s | FileCheck %s
; RUN: llc -mtriple=dxil-pc-shadermodel6.9-library -stop-before=dxil-op-lower -o - %s | FileCheck %s --check-prefix=SM69-CHECK

; CHECK-LABEL: define <17 x i32> @test_wave_readlane(
; CHECK-COUNT-17: call i32 @dx.op.waveReadLaneAt.i32(i32 117, i32 {{.*}}, i32 {{.*}})
; SM69-CHECK-LABEL: define <17 x i32> @test_wave_readlane(
; SM69-CHECK: call <17 x i32> @llvm.dx.wave.readlane.v17i32
define <17 x i32> @test_wave_readlane(<17 x i32> %a, i32 %lane) {
  %result = call <17 x i32> @llvm.dx.wave.readlane.v17i32(<17 x i32> %a, i32 %lane)
  ret <17 x i32> %result
}
declare <17 x i32> @llvm.dx.wave.readlane.v17i32(<17 x i32>, i32)
