; XFAIL: *
; RUN: llc -mtriple=dxil-pc-shadermodel6.9-library -stop-before=dxil-op-lower -o - %s | FileCheck %s

; CHECK-LABEL: define <17 x i32> @test_wave_readlane(
; CHECK: call <17 x i32> @llvm.dx.wave.readlane.v17i32
define <17 x i32> @test_wave_readlane(<17 x i32> %a, i32 %lane) {
  %result = call <17 x i32> @llvm.dx.wave.readlane.v17i32(<17 x i32> %a, i32 %lane)
  ret <17 x i32> %result
}
declare <17 x i32> @llvm.dx.wave.readlane.v17i32(<17 x i32>, i32)
