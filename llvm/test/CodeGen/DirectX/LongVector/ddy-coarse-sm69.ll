; XFAIL: *
; RUN: llc -mtriple=dxil-pc-shadermodel6.9-library -stop-before=dxil-op-lower -o - %s | FileCheck %s

; CHECK-LABEL: define <17 x float> @test_ddy_coarse(
; CHECK: call <17 x float> @llvm.dx.ddy.coarse.v17f32
define <17 x float> @test_ddy_coarse(<17 x float> %a) {
  %result = call <17 x float> @llvm.dx.ddy.coarse.v17f32(<17 x float> %a)
  ret <17 x float> %result
}
declare <17 x float> @llvm.dx.ddy.coarse.v17f32(<17 x float>)
