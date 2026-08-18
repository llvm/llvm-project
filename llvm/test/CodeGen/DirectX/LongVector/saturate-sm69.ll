; XFAIL: *
; RUN: llc -mtriple=dxil-pc-shadermodel6.9-library -stop-before=dxil-op-lower -o - %s | FileCheck %s

; CHECK-LABEL: define <17 x float> @test_saturate(
; CHECK: call <17 x float> @llvm.dx.saturate.v17f32
define <17 x float> @test_saturate(<17 x float> %a) {
  %result = call <17 x float> @llvm.dx.saturate.v17f32(<17 x float> %a)
  ret <17 x float> %result
}
declare <17 x float> @llvm.dx.saturate.v17f32(<17 x float>)
