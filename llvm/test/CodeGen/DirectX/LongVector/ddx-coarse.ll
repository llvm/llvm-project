; RUN: llc -mtriple=dxil-pc-shadermodel6.8-library -o - %s | FileCheck %s

; CHECK-LABEL: define <17 x float> @test_ddx_coarse(
; CHECK-COUNT-17: call float @dx.op.unary.f32(i32 83, float {{.*}})
define <17 x float> @test_ddx_coarse(<17 x float> %a) {
  %result = call <17 x float> @llvm.dx.ddx.coarse.v17f32(<17 x float> %a)
  ret <17 x float> %result
}
declare <17 x float> @llvm.dx.ddx.coarse.v17f32(<17 x float>)
