; RUN: llc -mtriple=dxil-pc-shadermodel6.8-library -o - %s | FileCheck %s

; CHECK-LABEL: define <17 x float> @test_frac(
; CHECK-COUNT-17: call float @dx.op.unary.f32(i32 22, float {{.*}})
define <17 x float> @test_frac(<17 x float> %a) {
  %result = call <17 x float> @llvm.dx.frac.v17f32(<17 x float> %a)
  ret <17 x float> %result
}
declare <17 x float> @llvm.dx.frac.v17f32(<17 x float>)
