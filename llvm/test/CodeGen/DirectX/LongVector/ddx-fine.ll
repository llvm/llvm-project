; RUN: llc -mtriple=dxil-pc-shadermodel6.8-library -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-SCALAR
; RUN: llc -mtriple=dxil-pc-shadermodel6.9-library -stop-before=dxil-op-lower -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-VECTOR

; CHECK-LABEL: define <17 x float> @test_ddx_fine(
; CHECK-SCALAR-COUNT-17: call float @dx.op.unary.f32(i32 85, float {{.*}})
; CHECK-VECTOR: call <17 x float> @llvm.dx.ddx.fine.v17f32
define <17 x float> @test_ddx_fine(<17 x float> %a) {
  %result = call <17 x float> @llvm.dx.ddx.fine.v17f32(<17 x float> %a)
  ret <17 x float> %result
}
declare <17 x float> @llvm.dx.ddx.fine.v17f32(<17 x float>)
