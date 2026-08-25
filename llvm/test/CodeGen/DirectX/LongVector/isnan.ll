; RUN: llc -mtriple=dxil-pc-shadermodel6.8-library -o - %s | FileCheck %s
; RUN: llc -mtriple=dxil-pc-shadermodel6.9-library -stop-before=dxil-op-lower -o - %s | FileCheck %s --check-prefix=SM69-CHECK

; CHECK-LABEL: define <17 x i1> @test_isnan(
; CHECK-COUNT-17: call i1 @dx.op.isSpecialFloat.f32(i32 8, float {{.*}})
; SM69-CHECK-LABEL: define <17 x i1> @test_isnan(
; SM69-CHECK: call <17 x i1> @llvm.dx.isnan.v17f32
define <17 x i1> @test_isnan(<17 x float> %a) {
  %result = call <17 x i1> @llvm.dx.isnan.v17f32(<17 x float> %a)
  ret <17 x i1> %result
}
declare <17 x i1> @llvm.dx.isnan.v17f32(<17 x float>)
