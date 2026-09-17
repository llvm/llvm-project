; RUN: llc -mtriple=dxil-pc-shadermodel6.8-library -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-SCALAR
; RUN: llc -mtriple=dxil-pc-shadermodel6.9-library -stop-before=dxil-op-lower -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-VECTOR

; CHECK-LABEL: define <17 x i1> @test_isinf(
; CHECK-SCALAR-COUNT-17: call i1 @dx.op.isSpecialFloat.f32(i32 9, float {{.*}})
; CHECK-VECTOR: call <17 x i1> @llvm.dx.isinf.v17f32
define <17 x i1> @test_isinf(<17 x float> %a) {
  %result = call <17 x i1> @llvm.dx.isinf.v17f32(<17 x float> %a)
  ret <17 x i1> %result
}
declare <17 x i1> @llvm.dx.isinf.v17f32(<17 x float>)
