; RUN: llc -mtriple=dxil-pc-shadermodel6.8-library -o - %s | FileCheck %s
; RUN: llc -mtriple=dxil-pc-shadermodel6.9-library -stop-before=dxil-op-lower -o - %s | FileCheck %s --check-prefix=SM69-CHECK

; CHECK-LABEL: define <17 x i32> @test_firstbitlow(
; CHECK-COUNT-17: call i32 @dx.op.unaryBits.i32(i32 32, i32 {{.*}})
; SM69-CHECK-LABEL: define <17 x i32> @test_firstbitlow(
; SM69-CHECK: call <17 x i32> @llvm.dx.firstbitlow.v17i32
define <17 x i32> @test_firstbitlow(<17 x i32> %a) {
  %result = call <17 x i32> @llvm.dx.firstbitlow.v17i32(<17 x i32> %a)
  ret <17 x i32> %result
}
declare <17 x i32> @llvm.dx.firstbitlow.v17i32(<17 x i32>)
