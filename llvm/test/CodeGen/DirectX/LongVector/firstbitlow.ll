; RUN: llc -mtriple=dxil-pc-shadermodel6.8-library -o - %s | FileCheck %s

; CHECK-LABEL: define <17 x i32> @test_firstbitlow(
; CHECK-COUNT-17: call i32 @dx.op.unaryBits.i32(i32 32, i32 {{.*}})
define <17 x i32> @test_firstbitlow(<17 x i32> %a) {
  %result = call <17 x i32> @llvm.dx.firstbitlow.v17i32(<17 x i32> %a)
  ret <17 x i32> %result
}
declare <17 x i32> @llvm.dx.firstbitlow.v17i32(<17 x i32>)
