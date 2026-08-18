; RUN: llc -mtriple=dxil-pc-shadermodel6.8-library -o - %s | FileCheck %s

; CHECK-LABEL: define <17 x i32> @test_firstbituhigh(
; CHECK-COUNT-17: call i32 @dx.op.unaryBits.i32(i32 33, i32 {{.*}})
define <17 x i32> @test_firstbituhigh(<17 x i32> %a) {
  %result = call <17 x i32> @llvm.dx.firstbituhigh.v17i32(<17 x i32> %a)
  ret <17 x i32> %result
}
declare <17 x i32> @llvm.dx.firstbituhigh.v17i32(<17 x i32>)
