; XFAIL: *
; RUN: llc -mtriple=dxil-pc-shadermodel6.9-library -stop-before=dxil-op-lower -o - %s | FileCheck %s

; CHECK-LABEL: define <17 x i32> @test_imad(
; CHECK: call <17 x i32> @llvm.dx.imad.v17i32
define <17 x i32> @test_imad(<17 x i32> %a, <17 x i32> %b, <17 x i32> %c) {
  %result = call <17 x i32> @llvm.dx.imad.v17i32(<17 x i32> %a, <17 x i32> %b, <17 x i32> %c)
  ret <17 x i32> %result
}
declare <17 x i32> @llvm.dx.imad.v17i32(<17 x i32>, <17 x i32>, <17 x i32>)
