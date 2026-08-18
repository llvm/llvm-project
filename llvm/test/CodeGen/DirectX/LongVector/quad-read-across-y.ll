; RUN: llc -mtriple=dxil-pc-shadermodel6.8-library -o - %s | FileCheck %s

; CHECK-LABEL: define <17 x i32> @test_quad_read_across_y(
; CHECK-COUNT-17: call i32 @dx.op.quadOp.i32(i32 123, i32 {{.*}}, i8 1)
define <17 x i32> @test_quad_read_across_y(<17 x i32> %a) {
  %result = call <17 x i32> @llvm.dx.quad.read.across.y.v17i32(<17 x i32> %a)
  ret <17 x i32> %result
}
declare <17 x i32> @llvm.dx.quad.read.across.y.v17i32(<17 x i32>)
