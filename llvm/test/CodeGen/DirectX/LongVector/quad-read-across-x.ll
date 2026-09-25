; RUN: llc -mtriple=dxil-pc-shadermodel6.8-library -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-SCALAR
; RUN: llc -mtriple=dxil-pc-shadermodel6.9-library -stop-before=dxil-op-lower -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-VECTOR

; CHECK-LABEL: define <17 x i32> @test_quad_read_across_x(
; CHECK-SCALAR-COUNT-17: call i32 @dx.op.quadOp.i32(i32 123, i32 {{.*}}, i8 0)
; CHECK-VECTOR: call <17 x i32> @llvm.dx.quad.read.across.x.v17i32
define <17 x i32> @test_quad_read_across_x(<17 x i32> %a) {
  %result = call <17 x i32> @llvm.dx.quad.read.across.x.v17i32(<17 x i32> %a)
  ret <17 x i32> %result
}
declare <17 x i32> @llvm.dx.quad.read.across.x.v17i32(<17 x i32>)
