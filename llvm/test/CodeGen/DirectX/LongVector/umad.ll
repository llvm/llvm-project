; RUN: llc -mtriple=dxil-pc-shadermodel6.8-library -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-SCALAR
; RUN: llc -mtriple=dxil-pc-shadermodel6.9-library -stop-before=dxil-op-lower -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-VECTOR

; CHECK-LABEL: define <17 x i32> @test_umad(
; CHECK-SCALAR-COUNT-17: call i32 @dx.op.tertiary.i32(i32 49, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
; CHECK-VECTOR: call <17 x i32> @llvm.dx.umad.v17i32
define <17 x i32> @test_umad(<17 x i32> %a, <17 x i32> %b, <17 x i32> %c) {
  %result = call <17 x i32> @llvm.dx.umad.v17i32(<17 x i32> %a, <17 x i32> %b, <17 x i32> %c)
  ret <17 x i32> %result
}
declare <17 x i32> @llvm.dx.umad.v17i32(<17 x i32>, <17 x i32>, <17 x i32>)
