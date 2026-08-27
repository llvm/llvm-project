; RUN: llc -mtriple=dxil-pc-shadermodel6.8-library -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-SCALAR
; RUN: llc -mtriple=dxil-pc-shadermodel6.9-library -stop-before=dxil-op-lower -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-VECTOR

; CHECK-LABEL: define <17 x i32> @test_wave_prefix_product(
; CHECK-SCALAR-COUNT-17: call i32 @dx.op.wavePrefixOp.i32(i32 121, i32 {{.*}}, i8 1, i8 0)
; CHECK-VECTOR: call <17 x i32> @llvm.dx.wave.prefix.product.v17i32
define <17 x i32> @test_wave_prefix_product(<17 x i32> %a) {
  %result = call <17 x i32> @llvm.dx.wave.prefix.product.v17i32(<17 x i32> %a)
  ret <17 x i32> %result
}
declare <17 x i32> @llvm.dx.wave.prefix.product.v17i32(<17 x i32>)
