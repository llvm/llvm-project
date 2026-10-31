; RUN: llc -mtriple=dxil-pc-shadermodel6.8-library -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-SCALAR
; RUN: llc -mtriple=dxil-pc-shadermodel6.9-library -stop-before=dxil-op-lower -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-VECTOR

; llvm.frexp is expanded into integer operations on the float bit pattern.

; CHECK-LABEL: define <17 x float> @test_frexp(
; CHECK-SCALAR-NOT: @llvm.frexp
; CHECK-SCALAR-COUNT-17: and i32 {{.*}}, 2139095040
; CHECK-VECTOR-NOT: @llvm.frexp
; CHECK-VECTOR: bitcast <17 x float> %a to <17 x i32>
; CHECK-VECTOR: and <17 x i32> {{.*}}, splat (i32 2139095040)
; CHECK-VECTOR: lshr <17 x i32> {{.*}}, splat (i32 23)
define <17 x float> @test_frexp(<17 x float> %a) {
  %r = call { <17 x float>, <17 x i32> } @llvm.frexp.v17f32.v17i32(<17 x float> %a)
  %m = extractvalue { <17 x float>, <17 x i32> } %r, 0
  ret <17 x float> %m
}
