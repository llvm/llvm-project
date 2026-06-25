; RUN: llc -O2 -mtriple=hexagon < %s | FileCheck %s

; Test that we optimize sext + mul pattern to use vmpyh instruction
; instead of expanding to scalar multiplies.

; CHECK-LABEL: test_sext_mul_v2i16:
; CHECK: vmpyh
; CHECK-NOT: vsxthw
; CHECK-NOT: mpyi
define <2 x i32> @test_sext_mul_v2i16(<2 x i16> %a, <2 x i16> %b) {
entry:
  %ext_a = sext <2 x i16> %a to <2 x i32>
  %ext_b = sext <2 x i16> %b to <2 x i32>
  %mul = mul nsw <2 x i32> %ext_a, %ext_b
  ret <2 x i32> %mul
}
