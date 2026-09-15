; Verify if materialization is not kicking in for big-endian targets
; RUN: llc -mtriple=aarch64_be-linux-gnu -o - %s | FileCheck %s

define <2 x i32> @test_const_v2i32_big_endian() {
; CHECK-LABEL: test_const_v2i32_big_endian:
; CHECK:      ldr     d0, [x8, :lo12:.LCPI0_0]
; CHECK-NOT:  mov
; CHECK:      ret
  ret <2 x i32> <i32 1, i32 2>
}

define <4 x i16> @test_const_v4i16_big_endian() {
; CHECK-LABEL: test_const_v4i16_big_endian:
; CHECK:      ldr     d0, [x8, :lo12:.LCPI1_0]
; CHECK-NOT:  mov
; CHECK:      ret
  ret <4 x i16> <i16 1, i16 2, i16 3, i16 4>
}

define <8 x i8> @test_const_v8i8_big_endian() {
; CHECK-LABEL: test_const_v8i8_big_endian:
; CHECK:      ldr     d0, [x8, :lo12:.LCPI2_0]
; CHECK-NOT:  mov
; CHECK:      ret
  ret <8 x i8> <i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8>
}

