; NOTE:
;  - This test assumes your backend prints the final real instructions
;    as "ctub/ctuh/ctuw/ctsb/ctsh/ctsw" (not the pseudos).
;  - If your assembly still shows pseudos, change the CHECKs to PSEUDO_*.
;
; RUN: llc -mtriple=riscv32 -O2 %s -o - | FileCheck %s

target triple = "riscv32"

; -----------------------------------------------------------------------------
; ZERO_EXTEND tests (your lowering sets tag based on *result width* (W) + signed flag)
; For zext to i32 => W=32 => Kind=2 => ctuw
; -----------------------------------------------------------------------------

define i32 @test_zext_i8_to_i32(i8 %x) {
; CHECK-LABEL: test_zext_i8_to_i32:
; CHECK: ctuw
  %y = zext i8 %x to i32
  ret i32 %y
}

define i32 @test_zext_i16_to_i32(i16 %x) {
; CHECK-LABEL: test_zext_i16_to_i32:
; CHECK: ctuw
  %y = zext i16 %x to i32
  ret i32 %y
}

; -----------------------------------------------------------------------------
; SIGN_EXTEND tests
; sext to i32 => W=32 => Kind=5 => ctsw
; -----------------------------------------------------------------------------

define i32 @test_sext_i8_to_i32(i8 %x) {
; CHECK-LABEL: test_sext_i8_to_i32:
; CHECK: ctsw
  %y = sext i8 %x to i32
  ret i32 %y
}

define i32 @test_sext_i16_to_i32(i16 %x) {
; CHECK-LABEL: test_sext_i16_to_i32:
; CHECK: ctsw
  %y = sext i16 %x to i32
  ret i32 %y
}

; -----------------------------------------------------------------------------
; TRUNCATE tests
; trunc to i8  => W=8  => Kind=0 => ctub
; trunc to i16 => W=16 => Kind=1 => ctuh
; -----------------------------------------------------------------------------

define i8 @test_trunc_i32_to_i8(i32 %x) {
; CHECK-LABEL: test_trunc_i32_to_i8:
; CHECK: ctub
  %y = trunc i32 %x to i8
  ret i8 %y
}

define i16 @test_trunc_i32_to_i16(i32 %x) {
; CHECK-LABEL: test_trunc_i32_to_i16:
; CHECK: ctuh
  %y = trunc i32 %x to i16
  ret i16 %y
}

; -----------------------------------------------------------------------------
; SIGN_EXTEND_INREG tests
; We build the classic sign-extend-within-register idioms:
;   sext_inreg i8  in i32: (x << 24) >> 24  => should become SIGN_EXTEND_INREG from i8
;   sext_inreg i16 in i32: (x << 16) >> 16  => should become SIGN_EXTEND_INREG from i16
;
; Your SIGN_EXTEND_INREG lowering uses FromVT bits to pick Kind:
;   FromBits=8  => Kind=3 => ctsb
;   FromBits=16 => Kind=4 => ctsh
; -----------------------------------------------------------------------------

define i32 @test_sext_inreg_8(i32 %x) {
; CHECK-LABEL: test_sext_inreg_8:
; CHECK: ctsb
  %shl = shl i32 %x, 24
  %shr = ashr i32 %shl, 24
  ret i32 %shr
}

define i32 @test_sext_inreg_16(i32 %x) {
; CHECK-LABEL: test_sext_inreg_16:
; CHECK: ctsh
  %shl = shl i32 %x, 16
  %shr = ashr i32 %shl, 16
  ret i32 %shr
}