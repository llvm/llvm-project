; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=pwr9 -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr < %s | \
; RUN:   FileCheck %s

; These tests show that for 32-bit and 64-bit scalars, combining a shift to
; a single multiply-high is only valid when the shift amount is the same as
; the width of the narrow type.

; That is, combining a shift to mulh is only valid for 32-bit when the shift
; amount is 32.
; Likewise, combining a shift to mulh is only valid for 64-bit when the shift
; amount is 64.

define i32 @test_mulhw(i32 %a, i32 %b) {
; CHECK-LABEL: test_mulhw:
; CHECK:     mulld
; CHECK-NOT: mulhw
; CHECK:     blr
  %1 = sext i32 %a to i64
  %2 = sext i32 %b to i64
  %mul = mul i64 %1, %2
  %shr = lshr i64 %mul, 33
  %tr = trunc i64 %shr to i32
  ret i32 %tr
}

define i32 @test_mulhu(i32 %a, i32 %b) {
; CHECK-LABEL: test_mulhu:
; CHECK:     mulld
; CHECK-NOT: mulhwu
; CHECK:     blr
  %1 = zext i32 %a to i64
  %2 = zext i32 %b to i64
  %mul = mul i64 %1, %2
  %shr = lshr i64 %mul, 33
  %tr = trunc i64 %shr to i32
  ret i32 %tr
}

define i64 @test_mulhd(i64 %a, i64 %b) {
; CHECK-LABEL: test_mulhd:
; CHECK:    mulhd
; CHECK:    mulld
; CHECK:    blr
  %1 = sext i64 %a to i128
  %2 = sext i64 %b to i128
  %mul = mul i128 %1, %2
  %shr = lshr i128 %mul, 63
  %tr = trunc i128 %shr to i64
  ret i64 %tr
}

define i64 @test_mulhdu(i64 %a, i64 %b) {
; CHECK-LABEL: test_mulhdu:
; CHECK:    mulhdu
; CHECK:    mulld
; CHECK:    blr
  %1 = zext i64 %a to i128
  %2 = zext i64 %b to i128
  %mul = mul i128 %1, %2
  %shr = lshr i128 %mul, 63
  %tr = trunc i128 %shr to i64
  ret i64 %tr
}

define signext i32 @test_mulhw_signext(i32 %a, i32 %b) {
; CHECK-LABEL: test_mulhw_signext:
; CHECK:     mulld
; CHECK-NOT: mulhw
; CHECK:     blr
  %1 = sext i32 %a to i64
  %2 = sext i32 %b to i64
  %mul = mul i64 %1, %2
  %shr = lshr i64 %mul, 33
  %tr = trunc i64 %shr to i32
  ret i32 %tr
}

define zeroext i32 @test_mulhu_zeroext(i32 %a, i32 %b) {
; CHECK-LABEL: test_mulhu_zeroext:
; CHECK:     mulld
; CHECK-NOT: mulhwu
; CHECK:     blr
  %1 = zext i32 %a to i64
  %2 = zext i32 %b to i64
  %mul = mul i64 %1, %2
  %shr = lshr i64 %mul, 33
  %tr = trunc i64 %shr to i32
  ret i32 %tr
}

define signext i64 @test_mulhd_signext(i64 %a, i64 %b) {
; CHECK-LABEL: test_mulhd_signext:
; CHECK:    mulhd
; CHECK:    mulld
; CHECK:    blr
  %1 = sext i64 %a to i128
  %2 = sext i64 %b to i128
  %mul = mul i128 %1, %2
  %shr = lshr i128 %mul, 63
  %tr = trunc i128 %shr to i64
  ret i64 %tr
}

define zeroext i64 @test_mulhdu_zeroext(i64 %a, i64 %b) {
; CHECK-LABEL: test_mulhdu_zeroext:
; CHECK:    mulhdu
; CHECK:    mulld
; CHECK:    blr
  %1 = zext i64 %a to i128
  %2 = zext i64 %b to i128
  %mul = mul i128 %1, %2
  %shr = lshr i128 %mul, 63
  %tr = trunc i128 %shr to i64
  ret i64 %tr
}
