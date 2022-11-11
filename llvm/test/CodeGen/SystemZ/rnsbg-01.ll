; Test sequences that can use RNSBG.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Test a simple mask, which is a wrap-around case.
define i32 @f1(i32 %a, i32 %b) {
; CHECK-LABEL: f1:
; CHECK: rnsbg %r2, %r3, 59, 56, 0
; CHECK: br %r14
  %orb = or i32 %b, 96
  %and = and i32 %a, %orb
  ret i32 %and
}

; ...and again with i64.
define i64 @f2(i64 %a, i64 %b) {
; CHECK-LABEL: f2:
; CHECK: rnsbg %r2, %r3, 59, 56, 0
; CHECK: br %r14
  %orb = or i64 %b, 96
  %and = and i64 %a, %orb
  ret i64 %and
}

; Test a case where no wraparound is needed.
define i32 @f3(i32 %a, i32 %b) {
; CHECK-LABEL: f3:
; CHECK: rnsbg %r2, %r3, 58, 61, 0
; CHECK: br %r14
  %orb = or i32 %b, -61
  %and = and i32 %a, %orb
  ret i32 %and
}

; ...and again with i64.
define i64 @f4(i64 %a, i64 %b) {
; CHECK-LABEL: f4:
; CHECK: rnsbg %r2, %r3, 58, 61, 0
; CHECK: br %r14
  %orb = or i64 %b, -61
  %and = and i64 %a, %orb
  ret i64 %and
}

; Test a case with just a left shift.  This can't use RNSBG.
define i32 @f6(i32 %a, i32 %b) {
; CHECK-LABEL: f6:
; CHECK: sll {{%r[0-5]}}
; CHECK: nr {{%r[0-5]}}
; CHECK: br %r14
  %shrb = shl i32 %b, 20
  %and = and i32 %a, %shrb
  ret i32 %and
}

; ...and again with i64.
define i64 @f7(i64 %a, i64 %b) {
; CHECK-LABEL: f7:
; CHECK: sllg {{%r[0-5]}}
; CHECK: ngr {{%r[0-5]}}
; CHECK: br %r14
  %shrb = shl i64 %b, 20
  %and = and i64 %a, %shrb
  ret i64 %and
}

; Test a case with just a rotate.  This can't use RNSBG.
define i32 @f8(i32 %a, i32 %b) {
; CHECK-LABEL: f8:
; CHECK: rll {{%r[0-5]}}
; CHECK: nr {{%r[0-5]}}
; CHECK: br %r14
  %shlb = shl i32 %b, 22
  %shrb = lshr i32 %b, 10
  %rotlb = or i32 %shlb, %shrb
  %and = and i32 %a, %rotlb
  ret i32 %and
}

; ...and again with i64, which can.
define i64 @f9(i64 %a, i64 %b) {
; CHECK-LABEL: f9:
; CHECK: rnsbg %r2, %r3, 0, 63, 44
; CHECK: br %r14
  %shlb = shl i64 %b, 44
  %shrb = lshr i64 %b, 20
  %rotlb = or i64 %shlb, %shrb
  %and = and i64 %a, %rotlb
  ret i64 %and
}

; Test a case with a left shift and OR, where the OR covers all shifted bits.
; We can do the whole thing using RNSBG.
define i32 @f10(i32 %a, i32 %b) {
; CHECK-LABEL: f10:
; CHECK: rnsbg %r2, %r3, 32, 56, 7
; CHECK: br %r14
  %shlb = shl i32 %b, 7
  %orb = or i32 %shlb, 127
  %and = and i32 %a, %orb
  ret i32 %and
}

; ...and again with i64.
define i64 @f11(i64 %a, i64 %b) {
; CHECK-LABEL: f11:
; CHECK: rnsbg %r2, %r3, 0, 56, 7
; CHECK: br %r14
  %shlb = shl i64 %b, 7
  %orb = or i64 %shlb, 127
  %and = and i64 %a, %orb
  ret i64 %and
}

; Test a case with a left shift and OR, where the OR doesn't cover all
; shifted bits.  We can't use RNSBG for the shift, but we can for the OR
; and AND.
define i32 @f12(i32 %a, i32 %b) {
; CHECK-LABEL: f12:
; CHECK: sll %r3, 7
; CHECK: rnsbg %r2, %r3, 32, 57, 0
; CHECK: br %r14
  %shlb = shl i32 %b, 7
  %orb = or i32 %shlb, 63
  %and = and i32 %a, %orb
  ret i32 %and
}

; ...and again with i64.
define i64 @f13(i64 %a, i64 %b) {
; CHECK-LABEL: f13:
; CHECK: sllg [[REG:%r[01345]]], %r3, 7
; CHECK: rnsbg %r2, [[REG]], 0, 57, 0
; CHECK: br %r14
  %shlb = shl i64 %b, 7
  %orb = or i64 %shlb, 63
  %and = and i64 %a, %orb
  ret i64 %and
}

; Test a case with a right shift and OR, where the OR covers all the shifted
; bits.  The whole thing can be done using RNSBG.
define i32 @f14(i32 %a, i32 %b) {
; CHECK-LABEL: f14:
; CHECK: rnsbg %r2, %r3, 60, 63, 37
; CHECK: br %r14
  %shrb = lshr i32 %b, 27
  %orb = or i32 %shrb, -16
  %and = and i32 %a, %orb
  ret i32 %and
}

; ...and again with i64.
define i64 @f15(i64 %a, i64 %b) {
; CHECK-LABEL: f15:
; CHECK: rnsbg %r2, %r3, 60, 63, 5
; CHECK: br %r14
  %shrb = lshr i64 %b, 59
  %orb = or i64 %shrb, -16
  %and = and i64 %a, %orb
  ret i64 %and
}

; Test a case with a right shift and OR, where the OR doesn't cover all the
; shifted bits.  The shift needs to be done separately, but the OR and AND
; can use RNSBG.
define i32 @f16(i32 %a, i32 %b) {
; CHECK-LABEL: f16:
; CHECK: srl %r3, 29
; CHECK: rnsbg %r2, %r3, 60, 63, 0
; CHECK: br %r14
  %shrb = lshr i32 %b, 29
  %orb = or i32 %shrb, -16
  %and = and i32 %a, %orb
  ret i32 %and
}

; ...and again with i64.
define i64 @f17(i64 %a, i64 %b) {
; CHECK-LABEL: f17:
; CHECK: srlg [[REG:%r[01345]]], %r3, 61
; CHECK: rnsbg %r2, [[REG]], 60, 63, 0
; CHECK: br %r14
  %shrb = lshr i64 %b, 61
  %orb = or i64 %shrb, -16
  %and = and i64 %a, %orb
  ret i64 %and
}

; Test a combination involving an ASHR in which the sign bits matter.
; We can't use RNSBG for the ASHR in that case, but we can for the rest.
define i32 @f18(i32 %a, i32 %b, ptr %dest) {
; CHECK-LABEL: f18:
; CHECK: sra %r3, 4
; CHECK: rnsbg %r2, %r3, 32, 62, 1
; CHECK: br %r14
  %ashrb = ashr i32 %b, 4
  store i32 %ashrb, ptr %dest
  %shlb = shl i32 %ashrb, 1
  %orb = or i32 %shlb, 1
  %and = and i32 %a, %orb
  ret i32 %and
}

; ...and again with i64.
define i64 @f19(i64 %a, i64 %b, ptr %dest) {
; CHECK-LABEL: f19:
; CHECK: srag [[REG:%r[0145]]], %r3, 34
; CHECK: rnsbg %r2, [[REG]], 0, 62, 1
; CHECK: br %r14
  %ashrb = ashr i64 %b, 34
  store i64 %ashrb, ptr %dest
  %shlb = shl i64 %ashrb, 1
  %orb = or i64 %shlb, 1
  %and = and i64 %a, %orb
  ret i64 %and
}

; Test a combination involving an ASHR in which the sign bits don't matter.
define i32 @f20(i32 %a, i32 %b, ptr %dest) {
; CHECK-LABEL: f20:
; CHECK: rnsbg %r2, %r3, 48, 62, 48
; CHECK: br %r14
  %ashrb = ashr i32 %b, 17
  store i32 %ashrb, ptr %dest
  %shlb = shl i32 %ashrb, 1
  %orb = or i32 %shlb, -65535
  %and = and i32 %a, %orb
  ret i32 %and
}

; ...and again with i64.
define i64 @f21(i64 %a, i64 %b, ptr %dest) {
; CHECK-LABEL: f21:
; CHECK: rnsbg %r2, %r3, 48, 62, 16
; CHECK: br %r14
  %ashrb = ashr i64 %b, 49
  store i64 %ashrb, ptr %dest
  %shlb = shl i64 %ashrb, 1
  %orb = or i64 %shlb, -65535
  %and = and i64 %a, %orb
  ret i64 %and
}

; Test a case with a shift, OR, and rotate where the OR covers all shifted bits.
define i64 @f22(i64 %a, i64 %b) {
; CHECK-LABEL: f22:
; CHECK: rnsbg %r2, %r3, 60, 54, 9
; CHECK: br %r14
  %shlb = shl i64 %b, 5
  %orb = or i64 %shlb, 31
  %shlorb = shl i64 %orb, 4
  %shrorb = lshr i64 %orb, 60
  %rotlorb = or i64 %shlorb, %shrorb
  %and = and i64 %a, %rotlorb
  ret i64 %and
}

; Check the handling of zext and AND, which isn't suitable for RNSBG.
define i64 @f23(i64 %a, i32 %b) {
; CHECK-LABEL: f23:
; CHECK-NOT: rnsbg
; CHECK: br %r14
  %add = add i32 %b, 1
  %ext = zext i32 %add to i64
  %and = and i64 %a, %ext
  ret i64 %and
}
