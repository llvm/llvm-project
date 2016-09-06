; Test removal of AND operations that don't affect last 6 bits of shift amount
; operand.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Test that AND is not removed when some lower 6 bits are not set.
define i32 @f1(i32 %a, i32 %sh) {
; CHECK-LABEL: f1:
; CHECK: nil{{[lf]}} %r3, 31
; CHECK: sll %r2, 0(%r3)
  %and = and i32 %sh, 31
  %shift = shl i32 %a, %and
  ret i32 %shift
}

; Test removal of AND mask with only bottom 6 bits set.
define i32 @f2(i32 %a, i32 %sh) {
; CHECK-LABEL: f2:
; CHECK-NOT: nil{{[lf]}} %r3, 63
; CHECK: sll %r2, 0(%r3)
  %and = and i32 %sh, 63
  %shift = shl i32 %a, %and
  ret i32 %shift
}

; Test removal of AND mask including but not limited to bottom 6 bits.
define i32 @f3(i32 %a, i32 %sh) {
; CHECK-LABEL: f3:
; CHECK-NOT: nil{{[lf]}} %r3, 255
; CHECK: sll %r2, 0(%r3)
  %and = and i32 %sh, 255
  %shift = shl i32 %a, %and
  ret i32 %shift
}

; Test removal of AND mask from SRA.
define i32 @f4(i32 %a, i32 %sh) {
; CHECK-LABEL: f4:
; CHECK-NOT: nil{{[lf]}} %r3, 63
; CHECK: sra %r2, 0(%r3)
  %and = and i32 %sh, 63
  %shift = ashr i32 %a, %and
  ret i32 %shift
}

; Test removal of AND mask from SRL.
define i32 @f5(i32 %a, i32 %sh) {
; CHECK-LABEL: f5:
; CHECK-NOT: nil{{[lf]}} %r3, 63
; CHECK: srl %r2, 0(%r3)
  %and = and i32 %sh, 63
  %shift = lshr i32 %a, %and
  ret i32 %shift
}

; Test removal of AND mask from SLLG.
define i64 @f6(i64 %a, i64 %sh) {
; CHECK-LABEL: f6:
; CHECK-NOT: nil{{[lf]}} %r3, 63
; CHECK: sllg %r2, %r2, 0(%r3)
  %and = and i64 %sh, 63
  %shift = shl i64 %a, %and
  ret i64 %shift
}

; Test removal of AND mask from SRAG.
define i64 @f7(i64 %a, i64 %sh) {
; CHECK-LABEL: f7:
; CHECK-NOT: nil{{[lf]}} %r3, 63
; CHECK: srag %r2, %r2, 0(%r3)
  %and = and i64 %sh, 63
  %shift = ashr i64 %a, %and
  ret i64 %shift
}

; Test removal of AND mask from SRLG.
define i64 @f8(i64 %a, i64 %sh) {
; CHECK-LABEL: f8:
; CHECK-NOT: nil{{[lf]}} %r3, 63
; CHECK: srlg %r2, %r2, 0(%r3)
  %and = and i64 %sh, 63
  %shift = lshr i64 %a, %and
  ret i64 %shift
}

; Test that AND with two register operands is not affected.
define i32 @f9(i32 %a, i32 %b, i32 %sh) {
; CHECK-LABEL: f9:
; CHECK: nr %r3, %r4
; CHECK: sll %r2, 0(%r3)
  %and = and i32 %sh, %b
  %shift = shl i32 %a, %and
  ret i32 %shift
}

; Test that AND is not entirely removed if the result is reused.
define i32 @f10(i32 %a, i32 %sh) {
; CHECK-LABEL: f10:
; CHECK: sll %r2, 0(%r3)
; CHECK: nil{{[lf]}} %r3, 63
; CHECK: ar %r2, %r3
  %and = and i32 %sh, 63
  %shift = shl i32 %a, %and
  %reuse = add i32 %and, %shift
  ret i32 %reuse
}
