; Test removal of AND operations that don't affect last 6 bits of rotate amount
; operand.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Test that AND is not removed when some lower 6 bits are not set.
define i32 @f1(i32 %val, i32 %amt) {
; CHECK-LABEL: f1:
; CHECK: nil{{[lf]}} %r3, 31
; CHECK: rll %r2, %r2, 0(%r3)
  %and = and i32 %amt, 31

  %inv = sub i32 32, %and
  %parta = shl i32 %val, %and
  %partb = lshr i32 %val, %inv

  %rotl = or i32 %parta, %partb

  ret i32 %rotl
}

; Test removal of AND mask with only bottom 6 bits set.
define i32 @f2(i32 %val, i32 %amt) {
; CHECK-LABEL: f2:
; CHECK-NOT: nil{{[lf]}} %r3, 63
; CHECK: rll %r2, %r2, 0(%r3)
  %and = and i32 %amt, 63

  %inv = sub i32 32, %and
  %parta = shl i32 %val, %and
  %partb = lshr i32 %val, %inv

  %rotl = or i32 %parta, %partb

  ret i32 %rotl
}

; Test removal of AND mask including but not limited to bottom 6 bits.
define i32 @f3(i32 %val, i32 %amt) {
; CHECK-LABEL: f3:
; CHECK-NOT: nil{{[lf]}} %r3, 255
; CHECK: rll %r2, %r2, 0(%r3)
  %and = and i32 %amt, 255

  %inv = sub i32 32, %and
  %parta = shl i32 %val, %and
  %partb = lshr i32 %val, %inv

  %rotl = or i32 %parta, %partb

  ret i32 %rotl
}

; Test removal of AND mask from RLLG.
define i64 @f4(i64 %val, i64 %amt) {
; CHECK-LABEL: f4:
; CHECK-NOT: nil{{[lf]}} %r3, 63
; CHECK: rllg %r2, %r2, 0(%r3)
  %and = and i64 %amt, 63

  %inv = sub i64 64, %and
  %parta = shl i64 %val, %and
  %partb = lshr i64 %val, %inv

  %rotl = or i64 %parta, %partb

  ret i64 %rotl
}

; Test that AND is not entirely removed if the result is reused.
define i32 @f5(i32 %val, i32 %amt) {
; CHECK-LABEL: f5:
; CHECK: rll %r2, %r2, 0(%r3)
; CHECK: nil{{[lf]}} %r3, 63
; CHECK: ar %r2, %r3
  %and = and i32 %amt, 63

  %inv = sub i32 32, %and
  %parta = shl i32 %val, %and
  %partb = lshr i32 %val, %inv

  %rotl = or i32 %parta, %partb

  %reuse = add i32 %and, %rotl
  ret i32 %reuse
}
