; RUN: llc < %s -mtriple=lanai-unknown-unknown | FileCheck %s

; Test left-shift i64 lowering does not result in call being inserted.

; CHECK-LABEL: shift
; CHECKT: bt __ashldi3
; CHECK: or	%r0, 0x0, %r[[T0:[0-9]+]]
; CHECK: mov	0x20, %r[[T1:[0-9]+]]
; CHECK: sub	%r[[T1]], %r[[ShAmt:[0-9]+]], %r[[T1]]
; CHECK: sub	%r0, %r[[T1]], %r[[T1]]
; CHECK: sh	%r[[ShOpB:[0-9]+]], %r[[T1]], %r[[T1]]
; CHECK: sub.f	%r[[ShAmt]], 0x0, %r0
; CHECK: sel.eq %r0, %r[[T1]], %r[[T1]]
; CHECK: sh	%r[[ShOpA:[0-9]+]], %r[[ShAmt]], %r[[T2:[0-9]+]]
; CHECK: or	%r[[T1]], %r[[T2]], %rv
; CHECK: sub.f	%r[[ShAmt]], 0x20, %r[[T1]]
; CHECK: sh.pl	%r[[ShOpB]], %r[[T1]], %rv
; CHECK: sh.mi	%r[[ShOpB]], %r[[ShAmt]], %r[[T0]]

define i64 @shift(i64 inreg, i32 inreg) {
  %3 = zext i32 %1 to i64
  %4 = shl i64 %0, %3
  ret i64 %4
}
