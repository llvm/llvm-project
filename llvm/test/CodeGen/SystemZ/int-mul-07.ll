; Test high-part i32->i64 multiplications.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; We don't provide *MUL_LOHI or MULH* for the patterns in this file,
; but they should at least still work.

; Check zero-extended multiplication in which only the high part is used.
define i32 @f1(i32 %a, i32 %b) {
; CHECK-LABEL: f1:
; CHECK: msgr
; CHECK: br %r14
  %ax = zext i32 %a to i64
  %bx = zext i32 %b to i64
  %mulx = mul i64 %ax, %bx
  %highx = lshr i64 %mulx, 32
  %high = trunc i64 %highx to i32
  ret i32 %high
}

; Check sign-extended multiplication in which only the high part is used.
define i32 @f2(i32 %a, i32 %b) {
; CHECK-LABEL: f2:
; CHECK: msgfr
; CHECK: br %r14
  %ax = sext i32 %a to i64
  %bx = sext i32 %b to i64
  %mulx = mul i64 %ax, %bx
  %highx = lshr i64 %mulx, 32
  %high = trunc i64 %highx to i32
  ret i32 %high
}

; Check zero-extended multiplication in which the result is split into
; high and low halves.
define i32 @f3(i32 %a, i32 %b) {
; CHECK-LABEL: f3:
; CHECK: msgr
; CHECK: br %r14
  %ax = zext i32 %a to i64
  %bx = zext i32 %b to i64
  %mulx = mul i64 %ax, %bx
  %highx = lshr i64 %mulx, 32
  %high = trunc i64 %highx to i32
  %low = trunc i64 %mulx to i32
  %or = or i32 %high, %low
  ret i32 %or
}

; Check sign-extended multiplication in which the result is split into
; high and low halves.
define i32 @f4(i32 %a, i32 %b) {
; CHECK-LABEL: f4:
; CHECK: msgfr
; CHECK: br %r14
  %ax = sext i32 %a to i64
  %bx = sext i32 %b to i64
  %mulx = mul i64 %ax, %bx
  %highx = lshr i64 %mulx, 32
  %high = trunc i64 %highx to i32
  %low = trunc i64 %mulx to i32
  %or = or i32 %high, %low
  ret i32 %or
}
