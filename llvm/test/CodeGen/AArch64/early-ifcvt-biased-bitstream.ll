; RUN: llc -mcpu=apple-m1 -verify-machineinstrs < %s | FileCheck %s
;
; Reading zero bits must not add speculative shifts and selects to the
; loop-carried consumed-bits dependency when the branch is strongly biased.

target triple = "aarch64-unknown-linux-gnu"

; CHECK-LABEL: decode_bits:
; CHECK-NOT: csel
; CHECK: cbnz
; CHECK-NOT: csel
; CHECK: ret
define i32 @decode_bits(ptr %counts, i64 %count, i64 %bits, i32 %initial, ptr %out) {
entry:
  br label %loop
loop:
  %i = phi i64 [0, %entry], [%nexti, %merge]
  %consumed = phi i32 [%initial, %entry], [%nextconsumed, %merge]
  %p = getelementptr i8, ptr %counts, i64 %i
  %n8 = load i8, ptr %p
  %zero = icmp eq i8 %n8, 0
  br i1 %zero, label %merge, label %read, !prof !0
read:
  %n = zext i8 %n8 to i32
  %start = and i32 %consumed, 63
  %start64 = zext i32 %start to i64
  %shifted = shl i64 %bits, %start64
  %neg = sub i32 0, %n
  %shift = and i32 %neg, 63
  %shift64 = zext i32 %shift to i64
  %value = lshr i64 %shifted, %shift64
  %added = add i32 %consumed, %n
  br label %merge
merge:
  %result = phi i64 [0, %loop], [%value, %read]
  %nextconsumed = phi i32 [%consumed, %loop], [%added, %read]
  %op = getelementptr i64, ptr %out, i64 %i
  store i64 %result, ptr %op
  %nexti = add i64 %i, 1
  %done = icmp eq i64 %nexti, %count
  br i1 %done, label %exit, label %loop
exit:
  ret i32 %nextconsumed
}

; CHECK-LABEL: decode_bits_balanced:
; CHECK: csel
; CHECK: csel
; CHECK: ret
define i32 @decode_bits_balanced(ptr %counts, i64 %count, i64 %bits, i32 %initial, ptr %out) {
entry:
  br label %loop
loop:
  %i = phi i64 [0, %entry], [%nexti, %merge]
  %consumed = phi i32 [%initial, %entry], [%nextconsumed, %merge]
  %p = getelementptr i8, ptr %counts, i64 %i
  %n8 = load i8, ptr %p
  %zero = icmp eq i8 %n8, 0
  br i1 %zero, label %merge, label %read, !prof !1
read:
  %n = zext i8 %n8 to i32
  %start = and i32 %consumed, 63
  %start64 = zext i32 %start to i64
  %shifted = shl i64 %bits, %start64
  %neg = sub i32 0, %n
  %shift = and i32 %neg, 63
  %shift64 = zext i32 %shift to i64
  %value = lshr i64 %shifted, %shift64
  %added = add i32 %consumed, %n
  br label %merge
merge:
  %result = phi i64 [0, %loop], [%value, %read]
  %nextconsumed = phi i32 [%consumed, %loop], [%added, %read]
  %op = getelementptr i64, ptr %out, i64 %i
  store i64 %result, ptr %op
  %nexti = add i64 %i, 1
  %done = icmp eq i64 %nexti, %count
  br i1 %done, label %exit, label %loop
exit:
  ret i32 %nextconsumed
}

!0 = !{!"branch_weights", i32 2000, i32 1}
!1 = !{!"branch_weights", i32 1, i32 1}
