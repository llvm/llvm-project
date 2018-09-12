; Test signed high-part i64->i128 multiplications on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 -asm-verbose=0 | FileCheck %s

declare i64 @foo()

; Check sign-extended multiplication in which only the high part is used.
define i64 @f1(i64 %dummy, i64 %a, i64 %b) {
; CHECK-LABEL: f1:
; CHECK-NOT: {{%r[234]}}
; CHECK: mgrk %r2, %r3, %r4
; CHECK: br %r14
  %ax = sext i64 %a to i128
  %bx = sext i64 %b to i128
  %mulx = mul i128 %ax, %bx
  %highx = lshr i128 %mulx, 64
  %high = trunc i128 %highx to i64
  ret i64 %high
}

; Check sign-extended multiplication in which only part of the high half
; is used.
define i64 @f2(i64 %dummy, i64 %a, i64 %b) {
; CHECK-LABEL: f2:
; CHECK-NOT: {{%r[234]}}
; CHECK: mgrk [[REG:%r[0-9]+]], %r3, %r4
; CHECK: srlg %r2, [[REG]], 3
; CHECK: br %r14
  %ax = sext i64 %a to i128
  %bx = sext i64 %b to i128
  %mulx = mul i128 %ax, %bx
  %highx = lshr i128 %mulx, 67
  %high = trunc i128 %highx to i64
  ret i64 %high
}

; Check sign-extended multiplication in which the result is split into
; high and low halves.
define i64 @f3(i64 %dummy, i64 %a, i64 %b) {
; CHECK-LABEL: f3:
; CHECK-NOT: {{%r[234]}}
; CHECK: mgrk %r2, %r3, %r4
; CHECK: ogr %r2, %r3
; CHECK: br %r14
  %ax = sext i64 %a to i128
  %bx = sext i64 %b to i128
  %mulx = mul i128 %ax, %bx
  %highx = lshr i128 %mulx, 64
  %high = trunc i128 %highx to i64
  %low = trunc i128 %mulx to i64
  %or = or i64 %high, %low
  ret i64 %or
}

; Check MG with no displacement.
define i64 @f4(i64 %dummy, i64 %a, i64 *%src) {
; CHECK-LABEL: f4:
; CHECK-NOT: {{%r[234]}}
; CHECK: mg %r2, 0(%r4)
; CHECK: br %r14
  %b = load i64, i64 *%src
  %ax = sext i64 %a to i128
  %bx = sext i64 %b to i128
  %mulx = mul i128 %ax, %bx
  %highx = lshr i128 %mulx, 64
  %high = trunc i128 %highx to i64
  ret i64 %high
}

; Check the high end of the aligned MG range.
define i64 @f5(i64 %dummy, i64 %a, i64 *%src) {
; CHECK-LABEL: f5:
; CHECK: mg %r2, 524280(%r4)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 65535
  %b = load i64, i64 *%ptr
  %ax = sext i64 %a to i128
  %bx = sext i64 %b to i128
  %mulx = mul i128 %ax, %bx
  %highx = lshr i128 %mulx, 64
  %high = trunc i128 %highx to i64
  ret i64 %high
}

; Check the next doubleword up, which requires separate address logic.
; Other sequences besides this one would be OK.
define i64 @f6(i64 %dummy, i64 %a, i64 *%src) {
; CHECK-LABEL: f6:
; CHECK: agfi %r4, 524288
; CHECK: mg %r2, 0(%r4)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 65536
  %b = load i64, i64 *%ptr
  %ax = sext i64 %a to i128
  %bx = sext i64 %b to i128
  %mulx = mul i128 %ax, %bx
  %highx = lshr i128 %mulx, 64
  %high = trunc i128 %highx to i64
  ret i64 %high
}

; Check the high end of the negative aligned MG range.
define i64 @f7(i64 %dummy, i64 %a, i64 *%src) {
; CHECK-LABEL: f7:
; CHECK: mg %r2, -8(%r4)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 -1
  %b = load i64, i64 *%ptr
  %ax = sext i64 %a to i128
  %bx = sext i64 %b to i128
  %mulx = mul i128 %ax, %bx
  %highx = lshr i128 %mulx, 64
  %high = trunc i128 %highx to i64
  ret i64 %high
}

; Check the low end of the MG range.
define i64 @f8(i64 %dummy, i64 %a, i64 *%src) {
; CHECK-LABEL: f8:
; CHECK: mg %r2, -524288(%r4)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 -65536
  %b = load i64, i64 *%ptr
  %ax = sext i64 %a to i128
  %bx = sext i64 %b to i128
  %mulx = mul i128 %ax, %bx
  %highx = lshr i128 %mulx, 64
  %high = trunc i128 %highx to i64
  ret i64 %high
}

; Check the next doubleword down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f9(i64 *%dest, i64 %a, i64 *%src) {
; CHECK-LABEL: f9:
; CHECK: agfi %r4, -524296
; CHECK: mg %r2, 0(%r4)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 -65537
  %b = load i64, i64 *%ptr
  %ax = sext i64 %a to i128
  %bx = sext i64 %b to i128
  %mulx = mul i128 %ax, %bx
  %highx = lshr i128 %mulx, 64
  %high = trunc i128 %highx to i64
  ret i64 %high
}

; Check that MG allows an index.
define i64 @f10(i64 *%dest, i64 %a, i64 %src, i64 %index) {
; CHECK-LABEL: f10:
; CHECK: mg %r2, 524287(%r5,%r4)
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524287
  %ptr = inttoptr i64 %add2 to i64 *
  %b = load i64, i64 *%ptr
  %ax = sext i64 %a to i128
  %bx = sext i64 %b to i128
  %mulx = mul i128 %ax, %bx
  %highx = lshr i128 %mulx, 64
  %high = trunc i128 %highx to i64
  ret i64 %high
}

