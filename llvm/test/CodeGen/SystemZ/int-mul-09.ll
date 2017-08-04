; Test multiplications between an i64 and a sign-extended i16 on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

declare i64 @foo()

; Check MGH with no displacement.
define i64 @f1(i64 %a, i16 *%src) {
; CHECK-LABEL: f1:
; CHECK: mgh %r2, 0(%r3)
; CHECK: br %r14
  %b = load i16, i16 *%src
  %bext = sext i16 %b to i64
  %mul = mul i64 %a, %bext
  ret i64 %mul
}

; Check the high end of the aligned MGH range.
define i64 @f2(i64 %a, i16 *%src) {
; CHECK-LABEL: f2:
; CHECK: mgh %r2, 524286(%r3)
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%src, i64 262143
  %b = load i16, i16 *%ptr
  %bext = sext i16 %b to i64
  %mul = mul i64 %a, %bext
  ret i64 %mul
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f3(i64 %a, i16 *%src) {
; CHECK-LABEL: f3:
; CHECK: agfi %r3, 524288
; CHECK: mgh %r2, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%src, i64 262144
  %b = load i16, i16 *%ptr
  %bext = sext i16 %b to i64
  %mul = mul i64 %a, %bext
  ret i64 %mul
}

; Check the high end of the negative aligned MGH range.
define i64 @f4(i64 %a, i16 *%src) {
; CHECK-LABEL: f4:
; CHECK: mgh %r2, -2(%r3)
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%src, i64 -1
  %b = load i16, i16 *%ptr
  %bext = sext i16 %b to i64
  %mul = mul i64 %a, %bext
  ret i64 %mul
}

; Check the low end of the MGH range.
define i64 @f5(i64 %a, i16 *%src) {
; CHECK-LABEL: f5:
; CHECK: mgh %r2, -524288(%r3)
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%src, i64 -262144
  %b = load i16, i16 *%ptr
  %bext = sext i16 %b to i64
  %mul = mul i64 %a, %bext
  ret i64 %mul
}

; Check the next word down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f6(i64 %a, i16 *%src) {
; CHECK-LABEL: f6:
; CHECK: agfi %r3, -524290
; CHECK: mgh %r2, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%src, i64 -262145
  %b = load i16, i16 *%ptr
  %bext = sext i16 %b to i64
  %mul = mul i64 %a, %bext
  ret i64 %mul
}

; Check that MGH allows an index.
define i64 @f7(i64 %a, i64 %src, i64 %index) {
; CHECK-LABEL: f7:
; CHECK: mgh %r2, 524284({{%r4,%r3|%r3,%r4}})
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524284
  %ptr = inttoptr i64 %add2 to i16 *
  %b = load i16, i16 *%ptr
  %bext = sext i16 %b to i64
  %mul = mul i64 %a, %bext
  ret i64 %mul
}

