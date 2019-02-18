; Test 32-bit GPR loads.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check the low end of the L range.
define i32 @f1(i32 *%src) {
; CHECK-LABEL: f1:
; CHECK: l %r2, 0(%r2)
; CHECK: br %r14
  %val = load i32, i32 *%src
  ret i32 %val
}

; Check the high end of the aligned L range.
define i32 @f2(i32 *%src) {
; CHECK-LABEL: f2:
; CHECK: l %r2, 4092(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 1023
  %val = load i32, i32 *%ptr
  ret i32 %val
}

; Check the next word up, which should use LY instead of L.
define i32 @f3(i32 *%src) {
; CHECK-LABEL: f3:
; CHECK: ly %r2, 4096(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 1024
  %val = load i32, i32 *%ptr
  ret i32 %val
}

; Check the high end of the aligned LY range.
define i32 @f4(i32 *%src) {
; CHECK-LABEL: f4:
; CHECK: ly %r2, 524284(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 131071
  %val = load i32, i32 *%ptr
  ret i32 %val
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i32 @f5(i32 *%src) {
; CHECK-LABEL: f5:
; CHECK: agfi %r2, 524288
; CHECK: l %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 131072
  %val = load i32, i32 *%ptr
  ret i32 %val
}

; Check the high end of the negative aligned LY range.
define i32 @f6(i32 *%src) {
; CHECK-LABEL: f6:
; CHECK: ly %r2, -4(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 -1
  %val = load i32, i32 *%ptr
  ret i32 %val
}

; Check the low end of the LY range.
define i32 @f7(i32 *%src) {
; CHECK-LABEL: f7:
; CHECK: ly %r2, -524288(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 -131072
  %val = load i32, i32 *%ptr
  ret i32 %val
}

; Check the next word down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i32 @f8(i32 *%src) {
; CHECK-LABEL: f8:
; CHECK: agfi %r2, -524292
; CHECK: l %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 -131073
  %val = load i32, i32 *%ptr
  ret i32 %val
}

; Check that L allows an index.
define i32 @f9(i64 %src, i64 %index) {
; CHECK-LABEL: f9:
; CHECK: l %r2, 4095({{%r3,%r2|%r2,%r3}})
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4095
  %ptr = inttoptr i64 %add2 to i32 *
  %val = load i32, i32 *%ptr
  ret i32 %val
}

; Check that LY allows an index.
define i32 @f10(i64 %src, i64 %index) {
; CHECK-LABEL: f10:
; CHECK: ly %r2, 4096({{%r3,%r2|%r2,%r3}})
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4096
  %ptr = inttoptr i64 %add2 to i32 *
  %val = load i32, i32 *%ptr
  ret i32 %val
}
