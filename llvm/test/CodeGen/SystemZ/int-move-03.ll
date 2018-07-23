; Test 64-bit GPR loads.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check LG with no displacement.
define i64 @f1(i64 *%src) {
; CHECK-LABEL: f1:
; CHECK: lg %r2, 0(%r2)
; CHECK: br %r14
  %val = load i64, i64 *%src
  ret i64 %val
}

; Check the high end of the aligned LG range.
define i64 @f2(i64 *%src) {
; CHECK-LABEL: f2:
; CHECK: lg %r2, 524280(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 65535
  %val = load i64, i64 *%ptr
  ret i64 %val
}

; Check the next doubleword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f3(i64 *%src) {
; CHECK-LABEL: f3:
; CHECK: agfi %r2, 524288
; CHECK: lg %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 65536
  %val = load i64, i64 *%ptr
  ret i64 %val
}

; Check the high end of the negative aligned LG range.
define i64 @f4(i64 *%src) {
; CHECK-LABEL: f4:
; CHECK: lg %r2, -8(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 -1
  %val = load i64, i64 *%ptr
  ret i64 %val
}

; Check the low end of the LG range.
define i64 @f5(i64 *%src) {
; CHECK-LABEL: f5:
; CHECK: lg %r2, -524288(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 -65536
  %val = load i64, i64 *%ptr
  ret i64 %val
}

; Check the next doubleword down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f6(i64 *%src) {
; CHECK-LABEL: f6:
; CHECK: agfi %r2, -524296
; CHECK: lg %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 -65537
  %val = load i64, i64 *%ptr
  ret i64 %val
}

; Check that LG allows an index.
define i64 @f7(i64 %src, i64 %index) {
; CHECK-LABEL: f7:
; CHECK: lg %r2, 524287({{%r3,%r2|%r2,%r3}})
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524287
  %ptr = inttoptr i64 %add2 to i64 *
  %val = load i64, i64 *%ptr
  ret i64 %val
}
