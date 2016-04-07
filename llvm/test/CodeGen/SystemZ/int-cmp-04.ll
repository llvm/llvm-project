; Test 64-bit signed comparison in which the second operand is sign-extended
; from an i16 memory value.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check CGH with no displacement.
define void @f1(i64 %lhs, i16 *%src, i64 *%dst) {
; CHECK-LABEL: f1:
; CHECK: cgh %r2, 0(%r3)
; CHECK: br %r14
  %half = load i16 , i16 *%src
  %rhs = sext i16 %half to i64
  %cond = icmp slt i64 %lhs, %rhs
  %res = select i1 %cond, i64 100, i64 200
  store i64 %res, i64 *%dst
  ret void
}

; Check the high end of the aligned CGH range.
define void @f2(i64 %lhs, i16 *%src, i64 *%dst) {
; CHECK-LABEL: f2:
; CHECK: cgh %r2, 524286(%r3)
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%src, i64 262143
  %half = load i16 , i16 *%ptr
  %rhs = sext i16 %half to i64
  %cond = icmp slt i64 %lhs, %rhs
  %res = select i1 %cond, i64 100, i64 200
  store i64 %res, i64 *%dst
  ret void
}

; Check the next halfword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f3(i64 %lhs, i16 *%src, i64 *%dst) {
; CHECK-LABEL: f3:
; CHECK: agfi %r3, 524288
; CHECK: cgh %r2, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%src, i64 262144
  %half = load i16 , i16 *%ptr
  %rhs = sext i16 %half to i64
  %cond = icmp slt i64 %lhs, %rhs
  %res = select i1 %cond, i64 100, i64 200
  store i64 %res, i64 *%dst
  ret void
}

; Check the high end of the negative aligned CGH range.
define void @f4(i64 %lhs, i16 *%src, i64 *%dst) {
; CHECK-LABEL: f4:
; CHECK: cgh %r2, -2(%r3)
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%src, i64 -1
  %half = load i16 , i16 *%ptr
  %rhs = sext i16 %half to i64
  %cond = icmp slt i64 %lhs, %rhs
  %res = select i1 %cond, i64 100, i64 200
  store i64 %res, i64 *%dst
  ret void
}

; Check the low end of the CGH range.
define void @f5(i64 %lhs, i16 *%src, i64 *%dst) {
; CHECK-LABEL: f5:
; CHECK: cgh %r2, -524288(%r3)
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%src, i64 -262144
  %half = load i16 , i16 *%ptr
  %rhs = sext i16 %half to i64
  %cond = icmp slt i64 %lhs, %rhs
  %res = select i1 %cond, i64 100, i64 200
  store i64 %res, i64 *%dst
  ret void
}

; Check the next halfword down, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f6(i64 %lhs, i16 *%src, i64 *%dst) {
; CHECK-LABEL: f6:
; CHECK: agfi %r3, -524290
; CHECK: cgh %r2, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%src, i64 -262145
  %half = load i16 , i16 *%ptr
  %rhs = sext i16 %half to i64
  %cond = icmp slt i64 %lhs, %rhs
  %res = select i1 %cond, i64 100, i64 200
  store i64 %res, i64 *%dst
  ret void
}

; Check that CGH allows an index.
define void @f7(i64 %lhs, i64 %base, i64 %index, i64 *%dst) {
; CHECK-LABEL: f7:
; CHECK: cgh %r2, 4096({{%r4,%r3|%r3,%r4}})
; CHECK: br %r14
  %add1 = add i64 %base, %index
  %add2 = add i64 %add1, 4096
  %ptr = inttoptr i64 %add2 to i16 *
  %half = load i16 , i16 *%ptr
  %rhs = sext i16 %half to i64
  %cond = icmp slt i64 %lhs, %rhs
  %res = select i1 %cond, i64 100, i64 200
  store i64 %res, i64 *%dst
  ret void
}

; Check the comparison can be reversed if that allows CGH to be used.
define double @f8(double %a, double %b, i64 %rhs, i16 *%src) {
; CHECK-LABEL: f8:
; CHECK: cgh %r2, 0(%r3)
; CHECK-NEXT: bhr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %half = load i16 , i16 *%src
  %lhs = sext i16 %half to i64
  %cond = icmp slt i64 %lhs, %rhs
  %res = select i1 %cond, double %a, double %b
  ret double %res
}
