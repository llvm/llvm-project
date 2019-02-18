; Test 32-bit signed comparison in which the second operand is sign-extended
; from an i16 memory value.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check the low end of the CH range.
define void @f1(i32 %lhs, i16 *%src, i32 *%dst) {
; CHECK-LABEL: f1:
; CHECK: ch %r2, 0(%r3)
; CHECK: br %r14
  %half = load i16, i16 *%src
  %rhs = sext i16 %half to i32
  %cond = icmp slt i32 %lhs, %rhs
  %res = select i1 %cond, i32 100, i32 200
  store i32 %res, i32 *%dst
  ret void
}

; Check the high end of the aligned CH range.
define void @f2(i32 %lhs, i16 *%src, i32 *%dst) {
; CHECK-LABEL: f2:
; CHECK: ch %r2, 4094(%r3)
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%src, i64 2047
  %half = load i16, i16 *%ptr
  %rhs = sext i16 %half to i32
  %cond = icmp slt i32 %lhs, %rhs
  %res = select i1 %cond, i32 100, i32 200
  store i32 %res, i32 *%dst
  ret void
}

; Check the next halfword up, which should use CHY instead of CH.
define void @f3(i32 %lhs, i16 *%src, i32 *%dst) {
; CHECK-LABEL: f3:
; CHECK: chy %r2, 4096(%r3)
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%src, i64 2048
  %half = load i16, i16 *%ptr
  %rhs = sext i16 %half to i32
  %cond = icmp slt i32 %lhs, %rhs
  %res = select i1 %cond, i32 100, i32 200
  store i32 %res, i32 *%dst
  ret void
}

; Check the high end of the aligned CHY range.
define void @f4(i32 %lhs, i16 *%src, i32 *%dst) {
; CHECK-LABEL: f4:
; CHECK: chy %r2, 524286(%r3)
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%src, i64 262143
  %half = load i16, i16 *%ptr
  %rhs = sext i16 %half to i32
  %cond = icmp slt i32 %lhs, %rhs
  %res = select i1 %cond, i32 100, i32 200
  store i32 %res, i32 *%dst
  ret void
}

; Check the next halfword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f5(i32 %lhs, i16 *%src, i32 *%dst) {
; CHECK-LABEL: f5:
; CHECK: agfi %r3, 524288
; CHECK: ch %r2, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%src, i64 262144
  %half = load i16, i16 *%ptr
  %rhs = sext i16 %half to i32
  %cond = icmp slt i32 %lhs, %rhs
  %res = select i1 %cond, i32 100, i32 200
  store i32 %res, i32 *%dst
  ret void
}

; Check the high end of the negative aligned CHY range.
define void @f6(i32 %lhs, i16 *%src, i32 *%dst) {
; CHECK-LABEL: f6:
; CHECK: chy %r2, -2(%r3)
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%src, i64 -1
  %half = load i16, i16 *%ptr
  %rhs = sext i16 %half to i32
  %cond = icmp slt i32 %lhs, %rhs
  %res = select i1 %cond, i32 100, i32 200
  store i32 %res, i32 *%dst
  ret void
}

; Check the low end of the CHY range.
define void @f7(i32 %lhs, i16 *%src, i32 *%dst) {
; CHECK-LABEL: f7:
; CHECK: chy %r2, -524288(%r3)
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%src, i64 -262144
  %half = load i16, i16 *%ptr
  %rhs = sext i16 %half to i32
  %cond = icmp slt i32 %lhs, %rhs
  %res = select i1 %cond, i32 100, i32 200
  store i32 %res, i32 *%dst
  ret void
}

; Check the next halfword down, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f8(i32 %lhs, i16 *%src, i32 *%dst) {
; CHECK-LABEL: f8:
; CHECK: agfi %r3, -524290
; CHECK: ch %r2, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%src, i64 -262145
  %half = load i16, i16 *%ptr
  %rhs = sext i16 %half to i32
  %cond = icmp slt i32 %lhs, %rhs
  %res = select i1 %cond, i32 100, i32 200
  store i32 %res, i32 *%dst
  ret void
}

; Check that CH allows an index.
define void @f9(i32 %lhs, i64 %base, i64 %index, i32 *%dst) {
; CHECK-LABEL: f9:
; CHECK: ch %r2, 4094({{%r4,%r3|%r3,%r4}})
; CHECK: br %r14
  %add1 = add i64 %base, %index
  %add2 = add i64 %add1, 4094
  %ptr = inttoptr i64 %add2 to i16 *
  %half = load i16, i16 *%ptr
  %rhs = sext i16 %half to i32
  %cond = icmp slt i32 %lhs, %rhs
  %res = select i1 %cond, i32 100, i32 200
  store i32 %res, i32 *%dst
  ret void
}

; Check that CHY allows an index.
define void @f10(i32 %lhs, i64 %base, i64 %index, i32 *%dst) {
; CHECK-LABEL: f10:
; CHECK: chy %r2, 4096({{%r4,%r3|%r3,%r4}})
; CHECK: br %r14
  %add1 = add i64 %base, %index
  %add2 = add i64 %add1, 4096
  %ptr = inttoptr i64 %add2 to i16 *
  %half = load i16, i16 *%ptr
  %rhs = sext i16 %half to i32
  %cond = icmp slt i32 %lhs, %rhs
  %res = select i1 %cond, i32 100, i32 200
  store i32 %res, i32 *%dst
  ret void
}

; Check the comparison can be reversed if that allows CH to be used.
define double @f11(double %a, double %b, i32 %rhs, i16 *%src) {
; CHECK-LABEL: f11:
; CHECK: ch %r2, 0(%r3)
; CHECK-NEXT: bhr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %half = load i16, i16 *%src
  %lhs = sext i16 %half to i32
  %cond = icmp slt i32 %lhs, %rhs
  %res = select i1 %cond, double %a, double %b
  ret double %res
}
