; Test 8-bit unsigned comparisons between memory and constants.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check ordered comparisons near the low end of the unsigned 8-bit range.
define double @f1(double %a, double %b, i8 *%ptr) {
; CHECK-LABEL: f1:
; CHECK: cli 0(%r2), 1
; CHECK-NEXT: bhr %r14
; CHECK: br %r14
  %val = load i8 , i8 *%ptr
  %cond = icmp ugt i8 %val, 1
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check ordered comparisons near the high end of the unsigned 8-bit range.
define double @f2(double %a, double %b, i8 *%ptr) {
; CHECK-LABEL: f2:
; CHECK: cli 0(%r2), 254
; CHECK-NEXT: blr %r14
; CHECK: br %r14
  %val = load i8 , i8 *%ptr
  %cond = icmp ult i8 %val, 254
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check tests for negative bytes.
define double @f3(double %a, double %b, i8 *%ptr) {
; CHECK-LABEL: f3:
; CHECK: cli 0(%r2), 127
; CHECK-NEXT: bhr %r14
; CHECK: br %r14
  %val = load i8 , i8 *%ptr
  %cond = icmp slt i8 %val, 0
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; ...and an alternative form.
define double @f4(double %a, double %b, i8 *%ptr) {
; CHECK-LABEL: f4:
; CHECK: cli 0(%r2), 127
; CHECK-NEXT: bhr %r14
; CHECK: br %r14
  %val = load i8 , i8 *%ptr
  %cond = icmp sle i8 %val, -1
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check tests for non-negative bytes.
define double @f5(double %a, double %b, i8 *%ptr) {
; CHECK-LABEL: f5:
; CHECK: cli 0(%r2), 128
; CHECK-NEXT: blr %r14
; CHECK: br %r14
  %val = load i8 , i8 *%ptr
  %cond = icmp sge i8 %val, 0
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; ...and an alternative form.
define double @f6(double %a, double %b, i8 *%ptr) {
; CHECK-LABEL: f6:
; CHECK: cli 0(%r2), 128
; CHECK-NEXT: blr %r14
; CHECK: br %r14
  %val = load i8 , i8 *%ptr
  %cond = icmp sgt i8 %val, -1
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check equality comparisons at the low end of the signed 8-bit range.
define double @f7(double %a, double %b, i8 *%ptr) {
; CHECK-LABEL: f7:
; CHECK: cli 0(%r2), 128
; CHECK-NEXT: ber %r14
; CHECK: br %r14
  %val = load i8 , i8 *%ptr
  %cond = icmp eq i8 %val, -128
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check equality comparisons at the low end of the unsigned 8-bit range.
define double @f8(double %a, double %b, i8 *%ptr) {
; CHECK-LABEL: f8:
; CHECK: cli 0(%r2), 0
; CHECK-NEXT: ber %r14
; CHECK: br %r14
  %val = load i8 , i8 *%ptr
  %cond = icmp eq i8 %val, 0
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check equality comparisons at the high end of the signed 8-bit range.
define double @f9(double %a, double %b, i8 *%ptr) {
; CHECK-LABEL: f9:
; CHECK: cli 0(%r2), 127
; CHECK-NEXT: ber %r14
; CHECK: br %r14
  %val = load i8 , i8 *%ptr
  %cond = icmp eq i8 %val, 127
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check equality comparisons at the high end of the unsigned 8-bit range.
define double @f10(double %a, double %b, i8 *%ptr) {
; CHECK-LABEL: f10:
; CHECK: cli 0(%r2), 255
; CHECK-NEXT: ber %r14
; CHECK: br %r14
  %val = load i8 , i8 *%ptr
  %cond = icmp eq i8 %val, 255
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the high end of the CLI range.
define double @f11(double %a, double %b, i8 *%src) {
; CHECK-LABEL: f11:
; CHECK: cli 4095(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%src, i64 4095
  %val = load i8 , i8 *%ptr
  %cond = icmp ult i8 %val, 127
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the next byte up, which should use CLIY instead of CLI.
define double @f12(double %a, double %b, i8 *%src) {
; CHECK-LABEL: f12:
; CHECK: cliy 4096(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%src, i64 4096
  %val = load i8 , i8 *%ptr
  %cond = icmp ult i8 %val, 127
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the high end of the CLIY range.
define double @f13(double %a, double %b, i8 *%src) {
; CHECK-LABEL: f13:
; CHECK: cliy 524287(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%src, i64 524287
  %val = load i8 , i8 *%ptr
  %cond = icmp ult i8 %val, 127
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the next byte up, which needs separate address logic.
; Other sequences besides this one would be OK.
define double @f14(double %a, double %b, i8 *%src) {
; CHECK-LABEL: f14:
; CHECK: agfi %r2, 524288
; CHECK: cli 0(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%src, i64 524288
  %val = load i8 , i8 *%ptr
  %cond = icmp ult i8 %val, 127
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the high end of the negative CLIY range.
define double @f15(double %a, double %b, i8 *%src) {
; CHECK-LABEL: f15:
; CHECK: cliy -1(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%src, i64 -1
  %val = load i8 , i8 *%ptr
  %cond = icmp ult i8 %val, 127
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the low end of the CLIY range.
define double @f16(double %a, double %b, i8 *%src) {
; CHECK-LABEL: f16:
; CHECK: cliy -524288(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%src, i64 -524288
  %val = load i8 , i8 *%ptr
  %cond = icmp ult i8 %val, 127
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the next byte down, which needs separate address logic.
; Other sequences besides this one would be OK.
define double @f17(double %a, double %b, i8 *%src) {
; CHECK-LABEL: f17:
; CHECK: agfi %r2, -524289
; CHECK: cli 0(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%src, i64 -524289
  %val = load i8 , i8 *%ptr
  %cond = icmp ult i8 %val, 127
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check that CLI does not allow an index
define double @f18(double %a, double %b, i64 %base, i64 %index) {
; CHECK-LABEL: f18:
; CHECK: agr %r2, %r3
; CHECK: cli 4095(%r2), 127
; CHECK: br %r14
  %add1 = add i64 %base, %index
  %add2 = add i64 %add1, 4095
  %ptr = inttoptr i64 %add2 to i8 *
  %val = load i8 , i8 *%ptr
  %cond = icmp ult i8 %val, 127
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check that CLIY does not allow an index
define double @f19(double %a, double %b, i64 %base, i64 %index) {
; CHECK-LABEL: f19:
; CHECK: agr %r2, %r3
; CHECK: cliy 4096(%r2), 127
; CHECK: br %r14
  %add1 = add i64 %base, %index
  %add2 = add i64 %add1, 4096
  %ptr = inttoptr i64 %add2 to i8 *
  %val = load i8 , i8 *%ptr
  %cond = icmp ult i8 %val, 127
  %res = select i1 %cond, double %a, double %b
  ret double %res
}
