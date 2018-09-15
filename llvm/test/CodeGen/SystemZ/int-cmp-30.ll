; Test 32-bit ordered comparisons that are really between a memory halfword
; and a constant.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check unsigned comparison near the low end of the CLHHSI range, using zero
; extension.
define double @f1(double %a, double %b, i16 *%ptr) {
; CHECK-LABEL: f1:
; CHECK: clhhsi 0(%r2), 1
; CHECK-NEXT: bhr %r14
; CHECK: br %r14
  %val = load i16, i16 *%ptr
  %ext = zext i16 %val to i32
  %cond = icmp ugt i32 %ext, 1
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check unsigned comparison near the low end of the CLHHSI range, using sign
; extension.
define double @f2(double %a, double %b, i16 *%ptr) {
; CHECK-LABEL: f2:
; CHECK: clhhsi 0(%r2), 1
; CHECK-NEXT: bhr %r14
; CHECK: br %r14
  %val = load i16, i16 *%ptr
  %ext = sext i16 %val to i32
  %cond = icmp ugt i32 %ext, 1
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check unsigned comparison near the high end of the CLHHSI range, using zero
; extension.
define double @f3(double %a, double %b, i16 *%ptr) {
; CHECK-LABEL: f3:
; CHECK: clhhsi 0(%r2), 65534
; CHECK-NEXT: blr %r14
; CHECK: br %r14
  %val = load i16, i16 *%ptr
  %ext = zext i16 %val to i32
  %cond = icmp ult i32 %ext, 65534
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check unsigned comparison near the high end of the CLHHSI range, using sign
; extension.
define double @f4(double %a, double %b, i16 *%ptr) {
; CHECK-LABEL: f4:
; CHECK: clhhsi 0(%r2), 65534
; CHECK-NEXT: blr %r14
; CHECK: br %r14
  %val = load i16, i16 *%ptr
  %ext = sext i16 %val to i32
  %cond = icmp ult i32 %ext, -2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check unsigned comparison above the high end of the CLHHSI range, using zero
; extension.  The condition is always true.
define double @f5(double %a, double %b, i16 *%ptr) {
; CHECK-LABEL: f5:
; CHECK-NOT: clhhsi
; CHECK: br %r14
  %val = load i16, i16 *%ptr
  %ext = zext i16 %val to i32
  %cond = icmp ult i32 %ext, 65536
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; When using unsigned comparison with sign extension, equality with values
; in the range [32768, MAX-32769] is impossible, and ordered comparisons with
; those values are effectively sign tests.  Since such comparisons are
; unlikely to occur in practice, we don't bother optimizing the second case,
; and simply ignore CLHHSI for this range.  First check the low end of the
; range.
define double @f6(double %a, double %b, i16 *%ptr) {
; CHECK-LABEL: f6:
; CHECK-NOT: clhhsi
; CHECK: br %r14
  %val = load i16, i16 *%ptr
  %ext = sext i16 %val to i32
  %cond = icmp ult i32 %ext, 32768
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; ...and then the high end.
define double @f7(double %a, double %b, i16 *%ptr) {
; CHECK-LABEL: f7:
; CHECK-NOT: clhhsi
; CHECK: br %r14
  %val = load i16, i16 *%ptr
  %ext = sext i16 %val to i32
  %cond = icmp ult i32 %ext, -32769
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check signed comparison near the low end of the CLHHSI range, using zero
; extension.  This is equivalent to unsigned comparison.
define double @f8(double %a, double %b, i16 *%ptr) {
; CHECK-LABEL: f8:
; CHECK: clhhsi 0(%r2), 1
; CHECK-NEXT: bhr %r14
; CHECK: br %r14
  %val = load i16, i16 *%ptr
  %ext = zext i16 %val to i32
  %cond = icmp sgt i32 %ext, 1
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check signed comparison near the low end of the CLHHSI range, using sign
; extension.  This should use CHHSI instead.
define double @f9(double %a, double %b, i16 *%ptr) {
; CHECK-LABEL: f9:
; CHECK: chhsi 0(%r2), 1
; CHECK-NEXT: bhr %r14
; CHECK: br %r14
  %val = load i16, i16 *%ptr
  %ext = sext i16 %val to i32
  %cond = icmp sgt i32 %ext, 1
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check signed comparison near the high end of the CLHHSI range, using zero
; extension.  This is equivalent to unsigned comparison.
define double @f10(double %a, double %b, i16 *%ptr) {
; CHECK-LABEL: f10:
; CHECK: clhhsi 0(%r2), 65534
; CHECK-NEXT: blr %r14
; CHECK: br %r14
  %val = load i16, i16 *%ptr
  %ext = zext i16 %val to i32
  %cond = icmp slt i32 %ext, 65534
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check signed comparison near the high end of the CLHHSI range, using sign
; extension.  This should use CHHSI instead.
define double @f11(double %a, double %b, i16 *%ptr) {
; CHECK-LABEL: f11:
; CHECK: chhsi 0(%r2), -2
; CHECK-NEXT: blr %r14
; CHECK: br %r14
  %val = load i16, i16 *%ptr
  %ext = sext i16 %val to i32
  %cond = icmp slt i32 %ext, -2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check signed comparison above the high end of the CLHHSI range, using zero
; extension.  The condition is always true.
define double @f12(double %a, double %b, i16 *%ptr) {
; CHECK-LABEL: f12:
; CHECK-NOT: cli
; CHECK: br %r14
  %val = load i16, i16 *%ptr
  %ext = zext i16 %val to i32
  %cond = icmp slt i32 %ext, 65536
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check signed comparison near the high end of the CHHSI range, using sign
; extension.
define double @f13(double %a, double %b, i16 *%ptr) {
; CHECK-LABEL: f13:
; CHECK: chhsi 0(%r2), 32766
; CHECK-NEXT: blr %r14
; CHECK: br %r14
  %val = load i16, i16 *%ptr
  %ext = sext i16 %val to i32
  %cond = icmp slt i32 %ext, 32766
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check signed comparison above the high end of the CHHSI range, using sign
; extension.  This condition is always true.
define double @f14(double %a, double %b, i16 *%ptr) {
; CHECK-LABEL: f14:
; CHECK-NOT: chhsi
; CHECK: br %r14
  %val = load i16, i16 *%ptr
  %ext = sext i16 %val to i32
  %cond = icmp slt i32 %ext, 32768
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check signed comparison near the low end of the CHHSI range, using sign
; extension.
define double @f15(double %a, double %b, i16 *%ptr) {
; CHECK-LABEL: f15:
; CHECK: chhsi 0(%r2), -32767
; CHECK-NEXT: bhr %r14
; CHECK: br %r14
  %val = load i16, i16 *%ptr
  %ext = sext i16 %val to i32
  %cond = icmp sgt i32 %ext, -32767
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check signed comparison below the low end of the CHHSI range, using sign
; extension.  This condition is always true.
define double @f16(double %a, double %b, i16 *%ptr) {
; CHECK-LABEL: f16:
; CHECK-NOT: chhsi
; CHECK: br %r14
  %val = load i16, i16 *%ptr
  %ext = sext i16 %val to i32
  %cond = icmp sgt i32 %ext, -32769
  %res = select i1 %cond, double %a, double %b
  ret double %res
}
