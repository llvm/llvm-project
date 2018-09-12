; Test 32-bit ordered comparisons that are really between a memory byte
; and a constant.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check unsigned comparison near the low end of the CLI range, using zero
; extension.
define double @f1(double %a, double %b, i8 *%ptr) {
; CHECK-LABEL: f1:
; CHECK: cli 0(%r2), 1
; CHECK-NEXT: bhr %r14
; CHECK: br %r14
  %val = load i8, i8 *%ptr
  %ext = zext i8 %val to i32
  %cond = icmp ugt i32 %ext, 1
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check unsigned comparison near the low end of the CLI range, using sign
; extension.
define double @f2(double %a, double %b, i8 *%ptr) {
; CHECK-LABEL: f2:
; CHECK: cli 0(%r2), 1
; CHECK-NEXT: bhr %r14
; CHECK: br %r14
  %val = load i8, i8 *%ptr
  %ext = sext i8 %val to i32
  %cond = icmp ugt i32 %ext, 1
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check unsigned comparison near the high end of the CLI range, using zero
; extension.
define double @f3(double %a, double %b, i8 *%ptr) {
; CHECK-LABEL: f3:
; CHECK: cli 0(%r2), 254
; CHECK-NEXT: blr %r14
; CHECK: br %r14
  %val = load i8, i8 *%ptr
  %ext = zext i8 %val to i32
  %cond = icmp ult i32 %ext, 254
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check unsigned comparison near the high end of the CLI range, using sign
; extension.
define double @f4(double %a, double %b, i8 *%ptr) {
; CHECK-LABEL: f4:
; CHECK: cli 0(%r2), 254
; CHECK-NEXT: blr %r14
; CHECK: br %r14
  %val = load i8, i8 *%ptr
  %ext = sext i8 %val to i32
  %cond = icmp ult i32 %ext, -2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check unsigned comparison above the high end of the CLI range, using zero
; extension.  The condition is always true.
define double @f5(double %a, double %b, i8 *%ptr) {
; CHECK-LABEL: f5:
; CHECK-NOT: cli {{.*}}
; CHECK: br %r14
  %val = load i8, i8 *%ptr
  %ext = zext i8 %val to i32
  %cond = icmp ult i32 %ext, 256
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; When using unsigned comparison with sign extension, equality with values
; in the range [128, MAX-129] is impossible, and ordered comparisons with
; those values are effectively sign tests.  Since such comparisons are
; unlikely to occur in practice, we don't bother optimizing the second case,
; and simply ignore CLI for this range.  First check the low end of the range.
define double @f6(double %a, double %b, i8 *%ptr) {
; CHECK-LABEL: f6:
; CHECK-NOT: cli {{.*}}
; CHECK: br %r14
  %val = load i8, i8 *%ptr
  %ext = sext i8 %val to i32
  %cond = icmp ult i32 %ext, 128
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; ...and then the high end.
define double @f7(double %a, double %b, i8 *%ptr) {
; CHECK-LABEL: f7:
; CHECK-NOT: cli {{.*}}
; CHECK: br %r14
  %val = load i8, i8 *%ptr
  %ext = sext i8 %val to i32
  %cond = icmp ult i32 %ext, -129
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check signed comparison near the low end of the CLI range, using zero
; extension.  This is equivalent to unsigned comparison.
define double @f8(double %a, double %b, i8 *%ptr) {
; CHECK-LABEL: f8:
; CHECK: cli 0(%r2), 1
; CHECK-NEXT: bhr %r14
; CHECK: br %r14
  %val = load i8, i8 *%ptr
  %ext = zext i8 %val to i32
  %cond = icmp sgt i32 %ext, 1
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check signed comparison near the low end of the CLI range, using sign
; extension.  This cannot use CLI.
define double @f9(double %a, double %b, i8 *%ptr) {
; CHECK-LABEL: f9:
; CHECK-NOT: cli {{.*}}
; CHECK: br %r14
  %val = load i8, i8 *%ptr
  %ext = sext i8 %val to i32
  %cond = icmp sgt i32 %ext, 1
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check signed comparison near the high end of the CLI range, using zero
; extension.  This is equivalent to unsigned comparison.
define double @f10(double %a, double %b, i8 *%ptr) {
; CHECK-LABEL: f10:
; CHECK: cli 0(%r2), 254
; CHECK-NEXT: blr %r14
; CHECK: br %r14
  %val = load i8, i8 *%ptr
  %ext = zext i8 %val to i32
  %cond = icmp slt i32 %ext, 254
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check signed comparison near the high end of the CLI range, using sign
; extension.  This cannot use CLI.
define double @f11(double %a, double %b, i8 *%ptr) {
; CHECK-LABEL: f11:
; CHECK-NOT: cli {{.*}}
; CHECK: br %r14
  %val = load i8, i8 *%ptr
  %ext = sext i8 %val to i32
  %cond = icmp slt i32 %ext, -2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check signed comparison above the high end of the CLI range, using zero
; extension.  The condition is always true.
define double @f12(double %a, double %b, i8 *%ptr) {
; CHECK-LABEL: f12:
; CHECK-NOT: cli {{.*}}
; CHECK: br %r14
  %val = load i8, i8 *%ptr
  %ext = zext i8 %val to i32
  %cond = icmp slt i32 %ext, 256
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check tests for nonnegative values.
define double @f13(double %a, double %b, i8 *%ptr) {
; CHECK-LABEL: f13:
; CHECK: cli 0(%r2), 128
; CHECK-NEXT: blr %r14
; CHECK: br %r14
  %val = load i8, i8 *%ptr
  %ext = sext i8 %val to i32
  %cond = icmp sge i32 %ext, 0
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; ...and another form
define double @f14(double %a, double %b, i8 *%ptr) {
; CHECK-LABEL: f14:
; CHECK: cli 0(%r2), 128
; CHECK-NEXT: blr %r14
; CHECK: br %r14
  %val = load i8, i8 *%ptr
  %ext = sext i8 %val to i32
  %cond = icmp sgt i32 %ext, -1
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check tests for negative values.
define double @f15(double %a, double %b, i8 *%ptr) {
; CHECK-LABEL: f15:
; CHECK: cli 0(%r2), 127
; CHECK-NEXT: bhr %r14
; CHECK: br %r14
  %val = load i8, i8 *%ptr
  %ext = sext i8 %val to i32
  %cond = icmp slt i32 %ext, 0
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; ...and another form
define double @f16(double %a, double %b, i8 *%ptr) {
; CHECK-LABEL: f16:
; CHECK: cli 0(%r2), 127
; CHECK-NEXT: bhr %r14
; CHECK: br %r14
  %val = load i8, i8 *%ptr
  %ext = sext i8 %val to i32
  %cond = icmp sle i32 %ext, -1
  %res = select i1 %cond, double %a, double %b
  ret double %res
}
