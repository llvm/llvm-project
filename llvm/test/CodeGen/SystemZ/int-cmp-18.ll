; Test 64-bit equality comparisons that are really between a memory byte
; and a constant.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check the low end of the 8-bit unsigned range, with zero extension.
define double @f1(double %a, double %b, ptr %ptr) {
; CHECK-LABEL: f1:
; CHECK: cli 0(%r2), 0
; CHECK-NEXT: ber %r14
; CHECK: br %r14
  %val = load i8, ptr %ptr
  %ext = zext i8 %val to i64
  %cond = icmp eq i64 %ext, 0
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the high end of the 8-bit unsigned range, with zero extension.
define double @f2(double %a, double %b, ptr %ptr) {
; CHECK-LABEL: f2:
; CHECK: cli 0(%r2), 255
; CHECK-NEXT: ber %r14
; CHECK: br %r14
  %val = load i8, ptr %ptr
  %ext = zext i8 %val to i64
  %cond = icmp eq i64 %ext, 255
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the next value up, with zero extension.  The condition is always false.
define double @f3(double %a, double %b, ptr %ptr) {
; CHECK-LABEL: f3:
; CHECK-NOT: cli
; CHECK: br %r14
  %val = load i8, ptr %ptr
  %ext = zext i8 %val to i64
  %cond = icmp eq i64 %ext, 256
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check comparisons with -1, with zero extension.
; This condition is also always false.
define double @f4(double %a, double %b, ptr %ptr) {
; CHECK-LABEL: f4:
; CHECK-NOT: cli
; CHECK: br %r14
  %val = load i8, ptr %ptr
  %ext = zext i8 %val to i64
  %cond = icmp eq i64 %ext, -1
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check comparisons with 0, using sign extension.
define double @f5(double %a, double %b, ptr %ptr) {
; CHECK-LABEL: f5:
; CHECK: cli 0(%r2), 0
; CHECK-NEXT: ber %r14
; CHECK: br %r14
  %val = load i8, ptr %ptr
  %ext = sext i8 %val to i64
  %cond = icmp eq i64 %ext, 0
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the high end of the signed 8-bit range, using sign extension.
define double @f6(double %a, double %b, ptr %ptr) {
; CHECK-LABEL: f6:
; CHECK: cli 0(%r2), 127
; CHECK-NEXT: ber %r14
; CHECK: br %r14
  %val = load i8, ptr %ptr
  %ext = sext i8 %val to i64
  %cond = icmp eq i64 %ext, 127
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the next value up, using sign extension.
; The condition is always false.
define double @f7(double %a, double %b, ptr %ptr) {
; CHECK-LABEL: f7:
; CHECK-NOT: cli
; CHECK: br %r14
  %val = load i8, ptr %ptr
  %ext = sext i8 %val to i64
  %cond = icmp eq i64 %ext, 128
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check comparisons with -1, using sign extension.
define double @f8(double %a, double %b, ptr %ptr) {
; CHECK-LABEL: f8:
; CHECK: cli 0(%r2), 255
; CHECK-NEXT: ber %r14
; CHECK: br %r14
  %val = load i8, ptr %ptr
  %ext = sext i8 %val to i64
  %cond = icmp eq i64 %ext, -1
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the low end of the signed 8-bit range, using sign extension.
define double @f9(double %a, double %b, ptr %ptr) {
; CHECK-LABEL: f9:
; CHECK: cli 0(%r2), 128
; CHECK-NEXT: ber %r14
; CHECK: br %r14
  %val = load i8, ptr %ptr
  %ext = sext i8 %val to i64
  %cond = icmp eq i64 %ext, -128
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the next value down, using sign extension.
; The condition is always false.
define double @f10(double %a, double %b, ptr %ptr) {
; CHECK-LABEL: f10:
; CHECK-NOT: cli
; CHECK: br %r14
  %val = load i8, ptr %ptr
  %ext = sext i8 %val to i64
  %cond = icmp eq i64 %ext, -129
  %res = select i1 %cond, double %a, double %b
  ret double %res
}
