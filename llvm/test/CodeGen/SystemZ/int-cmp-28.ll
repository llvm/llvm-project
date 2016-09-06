; Test 64-bit equality comparisons that are really between a memory halfword
; and a constant.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check the low end of the 16-bit unsigned range, with zero extension.
define double @f1(double %a, double %b, i16 *%ptr) {
; CHECK-LABEL: f1:
; CHECK: clhhsi 0(%r2), 0
; CHECK-NEXT: ber %r14
; CHECK: br %r14
  %val = load i16 , i16 *%ptr
  %ext = zext i16 %val to i64
  %cond = icmp eq i64 %ext, 0
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the high end of the 16-bit unsigned range, with zero extension.
define double @f2(double %a, double %b, i16 *%ptr) {
; CHECK-LABEL: f2:
; CHECK: clhhsi 0(%r2), 65535
; CHECK-NEXT: ber %r14
; CHECK: br %r14
  %val = load i16 , i16 *%ptr
  %ext = zext i16 %val to i64
  %cond = icmp eq i64 %ext, 65535
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the next value up, with zero extension.  The condition is always false.
define double @f3(double %a, double %b, i16 *%ptr) {
; CHECK-LABEL: f3:
; CHECK-NOT: clhhsi
; CHECK: br %r14
  %val = load i16 , i16 *%ptr
  %ext = zext i16 %val to i64
  %cond = icmp eq i64 %ext, 65536
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check comparisons with -1, with zero extension.
; This condition is also always false.
define double @f4(double %a, double %b, i16 *%ptr) {
; CHECK-LABEL: f4:
; CHECK-NOT: clhhsi
; CHECK: br %r14
  %val = load i16 , i16 *%ptr
  %ext = zext i16 %val to i64
  %cond = icmp eq i64 %ext, -1
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check comparisons with 0, using sign extension.
define double @f5(double %a, double %b, i16 *%ptr) {
; CHECK-LABEL: f5:
; CHECK: clhhsi 0(%r2), 0
; CHECK-NEXT: ber %r14
; CHECK: br %r14
  %val = load i16 , i16 *%ptr
  %ext = sext i16 %val to i64
  %cond = icmp eq i64 %ext, 0
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the high end of the signed 16-bit range, using sign extension.
define double @f6(double %a, double %b, i16 *%ptr) {
; CHECK-LABEL: f6:
; CHECK: clhhsi 0(%r2), 32767
; CHECK-NEXT: ber %r14
; CHECK: br %r14
  %val = load i16 , i16 *%ptr
  %ext = sext i16 %val to i64
  %cond = icmp eq i64 %ext, 32767
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the next value up, using sign extension.
; The condition is always false.
define double @f7(double %a, double %b, i16 *%ptr) {
; CHECK-LABEL: f7:
; CHECK-NOT: clhhsi
; CHECK: br %r14
  %val = load i16 , i16 *%ptr
  %ext = sext i16 %val to i64
  %cond = icmp eq i64 %ext, 32768
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check comparisons with -1, using sign extension.
define double @f8(double %a, double %b, i16 *%ptr) {
; CHECK-LABEL: f8:
; CHECK: clhhsi 0(%r2), 65535
; CHECK-NEXT: ber %r14
; CHECK: br %r14
  %val = load i16 , i16 *%ptr
  %ext = sext i16 %val to i64
  %cond = icmp eq i64 %ext, -1
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the low end of the signed 16-bit range, using sign extension.
define double @f9(double %a, double %b, i16 *%ptr) {
; CHECK-LABEL: f9:
; CHECK: clhhsi 0(%r2), 32768
; CHECK-NEXT: ber %r14
; CHECK: br %r14
  %val = load i16 , i16 *%ptr
  %ext = sext i16 %val to i64
  %cond = icmp eq i64 %ext, -32768
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the next value down, using sign extension.
; The condition is always false.
define double @f10(double %a, double %b, i16 *%ptr) {
; CHECK-LABEL: f10:
; CHECK-NOT: clhhsi
; CHECK: br %r14
  %val = load i16 , i16 *%ptr
  %ext = sext i16 %val to i64
  %cond = icmp eq i64 %ext, -32769
  %res = select i1 %cond, double %a, double %b
  ret double %res
}
