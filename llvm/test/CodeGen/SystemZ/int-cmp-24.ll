; Test 16-bit equality comparisons between memory and a constant.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check the low end of the unsigned 16-bit range.
define double @f1(double %a, double %b, i16 *%ptr) {
; CHECK-LABEL: f1:
; CHECK: clhhsi 0(%r2), 0
; CHECK-NEXT: ber %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %val = load i16, i16 *%ptr
  %cond = icmp eq i16 %val, 0
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the high end of the unsigned 16-bit range.
define double @f2(double %a, double %b, i16 *%ptr) {
; CHECK-LABEL: f2:
; CHECK: clhhsi 0(%r2), 65535
; CHECK-NEXT: ber %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %val = load i16, i16 *%ptr
  %cond = icmp eq i16 %val, 65535
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the low end of the signed 16-bit range.
define double @f3(double %a, double %b, i16 *%ptr) {
; CHECK-LABEL: f3:
; CHECK: clhhsi 0(%r2), 32768
; CHECK-NEXT: ber %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %val = load i16, i16 *%ptr
  %cond = icmp eq i16 %val, -32768
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the high end of the signed 16-bit range.
define double @f4(double %a, double %b, i16 *%ptr) {
; CHECK-LABEL: f4:
; CHECK: clhhsi 0(%r2), 32767
; CHECK-NEXT: ber %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %val = load i16, i16 *%ptr
  %cond = icmp eq i16 %val, 32767
  %res = select i1 %cond, double %a, double %b
  ret double %res
}
