; Test 16-bit unsigned comparisons between memory and a constant.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check a value near the low end of the unsigned 16-bit range.
define double @f1(double %a, double %b, i16 *%ptr) {
; CHECK-LABEL: f1:
; CHECK: clhhsi 0(%r2), 1
; CHECK-NEXT: bhr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %val = load i16 , i16 *%ptr
  %cond = icmp ugt i16 %val, 1
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check a value near the high end of the unsigned 16-bit range.
define double @f2(double %a, double %b, i16 *%ptr) {
; CHECK-LABEL: f2:
; CHECK: clhhsi 0(%r2), 65534
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %val = load i16 , i16 *%ptr
  %cond = icmp ult i16 %val, 65534
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the high end of the CLHHSI range.
define double @f3(double %a, double %b, i16 %i1, i16 *%base) {
; CHECK-LABEL: f3:
; CHECK: clhhsi 4094(%r3), 1
; CHECK-NEXT: bhr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%base, i64 2047
  %val = load i16 , i16 *%ptr
  %cond = icmp ugt i16 %val, 1
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the next halfword up, which needs separate address logic,
define double @f4(double %a, double %b, i16 *%base) {
; CHECK-LABEL: f4:
; CHECK: aghi %r2, 4096
; CHECK: clhhsi 0(%r2), 1
; CHECK-NEXT: bhr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%base, i64 2048
  %val = load i16 , i16 *%ptr
  %cond = icmp ugt i16 %val, 1
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check negative offsets, which also need separate address logic.
define double @f5(double %a, double %b, i16 *%base) {
; CHECK-LABEL: f5:
; CHECK: aghi %r2, -2
; CHECK: clhhsi 0(%r2), 1
; CHECK-NEXT: bhr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%base, i64 -1
  %val = load i16 , i16 *%ptr
  %cond = icmp ugt i16 %val, 1
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check that CLHHSI does not allow indices.
define double @f6(double %a, double %b, i64 %base, i64 %index) {
; CHECK-LABEL: f6:
; CHECK: agr {{%r2, %r3|%r3, %r2}}
; CHECK: clhhsi 0({{%r[23]}}), 1
; CHECK-NEXT: bhr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %add = add i64 %base, %index
  %ptr = inttoptr i64 %add to i16 *
  %val = load i16 , i16 *%ptr
  %cond = icmp ugt i16 %val, 1
  %res = select i1 %cond, double %a, double %b
  ret double %res
}
