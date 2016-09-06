; Test 32-bit unsigned comparisons between memory and a constant.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check ordered comparisons with a constant near the low end of the unsigned
; 16-bit range.
define double @f1(double %a, double %b, i32 *%ptr) {
; CHECK-LABEL: f1:
; CHECK: clfhsi 0(%r2), 1
; CHECK-NEXT: bhr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %val = load i32 , i32 *%ptr
  %cond = icmp ugt i32 %val, 1
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check ordered comparisons with the high end of the unsigned 16-bit range.
define double @f2(double %a, double %b, i32 *%ptr) {
; CHECK-LABEL: f2:
; CHECK: clfhsi 0(%r2), 65535
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %val = load i32 , i32 *%ptr
  %cond = icmp ult i32 %val, 65535
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the next value up, which can't use CLFHSI.
define double @f3(double %a, double %b, i32 *%ptr) {
; CHECK-LABEL: f3:
; CHECK-NOT: clfhsi
; CHECK: br %r14
  %val = load i32 , i32 *%ptr
  %cond = icmp ult i32 %val, 65536
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check equality comparisons with 32768, the lowest value for which
; we prefer CLFHSI to CHSI.
define double @f4(double %a, double %b, i32 *%ptr) {
; CHECK-LABEL: f4:
; CHECK: clfhsi 0(%r2), 32768
; CHECK-NEXT: ber %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %val = load i32 , i32 *%ptr
  %cond = icmp eq i32 %val, 32768
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check equality comparisons with the high end of the unsigned 16-bit range.
define double @f5(double %a, double %b, i32 *%ptr) {
; CHECK-LABEL: f5:
; CHECK: clfhsi 0(%r2), 65535
; CHECK-NEXT: ber %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %val = load i32 , i32 *%ptr
  %cond = icmp eq i32 %val, 65535
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the next value up, which can't use CLFHSI.
define double @f6(double %a, double %b, i32 *%ptr) {
; CHECK-LABEL: f6:
; CHECK-NOT: clfhsi
; CHECK: br %r14
  %val = load i32 , i32 *%ptr
  %cond = icmp eq i32 %val, 65536
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the high end of the CLFHSI range.
define double @f7(double %a, double %b, i32 %i1, i32 *%base) {
; CHECK-LABEL: f7:
; CHECK: clfhsi 4092(%r3), 1
; CHECK-NEXT: bhr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%base, i64 1023
  %val = load i32 , i32 *%ptr
  %cond = icmp ugt i32 %val, 1
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the next word up, which needs separate address logic,
define double @f8(double %a, double %b, i32 *%base) {
; CHECK-LABEL: f8:
; CHECK: aghi %r2, 4096
; CHECK: clfhsi 0(%r2), 1
; CHECK-NEXT: bhr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%base, i64 1024
  %val = load i32 , i32 *%ptr
  %cond = icmp ugt i32 %val, 1
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check negative offsets, which also need separate address logic.
define double @f9(double %a, double %b, i32 *%base) {
; CHECK-LABEL: f9:
; CHECK: aghi %r2, -4
; CHECK: clfhsi 0(%r2), 1
; CHECK-NEXT: bhr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%base, i64 -1
  %val = load i32 , i32 *%ptr
  %cond = icmp ugt i32 %val, 1
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check that CLFHSI does not allow indices.
define double @f10(double %a, double %b, i64 %base, i64 %index) {
; CHECK-LABEL: f10:
; CHECK: agr {{%r2, %r3|%r3, %r2}}
; CHECK: clfhsi 0({{%r[23]}}), 1
; CHECK-NEXT: bhr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %add = add i64 %base, %index
  %ptr = inttoptr i64 %add to i32 *
  %val = load i32 , i32 *%ptr
  %cond = icmp ugt i32 %val, 1
  %res = select i1 %cond, double %a, double %b
  ret double %res
}
