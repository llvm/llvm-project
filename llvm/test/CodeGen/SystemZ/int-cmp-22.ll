; Test 16-bit signed ordered comparisons between memory and a constant.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check comparisons with 0.
define double @f1(double %a, double %b, i16 *%ptr) {
; CHECK-LABEL: f1:
; CHECK: chhsi 0(%r2), 0
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %val = load i16, i16 *%ptr
  %cond = icmp slt i16 %val, 0
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check comparisons with 1.
define double @f2(double %a, double %b, i16 *%ptr) {
; CHECK-LABEL: f2:
; CHECK: chhsi 0(%r2), 0
; CHECK-NEXT: bler %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %val = load i16, i16 *%ptr
  %cond = icmp slt i16 %val, 1
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check a value near the high end of the signed 16-bit range.
define double @f3(double %a, double %b, i16 *%ptr) {
; CHECK-LABEL: f3:
; CHECK: chhsi 0(%r2), 32766
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %val = load i16, i16 *%ptr
  %cond = icmp slt i16 %val, 32766
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check comparisons with -1.
define double @f4(double %a, double %b, i16 *%ptr) {
; CHECK-LABEL: f4:
; CHECK: chhsi 0(%r2), -1
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %val = load i16, i16 *%ptr
  %cond = icmp slt i16 %val, -1
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check a value near the low end of the 16-bit signed range.
define double @f5(double %a, double %b, i16 *%ptr) {
; CHECK-LABEL: f5:
; CHECK: chhsi 0(%r2), -32766
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %val = load i16, i16 *%ptr
  %cond = icmp slt i16 %val, -32766
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the high end of the CHHSI range.
define double @f6(double %a, double %b, i16 %i1, i16 *%base) {
; CHECK-LABEL: f6:
; CHECK: chhsi 4094(%r3), 0
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%base, i64 2047
  %val = load i16, i16 *%ptr
  %cond = icmp slt i16 %val, 0
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the next halfword up, which needs separate address logic,
define double @f7(double %a, double %b, i16 *%base) {
; CHECK-LABEL: f7:
; CHECK: aghi %r2, 4096
; CHECK: chhsi 0(%r2), 0
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%base, i64 2048
  %val = load i16, i16 *%ptr
  %cond = icmp slt i16 %val, 0
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check negative offsets, which also need separate address logic.
define double @f8(double %a, double %b, i16 *%base) {
; CHECK-LABEL: f8:
; CHECK: aghi %r2, -2
; CHECK: chhsi 0(%r2), 0
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%base, i64 -1
  %val = load i16, i16 *%ptr
  %cond = icmp slt i16 %val, 0
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check that CHHSI does not allow indices.
define double @f9(double %a, double %b, i64 %base, i64 %index) {
; CHECK-LABEL: f9:
; CHECK: agr {{%r2, %r3|%r3, %r2}}
; CHECK: chhsi 0({{%r[23]}}), 0
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %add = add i64 %base, %index
  %ptr = inttoptr i64 %add to i16 *
  %val = load i16, i16 *%ptr
  %cond = icmp slt i16 %val, 0
  %res = select i1 %cond, double %a, double %b
  ret double %res
}
