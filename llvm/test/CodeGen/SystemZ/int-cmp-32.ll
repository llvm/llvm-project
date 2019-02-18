; Test 32-bit signed comparisons between memory and a constant.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check ordered comparisons with 0.
define double @f1(double %a, double %b, i32 *%ptr) {
; CHECK-LABEL: f1:
; CHECK: chsi 0(%r2), 0
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %val = load i32, i32 *%ptr
  %cond = icmp slt i32 %val, 0
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check ordered comparisons with 1.
define double @f2(double %a, double %b, i32 *%ptr) {
; CHECK-LABEL: f2:
; CHECK: chsi 0(%r2), 0
; CHECK-NEXT: bler %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %val = load i32, i32 *%ptr
  %cond = icmp slt i32 %val, 1
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check ordered comparisons with the high end of the signed 16-bit range.
define double @f3(double %a, double %b, i32 *%ptr) {
; CHECK-LABEL: f3:
; CHECK: chsi 0(%r2), 32767
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %val = load i32, i32 *%ptr
  %cond = icmp slt i32 %val, 32767
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the next value up, which can't use CHSI.
define double @f4(double %a, double %b, i32 *%ptr) {
; CHECK-LABEL: f4:
; CHECK-NOT: chsi
; CHECK: br %r14
  %val = load i32, i32 *%ptr
  %cond = icmp slt i32 %val, 32768
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check ordered comparisons with -1.
define double @f5(double %a, double %b, i32 *%ptr) {
; CHECK-LABEL: f5:
; CHECK: chsi 0(%r2), -1
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %val = load i32, i32 *%ptr
  %cond = icmp slt i32 %val, -1
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check ordered comparisons with the low end of the 16-bit signed range.
define double @f6(double %a, double %b, i32 *%ptr) {
; CHECK-LABEL: f6:
; CHECK: chsi 0(%r2), -32768
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %val = load i32, i32 *%ptr
  %cond = icmp slt i32 %val, -32768
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the next value down, which can't use CHSI.
define double @f7(double %a, double %b, i32 *%ptr) {
; CHECK-LABEL: f7:
; CHECK-NOT: chsi
; CHECK: br %r14
  %val = load i32, i32 *%ptr
  %cond = icmp slt i32 %val, -32769
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check equality comparisons with 0.
define double @f8(double %a, double %b, i32 *%ptr) {
; CHECK-LABEL: f8:
; CHECK: chsi 0(%r2), 0
; CHECK-NEXT: ber %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %val = load i32, i32 *%ptr
  %cond = icmp eq i32 %val, 0
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check equality comparisons with 1.
define double @f9(double %a, double %b, i32 *%ptr) {
; CHECK-LABEL: f9:
; CHECK: chsi 0(%r2), 1
; CHECK-NEXT: ber %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %val = load i32, i32 *%ptr
  %cond = icmp eq i32 %val, 1
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check equality comparisons with the high end of the signed 16-bit range.
define double @f10(double %a, double %b, i32 *%ptr) {
; CHECK-LABEL: f10:
; CHECK: chsi 0(%r2), 32767
; CHECK-NEXT: ber %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %val = load i32, i32 *%ptr
  %cond = icmp eq i32 %val, 32767
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the next value up, which can't use CHSI.
define double @f11(double %a, double %b, i32 *%ptr) {
; CHECK-LABEL: f11:
; CHECK-NOT: chsi
; CHECK: br %r14
  %val = load i32, i32 *%ptr
  %cond = icmp eq i32 %val, 32768
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check equality comparisons with -1.
define double @f12(double %a, double %b, i32 *%ptr) {
; CHECK-LABEL: f12:
; CHECK: chsi 0(%r2), -1
; CHECK-NEXT: ber %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %val = load i32, i32 *%ptr
  %cond = icmp eq i32 %val, -1
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check equality comparisons with the low end of the 16-bit signed range.
define double @f13(double %a, double %b, i32 *%ptr) {
; CHECK-LABEL: f13:
; CHECK: chsi 0(%r2), -32768
; CHECK-NEXT: ber %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %val = load i32, i32 *%ptr
  %cond = icmp eq i32 %val, -32768
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the next value down, which should be treated as a positive value.
define double @f14(double %a, double %b, i32 *%ptr) {
; CHECK-LABEL: f14:
; CHECK-NOT: chsi
; CHECK: br %r14
  %val = load i32, i32 *%ptr
  %cond = icmp eq i32 %val, -32769
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the high end of the CHSI range.
define double @f15(double %a, double %b, i32 %i1, i32 *%base) {
; CHECK-LABEL: f15:
; CHECK: chsi 4092(%r3), 0
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%base, i64 1023
  %val = load i32, i32 *%ptr
  %cond = icmp slt i32 %val, 0
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the next word up, which needs separate address logic,
define double @f16(double %a, double %b, i32 *%base) {
; CHECK-LABEL: f16:
; CHECK: aghi %r2, 4096
; CHECK: chsi 0(%r2), 0
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%base, i64 1024
  %val = load i32, i32 *%ptr
  %cond = icmp slt i32 %val, 0
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check negative offsets, which also need separate address logic.
define double @f17(double %a, double %b, i32 *%base) {
; CHECK-LABEL: f17:
; CHECK: aghi %r2, -4
; CHECK: chsi 0(%r2), 0
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%base, i64 -1
  %val = load i32, i32 *%ptr
  %cond = icmp slt i32 %val, 0
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check that CHSI does not allow indices.
define double @f18(double %a, double %b, i64 %base, i64 %index) {
; CHECK-LABEL: f18:
; CHECK: agr {{%r2, %r3|%r3, %r2}}
; CHECK: chsi 0({{%r[23]}}), 0
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %add = add i64 %base, %index
  %ptr = inttoptr i64 %add to i32 *
  %val = load i32, i32 *%ptr
  %cond = icmp slt i32 %val, 0
  %res = select i1 %cond, double %a, double %b
  ret double %res
}
