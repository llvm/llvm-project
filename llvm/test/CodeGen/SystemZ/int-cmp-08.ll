; Test 64-bit unsigned comparison in which the second operand is a variable.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check CLGR.
define double @f1(double %a, double %b, i64 %i1, i64 %i2) {
; CHECK-LABEL: f1:
; CHECK: clgrbl %r2, %r3, 0(%r14)
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp ult i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check CLG with no displacement.
define double @f2(double %a, double %b, i64 %i1, i64 *%ptr) {
; CHECK-LABEL: f2:
; CHECK: clg %r2, 0(%r3)
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %i2 = load i64, i64 *%ptr
  %cond = icmp ult i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the high end of the aligned CLG range.
define double @f3(double %a, double %b, i64 %i1, i64 *%base) {
; CHECK-LABEL: f3:
; CHECK: clg %r2, 524280(%r3)
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%base, i64 65535
  %i2 = load i64, i64 *%ptr
  %cond = icmp ult i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the next doubleword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define double @f4(double %a, double %b, i64 %i1, i64 *%base) {
; CHECK-LABEL: f4:
; CHECK: agfi %r3, 524288
; CHECK: clg %r2, 0(%r3)
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%base, i64 65536
  %i2 = load i64, i64 *%ptr
  %cond = icmp ult i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the high end of the negative aligned CLG range.
define double @f5(double %a, double %b, i64 %i1, i64 *%base) {
; CHECK-LABEL: f5:
; CHECK: clg %r2, -8(%r3)
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%base, i64 -1
  %i2 = load i64, i64 *%ptr
  %cond = icmp ult i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the low end of the CLG range.
define double @f6(double %a, double %b, i64 %i1, i64 *%base) {
; CHECK-LABEL: f6:
; CHECK: clg %r2, -524288(%r3)
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%base, i64 -65536
  %i2 = load i64, i64 *%ptr
  %cond = icmp ult i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the next doubleword down, which needs separate address logic.
; Other sequences besides this one would be OK.
define double @f7(double %a, double %b, i64 %i1, i64 *%base) {
; CHECK-LABEL: f7:
; CHECK: agfi %r3, -524296
; CHECK: clg %r2, 0(%r3)
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%base, i64 -65537
  %i2 = load i64, i64 *%ptr
  %cond = icmp ult i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check that CLG allows an index.
define double @f8(double %a, double %b, i64 %i1, i64 %base, i64 %index) {
; CHECK-LABEL: f8:
; CHECK: clg %r2, 524280({{%r4,%r3|%r3,%r4}})
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %add1 = add i64 %base, %index
  %add2 = add i64 %add1, 524280
  %ptr = inttoptr i64 %add2 to i64 *
  %i2 = load i64, i64 *%ptr
  %cond = icmp ult i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the comparison can be reversed if that allows CLG to be used.
define double @f9(double %a, double %b, i64 %i2, i64 *%ptr) {
; CHECK-LABEL: f9:
; CHECK: clg %r2, 0(%r3)
; CHECK-NEXT: bhr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %i1 = load i64, i64 *%ptr
  %cond = icmp ult i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}
