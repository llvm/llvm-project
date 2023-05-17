; Test conversions of signed i64s to floating-point values.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Test i64->f32.
define float @f1(i64 %i) {
; CHECK-LABEL: f1:
; CHECK: cegbr %f0, %r2
; CHECK: br %r14
  %conv = sitofp i64 %i to float
  ret float %conv
}

; Test i64->f64.
define double @f2(i64 %i) {
; CHECK-LABEL: f2:
; CHECK: cdgbr %f0, %r2
; CHECK: br %r14
  %conv = sitofp i64 %i to double
  ret double %conv
}

; Test i64->f128.
define void @f3(i64 %i, ptr %dst) {
; CHECK-LABEL: f3:
; CHECK: cxgbr %f0, %r2
; CHECK: std %f0, 0(%r3)
; CHECK: std %f2, 8(%r3)
; CHECK: br %r14
  %conv = sitofp i64 %i to fp128
  store fp128 %conv, ptr %dst
  ret void
}
