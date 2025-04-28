; Test conversions of unsigned i64s to floating-point values (z10 only).
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s

; Test i64->f32.  There's no native support for unsigned i64-to-fp conversions,
; but we should be able to implement them using signed i64-to-fp conversions.
define float @f1(i64 %i) {
; CHECK-LABEL: f1:
; CHECK: cegbr
; CHECK: aebr
; CHECK: br %r14
  %conv = uitofp i64 %i to float
  ret float %conv
}

; Test i64->f64.
define double @f2(i64 %i) {
; CHECK-LABEL: f2:
; CHECK: ldgr
; CHECK: adbr
; CHECK: br %r14
  %conv = uitofp i64 %i to double
  ret double %conv
}

; Test i64->f128.
define void @f3(i64 %i, ptr %dst) {
; CHECK-LABEL: f3:
; CHECK: cxgbr
; CHECK: axbr
; CHECK: br %r14
  %conv = uitofp i64 %i to fp128
  store fp128 %conv, ptr %dst
  ret void
}
