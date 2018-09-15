; Test floating-point negation.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test f32.
define float @f1(float %f) {
; CHECK-LABEL: f1:
; CHECK: lcdfr %f0, %f0
; CHECK: br %r14
  %res = fsub float -0.0, %f
  ret float %res
}

; Test f64.
define double @f2(double %f) {
; CHECK-LABEL: f2:
; CHECK: lcdfr %f0, %f0
; CHECK: br %r14
  %res = fsub double -0.0, %f
  ret double %res
}

; Test f128.  With the loads and stores, a pure negation would probably
; be better implemented using an XI on the upper byte.  Do some extra
; processing so that using FPRs is unequivocally better.
define void @f3(fp128 *%ptr, fp128 *%ptr2) {
; CHECK-LABEL: f3:
; CHECK: lcxbr
; CHECK: dxbr
; CHECK: br %r14
  %orig = load fp128, fp128 *%ptr
  %negzero = fpext float -0.0 to fp128
  %neg = fsub fp128 0xL00000000000000008000000000000000, %orig
  %op2 = load fp128, fp128 *%ptr2
  %res = fdiv fp128 %neg, %op2
  store fp128 %res, fp128 *%ptr
  ret void
}
