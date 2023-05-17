; Test floating-point negation on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

; Test f32.
define float @f1(float %f) {
; CHECK-LABEL: f1:
; CHECK: lcdfr %f0, %f0
; CHECK: br %r14
  %res = fneg float %f
  ret float %res
}

; Test f64.
define double @f2(double %f) {
; CHECK-LABEL: f2:
; CHECK: lcdfr %f0, %f0
; CHECK: br %r14
  %res = fneg double %f
  ret double %res
}

; Test f128.  With the loads and stores, a pure negation would probably
; be better implemented using an XI on the upper byte.  Do some extra
; processing so that using FPRs is unequivocally better.
define void @f3(ptr %ptr, ptr %ptr2) {
; CHECK-LABEL: f3:
; CHECK-DAG: vl [[REG1:%v[0-9]+]], 0(%r2)
; CHECK-DAG: vl [[REG2:%v[0-9]+]], 0(%r3)
; CHECK-DAG: wflcxb [[NEGREG1:%v[0-9]+]], [[REG1]]
; CHECK: wfdxb [[RES:%v[0-9]+]], [[NEGREG1]], [[REG2]]
; CHECK: vst [[RES]], 0(%r2)
; CHECK: br %r14
  %orig = load fp128, ptr %ptr
  %negzero = fpext float -0.0 to fp128
  %neg = fneg fp128 %orig
  %op2 = load fp128, ptr %ptr2
  %res = fdiv fp128 %neg, %op2
  store fp128 %res, ptr %ptr
  ret void
}
