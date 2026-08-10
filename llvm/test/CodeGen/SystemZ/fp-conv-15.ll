; Test f128 floating-point truncations/extensions on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

; Test f128->f64.
define double @f1(ptr %ptr) {
; CHECK-LABEL: f1:
; CHECK: vl [[REG:%v[0-9]+]], 0(%r2)
; CHECK: wflrx %f0, [[REG]], 0, 0
; CHECK: br %r14
  %val = load fp128, ptr %ptr
  %res = fptrunc fp128 %val to double
  ret double %res
}

; Test f128->f32.
define float @f2(ptr %ptr) {
; CHECK-LABEL: f2:
; CHECK: vl [[REG:%v[0-9]+]], 0(%r2)
; CHECK: wflrx %f0, [[REG]], 0, 3
; CHECK: ledbra %f0, 0, %f0, 0
; CHECK: br %r14
  %val = load fp128, ptr %ptr
  %res = fptrunc fp128 %val to float
  ret float %res
}

; Test f64->f128.
define void @f3(ptr %dst, double %val) {
; CHECK-LABEL: f3:
; CHECK: wflld [[RES:%v[0-9]+]], %f0
; CHECK: vst [[RES]], 0(%r2)
; CHECK: br %r14
  %res = fpext double %val to fp128
  store fp128 %res, ptr %dst
  ret void
}

; Test f32->f128.
define void @f4(ptr %dst, float %val) {
; CHECK-LABEL: f4:
; CHECK: ldebr %f0, %f0
; CHECK: wflld [[RES:%v[0-9]+]], %f0
; CHECK: vst [[RES]], 0(%r2)
; CHECK: br %r14
  %res = fpext float %val to fp128
  store fp128 %res, ptr %dst
  ret void
}

