; Test loads of negative floating-point zero.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Test f32.
define float @f1() {
; CHECK-LABEL: f1:
; CHECK: lzer [[REGISTER:%f[0-5]+]]
; CHECK: lcdfr %f0, [[REGISTER]]
; CHECK: br %r14
  ret float -0.0
}

; Test f64.
define double @f2() {
; CHECK-LABEL: f2:
; CHECK: lzdr [[REGISTER:%f[0-5]+]]
; CHECK: lcdfr %f0, [[REGISTER]]
; CHECK: br %r14
  ret double -0.0
}

; Test f128.
define void @f3(ptr %x) {
; CHECK-LABEL: f3:
; CHECK: lzxr [[REGISTER:%f[0-5]+]]
; CHECK: lcxbr %f0, [[REGISTER]]
; CHECK: br %r14
  store fp128 0xL00000000000000008000000000000000, ptr %x
  ret void
}
