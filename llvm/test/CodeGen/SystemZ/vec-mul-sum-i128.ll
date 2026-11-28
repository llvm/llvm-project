; Test that an i128 multiply-accumulate of two 64-bit products is done with
; VMSLG.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

define i128 @f1(i64 %a0, i64 %a1, i64 %b0, i64 %b1, i128 %c) {
; CHECK-LABEL: f1:
; CHECK:       vlvgp [[B:%v[0-9]+]], %r5, %r6
; CHECK-NEXT:  vlvgp [[A:%v[0-9]+]], %r3, %r4
; CHECK-NEXT:  vmslg [[R:%v[0-9]+]], [[A]], [[B]], {{%v[0-9]+}}, 0
; CHECK-NEXT:  vst [[R]], 0(%r2), 3
; CHECK-NEXT:  br %r14
  %za0 = zext i64 %a0 to i128
  %za1 = zext i64 %a1 to i128
  %zb0 = zext i64 %b0 to i128
  %zb1 = zext i64 %b1 to i128
  %m0 = mul i128 %za0, %zb0
  %m1 = mul i128 %za1, %zb1
  %s = add i128 %m0, %m1
  %r = add i128 %c, %s
  ret i128 %r
}

; A single product must not use VMSLG.
define i128 @f2(i64 %a, i64 %b, i128 %c) {
; CHECK-LABEL: f2:
; CHECK-NOT:   vmslg
; CHECK:       br %r14
  %za = zext i64 %a to i128
  %zb = zext i64 %b to i128
  %m = mul i128 %za, %zb
  %r = add i128 %m, %c
  ret i128 %r
}
