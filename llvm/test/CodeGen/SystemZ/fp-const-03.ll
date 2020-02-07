; Test loads of 32-bit floating-point constants.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s -check-prefix=CONST

define float @f1() {
; CHECK-LABEL: f1:
; CHECK: larl [[REGISTER:%r[1-5]]], {{.*}}
; CHECK: le %f0, 0([[REGISTER]])
; CHECK: br %r14
;
; CONST: .long 0x3f800001
  ret float 0x3ff0000020000000
}
