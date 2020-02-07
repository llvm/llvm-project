; Test loads of 64-bit floating-point constants that can be represented
; as 32-bit constants.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s -check-prefix=CONST

define double @f1() {
; CHECK-LABEL: f1:
; CHECK: larl [[REGISTER:%r[1-5]]], {{.*}}
; CHECK: ldeb %f0, 0([[REGISTER]])
; CHECK: br %r14
;
; CONST: .long 0x3f800001
  ret double 0x3ff0000020000000
}
