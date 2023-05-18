; Test loads of 64-bit floating-point constants that could be represented
; as 32-bit constants, but should not be.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s -check-prefix=CONST

define double @f1() {
; CHECK-LABEL: f1:
; CHECK: larl [[REGISTER:%r[1-5]]], {{.*}}
; CHECK: ld %f0, 0([[REGISTER]])
; CHECK: br %r14
;
; CONST: .quad	0x3ff0000020000000              # double 1.0000001192092896
  ret double 0x3ff0000020000000
}
