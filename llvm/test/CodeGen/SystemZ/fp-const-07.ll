; Test loads of 128-bit floating-point constants that can be represented
; as 64-bit constants.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s -check-prefix=CONST

define void @f1(ptr %x) {
; CHECK-LABEL: f1:
; CHECK: larl [[REGISTER:%r[1-5]+]], {{.*}}
; CHECK: lxdb %f0, 0([[REGISTER]])
; CHECK: std %f0, 0(%r2)
; CHECK: std %f2, 8(%r2)
; CHECK: br %r14
;
; CONST: .quad 0x3ff0000010000000
  store fp128 0xL00000000000000003fff000001000000, ptr %x
  ret void
}
