; Test loads of 128-bit floating-point constants that can be represented
; as 32-bit constants.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s -check-prefix=CONST

define void @f1(fp128 *%x) {
; CHECK-LABEL: f1:
; CHECK: larl [[REGISTER:%r[1-5]+]], {{.*}}
; CHECK: lxeb %f0, 0([[REGISTER]])
; CHECK: std %f0, 0(%r2)
; CHECK: std %f2, 8(%r2)
; CHECK: br %r14
;
; CONST: .long 0x3f800001
  store fp128 0xL00000000000000003fff000002000000, fp128 *%x
  ret void
}
