; Test loads of 128-bit floating-point constants in which the low bit of
; the significand is set.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s -check-prefix=CONST

define void @f1(ptr %x) {
; CHECK-LABEL: f1:
; CHECK: larl [[REGISTER:%r[1-5]+]], {{.*}}
; CHECK: ld %f0, 0([[REGISTER]])
; CHECK: ld %f2, 8([[REGISTER]])
; CHECK: std %f0, 0(%r2)
; CHECK: std %f2, 8(%r2)
; CHECK: br %r14
;
; CONST: .quad 0x3fff000000000000
; CONST: .quad 0x0000000000000001
  store fp128 0xL00000000000000013fff000000000000, ptr %x
  ret void
}
