; Test loads of 128-bit floating-point constants.  This value would actually
; fit within the x86 80-bit format, so the test make sure we don't try to
; extend from an f80.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s -check-prefix=CONST

define void @f1(fp128 *%x) {
; CHECK-LABEL: f1:
; CHECK: larl [[REGISTER:%r[1-5]+]], {{.*}}
; CHECK: ld %f0, 0([[REGISTER]])
; CHECK: ld %f2, 8([[REGISTER]])
; CHECK: std %f0, 0(%r2)
; CHECK: std %f2, 8(%r2)
; CHECK: br %r14
;
; CONST: .quad 0x3fff000000000000
; CONST: .quad 0x0800000000000000
  store fp128 0xL08000000000000003fff000000000000, fp128 *%x
  ret void
}
