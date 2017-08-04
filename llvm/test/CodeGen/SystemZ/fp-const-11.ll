; Test loads of f128 floating-point constants on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s -check-prefix=CONST

; Test loading zero.
define void @f1(fp128 *%x) {
; CHECK-LABEL: f1:
; CHECK: vzero [[REG:%v[0-9]+]]
; CHECK: vst [[REG]], 0(%r2)
; CHECK: br %r14
  store fp128 0xL00000000000000000000000000000000, fp128 *%x
  ret void
}

; Test loading of negative floating-point zero.
define void @f2(fp128 *%x) {
; CHECK-LABEL: f2:
; CHECK: vzero [[REG:%v[0-9]+]]
; CHECK: wflnxb [[REG]], [[REG]]
; CHECK: vst [[REG]], 0(%r2)
; CHECK: br %r14
  store fp128 0xL00000000000000008000000000000000, fp128 *%x
  ret void
}

; Test loading of a 128-bit floating-point constant.  This value would
; actually fit within the 32-bit format, but we don't have extending
; loads into vector registers.
define void @f3(fp128 *%x) {
; CHECK-LABEL: f3:
; CHECK: larl [[REGISTER:%r[1-5]+]], {{.*}}
; CHECK: vl [[REG:%v[0-9]+]], 0([[REGISTER]])
; CHECK: vst [[REG]], 0(%r2)
; CHECK: br %r14
; CONST: .quad 4611404543484231680
; CONST: .quad 0
  store fp128 0xL00000000000000003fff000002000000, fp128 *%x
  ret void
}
