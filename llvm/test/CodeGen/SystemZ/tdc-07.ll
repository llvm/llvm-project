; Test the Test Data Class instruction on z14
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

declare i32 @llvm.s390.tdc.f128(fp128, i64)

; Check using as i32 - f128
define i32 @f3(fp128 %x) {
; CHECK-LABEL: f3
; CHECK: vl %v0, 0(%r2)
; CHECK: vrepg  %v2, %v0, 1
; CHECK: tcxb %f0, 123
; CHECK: ipm %r2
; CHECK: srl %r2, 28
  %res = call i32 @llvm.s390.tdc.f128(fp128 %x, i64 123)
  ret i32 %res
}

