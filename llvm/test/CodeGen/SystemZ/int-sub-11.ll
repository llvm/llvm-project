; Test of subtraction that involves a constant as the first operand
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

; Check highest 16-bit signed int immediate value.
define i64 @f1(i64 %a) {
; CHECK-LABEL: f1:
; CHECK: lghi %r0, 32767
; CHECK: sgrk %r2, %r0, %r2
; CHECK: br %r14
  %sub = sub i64 32767, %a
  ret i64 %sub
}
; Check highest 32-bit signed int immediate value.
define i64 @f2(i64 %a) {
; CHECK-LABEL: f2:
; CHECK: lgfi %r0, 2147483647
; CHECK: sgrk %r2, %r0, %r2
; CHECK: br %r14
  %sub = sub i64 2147483647, %a
  ret i64 %sub
}
