; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z16 | FileCheck %s
;
; Check that a multiply-and-add  *not* result for half.

define half @f1(half %arg, half %A2, half %A3) {
; CHECK-LABEL: f1:
; CHECK: brasl   %r14, __extendhfsf2@PLT
; CHECK: brasl   %r14, __extendhfsf2@PLT
; CHECK: meebr   %f0, %f10
; CHECK: brasl   %r14, __truncsfhf2@PLT
; CHECK: brasl   %r14, __extendhfsf2@PLT
; CHECK: brasl   %r14, __extendhfsf2@PLT
; CHECK: wfasb   %f0, %f9, %f0
; CHECK: brasl   %r14, __truncsfhf2@PLT

bb:
  %i = fmul contract half %arg, %A2
  %i4 = fadd contract half %i, %A3
  ret half %i4
}
