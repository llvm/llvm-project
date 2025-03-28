; RUN: llc < %s -mtriple=xcore -mcpu=xs1b-generic | FileCheck %s

; CHECK: .section .cp.rodata.cst4,"aMc",@progbits,4
; CHECK: .LCPI0_0:
; CHECK: .long 12345678
; CHECK-LABEL: f:
; CHECK: ldw r0, cp[.LCPI0_0]
define i32 @f() {
entry:
	ret i32 12345678
}

define i32 @g() {
entry:
; CHECK-LABEL: g:
; CHECK: mkmsk r0, 1
; CHECK: retsp 0
  ret i32 1;
}
