; RUN: llc < %s -march=xcore -mcpu=xs1b-generic | FileCheck %s

define i32 *@addr_G() {
entry:
; CHECK: addr_G:
; CHECK: get r11, id
	ret i32* @G
}

@G = thread_local global i32 15
; CHECK: .section .dp.data,"awd",@progbits
; CHECK: G:
; CHECK: .long 15
; CHECK: .long 15
; CHECK: .long 15
; CHECK: .long 15
; CHECK: .long 15
; CHECK: .long 15
; CHECK: .long 15
; CHECK: .long 15
