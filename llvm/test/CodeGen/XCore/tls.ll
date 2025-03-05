; RUN: llc < %s -mtriple=xcore -mcpu=xs1b-generic | FileCheck %s

define ptr @addr_G() {
entry:
; CHECK-LABEL: addr_G:
; CHECK: get r11, id
	ret ptr @G
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
