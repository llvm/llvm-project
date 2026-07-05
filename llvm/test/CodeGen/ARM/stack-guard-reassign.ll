; RUN: llc -O0 --frame-pointer=none -mtriple=arm-- -o - %S/../Inputs/stack-guard-reassign.ll | FileCheck %s

; Verify that the offset assigned to the stack protector is at the top of the
; frame, covering the locals.
; CHECK-LABEL: fn:
; CHECK:      sub sp, sp, #16
; CHECK-NEXT: sub sp, sp, #65536
; CHECK-NEXT: ldr r1, .LCPI0_0
; CHECK-NEXT: ldr r1, [r1]
; CHECK-NEXT: add lr, sp, #65536
; CHECK-NEXT: str r1, [lr, #12]
; CHECK: .LCPI0_0:
; CHECK-NEXT: .long __stack_chk_guard
