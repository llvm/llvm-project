; RUN: llc -O0 --frame-pointer=all -mtriple=aarch64-- -o - %S/../Inputs/stack-guard-reassign.ll | FileCheck %s

; Verify that the offset assigned to the stack protector is at the top of the
; frame, covering the locals.
; CHECK-LABEL: fn:
; CHECK:      sub x8, x29, #24
; CHECK-NEXT: adrp x9, __stack_chk_guard
; CHECK-NEXT: ldr x9, [x9, :lo12:__stack_chk_guard]
; CHECK-NEXT: str x9, [x8]
