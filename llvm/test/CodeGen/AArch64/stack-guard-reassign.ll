; RUN: llc -O0 --frame-pointer=all -mtriple=aarch64-- -o - %S/../Inputs/stack-guard-reassign.ll | FileCheck %s

; Verify that the offset assigned to the stack protector is at the top of the
; frame, covering the locals.
; CHECK-LABEL: fn:
; CHECK:      adrp [[REG:x[0-9]+]], __stack_chk_guard
; CHECK-NEXT: ldr [[REG]], [[[REG]], :lo12:__stack_chk_guard]
; CHECK-NEXT: stur [[REG]], [x29, #-8]
