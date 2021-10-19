; RUN: llc -O0 --frame-pointer=none -mtriple=powerpc-- -o - %S/../Inputs/stack-guard-reassign.ll | FileCheck %s

; Verify that the offset assigned to the stack protector is at the top of the
; frame, covering the locals.
; CHECK-LABEL: fn:
; CHECK:      mflr 0
; CHECK-NEXT: stw 0, 4(1)
; CHECK-NEXT: lis 0, -2
; CHECK-NEXT: ori 0, 0, 65504
; CHECK-NEXT: stwux 1, 1, 0
; CHECK-NEXT: sub 0, 1, 0
; CHECK-NEXT: lis 4, __stack_chk_guard@ha
; CHECK-NEXT: stw 4, 16(1)
; CHECK-NEXT: lwz 4, __stack_chk_guard@l(4)
; CHECK-NEXT: lis 5, 1
; CHECK-NEXT: ori 5, 5, 28
; CHECK-NEXT: stwx 4, 1, 5
