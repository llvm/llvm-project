; RUN: llc -O0 --frame-pointer=none -mtriple=powerpc-- -o - %S/../Inputs/stack-guard-reassign.ll | FileCheck %s

; Verify that the offset assigned to the stack protector is at the top of the
; frame, covering the locals.
; CHECK-LABEL: fn:
; CHECK:      mflr 0
; CHECK-NEXT: stw 0, 4(1)
; CHECK-NEXT: lis 0, -2
; CHECK-NEXT: ori 0, 0, 65488
; CHECK-NEXT: stwux 1, 1, 0
; CHECK-NEXT: subf 0, 0, 1
; CHECK-NEXT: lis 4, 1
; CHECK-NEXT: ori 4, 4, 44
; CHECK-NEXT: add 4, 1, 4
; CHECK-NEXT: lis 5, __stack_chk_guard@ha
; CHECK-NEXT: lwz 6, __stack_chk_guard@l(5)
; CHECK-NEXT: stw 6, 0(4)
