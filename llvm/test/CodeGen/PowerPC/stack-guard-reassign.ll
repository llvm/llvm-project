; RUN: llc -O0 --frame-pointer=none -mtriple=powerpc-- -o - %S/../Inputs/stack-guard-reassign.ll | FileCheck %s

; Verify that the offset assigned to the stack protector is at the top of the
; frame, covering the locals.
; CHECK-LABEL: fn:
; CHECK:      mflr 0
; CHECK-NEXT: stw 0, 4(1)
; CHECK-NEXT: lis 0, -2
; CHECK-NEXT: ori 0, 0, 65488
; CHECK-NEXT: stwux 1, 1, 0
; CHECK-NEXT: sub 0, 1, 0
; CHECK-NEXT: lis 4, __stack_chk_guard@ha
; CHECK-NEXT: lwz 5, __stack_chk_guard@l(4)
; CHECK-NEXT: lis 6, 1
; CHECK-NEXT: ori 6, 6, 44
; CHECK-NEXT: stwx 5, 1, 6
