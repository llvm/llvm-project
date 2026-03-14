; RUN: llc < %s | FileCheck %s
target datalayout = "e-p:16:8:8-i8:8:8-i16:8:8-i32:8:8"
target triple = "msp430-generic-generic"

; Test that correct register names are accepted *inside* inline asm listings.
; Tested with PUSH instruction since it does not support memory operands.

define void @accepted_rN() nounwind {
; CHECK-LABEL: accepted_rN
  call void asm sideeffect "push r0", ""() nounwind
; CHECK: push r0
  call void asm sideeffect "push r1", ""() nounwind
; CHECK: push r1
  call void asm sideeffect "push r2", ""() nounwind
; CHECK: push r2
  call void asm sideeffect "push r3", ""() nounwind
; CHECK: push r3
  call void asm sideeffect "push r4", ""() nounwind
; CHECK: push r4
  call void asm sideeffect "push r5", ""() nounwind
; CHECK: push r5
  call void asm sideeffect "push r6", ""() nounwind
; CHECK: push r6
  call void asm sideeffect "push r7", ""() nounwind
; CHECK: push r7
  call void asm sideeffect "push r8", ""() nounwind
; CHECK: push r8
  call void asm sideeffect "push r9", ""() nounwind
; CHECK: push r9
  call void asm sideeffect "push r10", ""() nounwind
; CHECK: push r10
  call void asm sideeffect "push r11", ""() nounwind
; CHECK: push r11
  call void asm sideeffect "push r12", ""() nounwind
; CHECK: push r12
  call void asm sideeffect "push r13", ""() nounwind
; CHECK: push r13
  call void asm sideeffect "push r14", ""() nounwind
; CHECK: push r14
  call void asm sideeffect "push r15", ""() nounwind
; CHECK: push r15
  ret void
}

define void @accepted_reg_aliases() nounwind {
; CHECK-LABEL: accepted_reg_aliases
; Ensure register aliases are renamed as expected
call void asm sideeffect "push pc", ""() nounwind
; CHECK: push r0
call void asm sideeffect "push sp", ""() nounwind
; CHECK: push r1
call void asm sideeffect "push sr", ""() nounwind
; CHECK: push r2
call void asm sideeffect "push cg", ""() nounwind
; CHECK: push r3
call void asm sideeffect "push fp", ""() nounwind
; CHECK: push r4
        ret void
}
