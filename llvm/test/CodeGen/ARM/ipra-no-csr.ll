; RUN: llc -mtriple armv7a--none-eabi < %s              | FileCheck %s
; RUN: llc -mtriple armv7a--none-eabi < %s -enable-ipra | FileCheck %s

; Other targets disable callee-saved registers for internal functions when
; using IPRA, but that isn't profitable for ARM because the PUSH/POP
; instructions can more efficiently save registers than using individual
; LDR/STRs in the caller.

define internal void @callee() norecurse {
; CHECK-LABEL: callee:
entry:
; CHECK: push {r4, lr}
; CHECK: pop {r4, pc}
  tail call void asm sideeffect "", "~{r4}"()
  ret void
}

define void @caller() {
entry:
  call void @callee()
  ret void
}
