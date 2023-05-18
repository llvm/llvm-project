; RUN: llc -mtriple thumbv7m-apple-macho -frame-pointer=all -o - %s | FileCheck %s

define void @func() {
; CHECK-LABEL: func:
; CHECK: push {r6, r7, lr}
; CHECK: add r7, sp, #4
  call void @bar()
  call void asm sideeffect "", "~{r11}"()
  ret void
}

declare void @bar()
