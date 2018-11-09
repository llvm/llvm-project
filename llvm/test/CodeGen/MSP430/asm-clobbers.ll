; RUN: llc < %s | FileCheck %s

target datalayout = "e-m:e-p:16:16-i32:16:32-a:16-n8:16"
target triple = "msp430---elf"

define void @test() {
entry:
; CHECK-LABEL: test:
; CHECK: push.w r10
  call void asm sideeffect "", "~{r10}"()
; CHECK: pop.w r10
  ret void
}
