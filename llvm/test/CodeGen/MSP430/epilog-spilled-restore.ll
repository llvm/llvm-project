; RUN: llc < %s | FileCheck %s

target datalayout = "e-m:e-p:16:16-i32:16-i64:16-f32:16-f64:16-a:8-n8:16-S16"
target triple = "msp430"

; TODO check other counts
; TODO check 7 regs! (something strange happens...)

define void @pops_4_regs() nounwind {
; CHECK-LABEL: pops_4_regs
  call void asm sideeffect "", "~{r7},~{r8},~{r9},~{r10}"()
  ret void
; CHECK: jmp __mspabi_func_epilog_4
}
