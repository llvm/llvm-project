; Check ah is not allocatable for register class gr8_norex2
; RUN: not llc < %s -mtriple=x86_64-unknown-unknown 2>&1 | FileCheck %s

define void @gr8_norex2() {
; CHECK: error: inline assembly requires more registers than available
  %1 = tail call i8 asm sideeffect "movb %r14b, $0", "=r,~{al},~{rbx},~{rcx},~{rdx},~{rdi},~{rsi},~{rbp},~{rsp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15},~{dirflag},~{fpsr},~{flags}"()
  ret void
}

