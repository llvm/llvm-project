; Check r16-r31 can not be used with 'q','r','l' constraint for backward compatibility.
; RUN: not llc < %s -mtriple=x86_64-unknown-unknown -mattr=+egpr 2>&1 | FileCheck %s

define void @q() {
; CHECK: error: inline assembly requires more registers than available
  %a = call i32 asm sideeffect "movq %rax, $0", "=q,~{rax},~{rbx},~{rcx},~{rdx},~{rdi},~{rsi},~{rbp},~{rsp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  ret void
}

define void @r() {
; CHECK: error: inline assembly requires more registers than available
  %a = call i32 asm sideeffect "movq %rax, $0", "=r,~{rax},~{rbx},~{rcx},~{rdx},~{rdi},~{rsi},~{rbp},~{rsp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  ret void
}

define void @l() {
; CHECK: error: inline assembly requires more registers than available
  %a = call i32 asm sideeffect "movq %rax, $0", "=l,~{rax},~{rbx},~{rcx},~{rdx},~{rdi},~{rsi},~{rbp},~{rsp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  ret void
}

