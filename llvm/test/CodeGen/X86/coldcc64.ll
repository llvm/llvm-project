; RUN: llc < %s | FileCheck %s

target triple = "x86_64-linux-gnu"

define coldcc void @foo() {
; CHECK: pushq %rbp
; CHECK: pushq %r15
; CHECK: pushq %r14
; CHECK: pushq %r13
; CHECK: pushq %r12
; CHECK: pushq %r11
; CHECK: pushq %r10
; CHECK: pushq %r9
; CHECK: pushq %r8
; CHECK: pushq %rdi
; CHECK: pushq %rsi
; CHECK: pushq %rdx
; CHECK: pushq %rcx
; CHECK: pushq %rbx
; CHECK: movaps %xmm15
; CHECK: movaps %xmm0
  call void asm sideeffect "", "~{xmm15},~{xmm0},~{rbp},~{r15},~{r14},~{r13},~{r12},~{r11},~{r10},~{r9},~{r8},~{rdi},~{rsi},~{rdx},~{rcx},~{rbx}"()
  ret void
}
