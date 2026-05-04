; Check r16-r31 can not be used with 'q','r','l' constraint for backward compatibility.
; RUN: not llc -mtriple=x86_64 < %s 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: not llc -mtriple=x86_64 -mattr=+egpr < %s 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: llc -mtriple=x86_64 -mattr=+egpr,+inline-asm-use-gpr32 < %s | FileCheck %s

define void @q() {
; ERR: error: inline assembly requires more registers than available
; CHECK:    movq    %rax, %r16
  %a = call i64 asm sideeffect "movq %rax, $0", "=q,~{rax},~{rbx},~{rcx},~{rdx},~{rdi},~{rsi},~{rbp},~{rsp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  ret void
}

define void @r() {
; ERR: error: inline assembly requires more registers than available
; CHECK:    movq    %rax, %r16
  %a = call i64 asm sideeffect "movq %rax, $0", "=r,~{rax},~{rbx},~{rcx},~{rdx},~{rdi},~{rsi},~{rbp},~{rsp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  ret void
}

define void @l() {
; ERR: error: inline assembly requires more registers than available
; CHECK:    movq    %rax, %r16
  %a = call i64 asm sideeffect "movq %rax, $0", "=l,~{rax},~{rbx},~{rcx},~{rdx},~{rdi},~{rsi},~{rbp},~{rsp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  ret void
}

