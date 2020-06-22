; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -x86-speculative-load-hardening | FileCheck %s

define i32 @foo(void ()** %0) {
; CHECK-LABEL: foo:
; CHECK:         callq *(%{{.*}})
; CHECK-NEXT:  .Lslh_ret_addr0:
; CHECK-NEXT:    movq %rsp, %rcx
; CHECK-NEXT:    movq -{{[0-9]+}}(%rsp), %rax
; CHECK-NEXT:    sarq $63, %rcx
; CHECK-NEXT:    cmpq $.Lslh_ret_addr0, %rax
  %2 = load void ()*, void ()** %0
  call void asm sideeffect "", "~{bx},~{cx},~{dx},~{bp},~{si},~{di},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15},~{dirflag},~{fpsr},~{flags}"()
  call void %2()
  ret i32 0
}
