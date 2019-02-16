; RUN: llc -O3 -disable-peephole -mtriple=x86_64-unknown-unknown -mattr=+bmi,+bmi2 < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

; Stack reload folding tests.
;
; By including a nop call with sideeffects we can force a partial register spill of the
; relevant registers and check that the reload is correctly folded into the instruction.

define i32 @stack_fold_bzhi_u32(i32 %a0, i32 %a1)   {
  ;CHECK-LABEL: stack_fold_bzhi_u32
  ;CHECK:       bzhil %eax, {{-?[0-9]*}}(%rsp), %eax {{.*#+}} 4-byte Folded Reload
  %1 = tail call i32 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = tail call i32 @llvm.x86.bmi.bzhi.32(i32 %a0, i32 %a1)
  ret i32 %2
}
declare i32 @llvm.x86.bmi.bzhi.32(i32, i32)

define i64 @stack_fold_bzhi_u64(i64 %a0, i64 %a1)   {
  ;CHECK-LABEL: stack_fold_bzhi_u64
  ;CHECK:       bzhiq %rax, {{-?[0-9]*}}(%rsp), %rax {{.*#+}} 8-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = tail call i64 @llvm.x86.bmi.bzhi.64(i64 %a0, i64 %a1)
  ret i64 %2
}
declare i64 @llvm.x86.bmi.bzhi.64(i64, i64)

define i32 @stack_fold_pdep_u32(i32 %a0, i32 %a1)   {
  ;CHECK-LABEL: stack_fold_pdep_u32
  ;CHECK:       pdepl {{-?[0-9]*}}(%rsp), %eax, %eax {{.*#+}} 4-byte Folded Reload
  %1 = tail call i32 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = tail call i32 @llvm.x86.bmi.pdep.32(i32 %a0, i32 %a1)
  ret i32 %2
}
declare i32 @llvm.x86.bmi.pdep.32(i32, i32)

define i64 @stack_fold_pdep_u64(i64 %a0, i64 %a1)   {
  ;CHECK-LABEL: stack_fold_pdep_u64
  ;CHECK:       pdepq {{-?[0-9]*}}(%rsp), %rax, %rax {{.*#+}} 8-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = tail call i64 @llvm.x86.bmi.pdep.64(i64 %a0, i64 %a1)
  ret i64 %2
}
declare i64 @llvm.x86.bmi.pdep.64(i64, i64)

define i32 @stack_fold_pext_u32(i32 %a0, i32 %a1)   {
  ;CHECK-LABEL: stack_fold_pext_u32
  ;CHECK:       pextl {{-?[0-9]*}}(%rsp), %eax, %eax {{.*#+}} 4-byte Folded Reload
  %1 = tail call i32 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = tail call i32 @llvm.x86.bmi.pext.32(i32 %a0, i32 %a1)
  ret i32 %2
}
declare i32 @llvm.x86.bmi.pext.32(i32, i32)

define i64 @stack_fold_pext_u64(i64 %a0, i64 %a1)   {
  ;CHECK-LABEL: stack_fold_pext_u64
  ;CHECK:       pextq {{-?[0-9]*}}(%rsp), %rax, %rax {{.*#+}} 8-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = tail call i64 @llvm.x86.bmi.pext.64(i64 %a0, i64 %a1)
  ret i64 %2
}
declare i64 @llvm.x86.bmi.pext.64(i64, i64)
