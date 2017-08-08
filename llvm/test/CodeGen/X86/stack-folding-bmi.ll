; RUN: llc -O3 -disable-peephole -mtriple=x86_64-unknown-unknown -mattr=+bmi < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

; Stack reload folding tests.
;
; By including a nop call with sideeffects we can force a partial register spill of the
; relevant registers and check that the reload is correctly folded into the instruction.

define i32 @stack_fold_andn_u32(i32 %a0, i32 %a1) {
  ;CHECK-LABEL: stack_fold_andn_u32
  ;CHECK:       andnl {{-?[0-9]*}}(%rsp), %eax, %eax {{.*#+}} 4-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = xor i32 %a0, -1
  %3 = and i32 %a1, %2
  ret i32 %3
}

define i64 @stack_fold_andn_u64(i64 %a0, i64 %a1) {
  ;CHECK-LABEL: stack_fold_andn_u64
  ;CHECK:       andnq {{-?[0-9]*}}(%rsp), %rax, %rax {{.*#+}} 8-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = xor i64 %a0, -1
  %3 = and i64 %a1, %2
  ret i64 %3
}

define i32 @stack_fold_bextr_u32(i32 %a0, i32 %a1) {
  ;CHECK-LABEL: stack_fold_bextr_u32
  ;CHECK:       # BB#0:
  ;CHECK:       bextrl %eax, {{-?[0-9]*}}(%rsp), %eax {{.*#+}} 4-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = tail call i32 @llvm.x86.bmi.bextr.32(i32 %a0, i32 %a1)
  ret i32 %2
}
declare i32 @llvm.x86.bmi.bextr.32(i32, i32)

define i64 @stack_fold_bextr_u64(i64 %a0, i64 %a1) {
  ;CHECK-LABEL: stack_fold_bextr_u64
  ;CHECK:       # BB#0:
  ;CHECK:       bextrq %rax, {{-?[0-9]*}}(%rsp), %rax {{.*#+}} 8-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = tail call i64 @llvm.x86.bmi.bextr.64(i64 %a0, i64 %a1)
  ret i64 %2
}
declare i64 @llvm.x86.bmi.bextr.64(i64, i64)

define i32 @stack_fold_blsi_u32(i32 %a0) {
  ;CHECK-LABEL: stack_fold_blsi_u32
  ;CHECK:       blsil {{-?[0-9]*}}(%rsp), %eax {{.*#+}} 4-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = sub i32 0, %a0
  %3 = and i32 %2, %a0
  ret i32 %3
}

define i64 @stack_fold_blsi_u64(i64 %a0) {
  ;CHECK-LABEL: stack_fold_blsi_u64
  ;CHECK:       blsiq {{-?[0-9]*}}(%rsp), %rax {{.*#+}} 8-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = sub i64 0, %a0
  %3 = and i64 %2, %a0
  ret i64 %3
}

define i32 @stack_fold_blsmsk_u32(i32 %a0) {
  ;CHECK-LABEL: stack_fold_blsmsk_u32
  ;CHECK:       blsmskl {{-?[0-9]*}}(%rsp), %eax {{.*#+}} 4-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = sub i32 %a0, 1
  %3 = xor i32 %2, %a0
  ret i32 %3
}

define i64 @stack_fold_blsmsk_u64(i64 %a0) {
  ;CHECK-LABEL: stack_fold_blsmsk_u64
  ;CHECK:       blsmskq {{-?[0-9]*}}(%rsp), %rax {{.*#+}} 8-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = sub i64 %a0, 1
  %3 = xor i64 %2, %a0
  ret i64 %3
}

define i32 @stack_fold_blsr_u32(i32 %a0) {
  ;CHECK-LABEL: stack_fold_blsr_u32
  ;CHECK:       blsrl {{-?[0-9]*}}(%rsp), %eax {{.*#+}} 4-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = sub i32 %a0, 1
  %3 = and i32 %2, %a0
  ret i32 %3
}

define i64 @stack_fold_blsr_u64(i64 %a0) {
  ;CHECK-LABEL: stack_fold_blsr_u64
  ;CHECK:       blsrq {{-?[0-9]*}}(%rsp), %rax {{.*#+}} 8-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = sub i64 %a0, 1
  %3 = and i64 %2, %a0
  ret i64 %3
}

;TODO stack_fold_tzcnt_u16

define i32 @stack_fold_tzcnt_u32(i32 %a0) {
  ;CHECK-LABEL: stack_fold_tzcnt_u32
  ;CHECK:       tzcntl {{-?[0-9]*}}(%rsp), %eax {{.*#+}} 4-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = tail call i32 @llvm.cttz.i32(i32 %a0, i1 0)
  ret i32 %2
}
declare i32 @llvm.cttz.i32(i32, i1)

define i64 @stack_fold_tzcnt_u64(i64 %a0) {
  ;CHECK-LABEL: stack_fold_tzcnt_u64
  ;CHECK:       tzcntq {{-?[0-9]*}}(%rsp), %rax {{.*#+}} 8-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = tail call i64 @llvm.cttz.i64(i64 %a0, i1 0)
  ret i64 %2
}
declare i64 @llvm.cttz.i64(i64, i1)
