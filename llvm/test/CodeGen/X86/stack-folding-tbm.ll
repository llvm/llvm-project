; RUN: llc -O3 -disable-peephole -mtriple=x86_64-unknown-unknown -mattr=+bmi,+tbm < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

; Stack reload folding tests.
;
; By including a nop call with sideeffects we can force a partial register spill of the
; relevant registers and check that the reload is correctly folded into the instruction.

define i32 @stack_fold_bextri_u32(i32 %a0) {
  ;CHECK-LABEL: stack_fold_bextri_u32
  ;CHECK:       # %bb.0:
  ;CHECK:       bextrl $3841, {{-?[0-9]*}}(%rsp), %eax {{.*#+}} 4-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = tail call i32 @llvm.x86.tbm.bextri.u32(i32 %a0, i32 3841)
  ret i32 %2
}
declare i32 @llvm.x86.tbm.bextri.u32(i32, i32)

define i64 @stack_fold_bextri_u64(i64 %a0) {
  ;CHECK-LABEL: stack_fold_bextri_u64
  ;CHECK:       # %bb.0:
  ;CHECK:       bextrq $3841, {{-?[0-9]*}}(%rsp), %rax {{.*#+}} 8-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = tail call i64 @llvm.x86.tbm.bextri.u64(i64 %a0, i64 3841)
  ret i64 %2
}
declare i64 @llvm.x86.tbm.bextri.u64(i64, i64)

define i32 @stack_fold_blcfill_u32(i32 %a0) {
  ;CHECK-LABEL: stack_fold_blcfill_u32
  ;CHECK:       blcfilll {{-?[0-9]*}}(%rsp), %eax {{.*#+}} 4-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = add i32 %a0, 1
  %3 = and i32 %a0, %2
  ret i32 %3
}

define i64 @stack_fold_blcfill_u64(i64 %a0) {
  ;CHECK-LABEL: stack_fold_blcfill_u64
  ;CHECK:       blcfillq {{-?[0-9]*}}(%rsp), %rax {{.*#+}} 8-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = add i64 %a0, 1
  %3 = and i64 %a0, %2
  ret i64 %3
}

define i32 @stack_fold_blci_u32(i32 %a0) {
  ;CHECK-LABEL: stack_fold_blci_u32
  ;CHECK:       blcil {{-?[0-9]*}}(%rsp), %eax {{.*#+}} 4-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = add i32 %a0, 1
  %3 = xor i32 %2, -1
  %4 = or i32 %a0, %3
  ret i32 %4
}

define i64 @stack_fold_blci_u64(i64 %a0) {
  ;CHECK-LABEL: stack_fold_blci_u64
  ;CHECK:       blciq {{-?[0-9]*}}(%rsp), %rax {{.*#+}} 8-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = add i64 %a0, 1
  %3 = xor i64 %2, -1
  %4 = or i64 %a0, %3
  ret i64 %4
}

define i32 @stack_fold_blcic_u32(i32 %a0) {
  ;CHECK-LABEL: stack_fold_blcic_u32
  ;CHECK:       blcicl {{-?[0-9]*}}(%rsp), %eax {{.*#+}} 4-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = add i32 %a0, 1
  %3 = xor i32 %a0, -1
  %4 = and i32 %2, %3
  ret i32 %4
}

define i64 @stack_fold_blcic_u64(i64 %a0) {
  ;CHECK-LABEL: stack_fold_blcic_u64
  ;CHECK:       blcicq {{-?[0-9]*}}(%rsp), %rax {{.*#+}} 8-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = add i64 %a0, 1
  %3 = xor i64 %a0, -1
  %4 = and i64 %2, %3
  ret i64 %4
}

define i32 @stack_fold_blcmsk_u32(i32 %a0) {
  ;CHECK-LABEL: stack_fold_blcmsk_u32
  ;CHECK:       blcmskl {{-?[0-9]*}}(%rsp), %eax {{.*#+}} 4-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = add i32 %a0, 1
  %3 = xor i32 %a0, %2
  ret i32 %3
}

define i64 @stack_fold_blcmsk_u64(i64 %a0) {
  ;CHECK-LABEL: stack_fold_blcmsk_u64
  ;CHECK:       blcmskq {{-?[0-9]*}}(%rsp), %rax {{.*#+}} 8-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = add i64 %a0, 1
  %3 = xor i64 %a0, %2
  ret i64 %3
}

define i32 @stack_fold_blcs_u32(i32 %a0) {
  ;CHECK-LABEL: stack_fold_blcs_u32
  ;CHECK:       blcsl {{-?[0-9]*}}(%rsp), %eax {{.*#+}} 4-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = add i32 %a0, 1
  %3 = or i32 %a0, %2
  ret i32 %3
}

define i64 @stack_fold_blcs_u64(i64 %a0) {
  ;CHECK-LABEL: stack_fold_blcs_u64
  ;CHECK:       blcsq {{-?[0-9]*}}(%rsp), %rax {{.*#+}} 8-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = add i64 %a0, 1
  %3 = or i64 %a0, %2
  ret i64 %3
}

define i32 @stack_fold_blsfill_u32(i32 %a0) {
  ;CHECK-LABEL: stack_fold_blsfill_u32
  ;CHECK:       blsfilll {{-?[0-9]*}}(%rsp), %eax {{.*#+}} 4-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = sub i32 %a0, 1
  %3 = or i32 %a0, %2
  ret i32 %3
}

define i64 @stack_fold_blsfill_u64(i64 %a0) {
  ;CHECK-LABEL: stack_fold_blsfill_u64
  ;CHECK:       blsfillq {{-?[0-9]*}}(%rsp), %rax {{.*#+}} 8-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = sub i64 %a0, 1
  %3 = or i64 %a0, %2
  ret i64 %3
}

define i32 @stack_fold_blsic_u32(i32 %a0) {
  ;CHECK-LABEL: stack_fold_blsic_u32
  ;CHECK:       blsicl {{-?[0-9]*}}(%rsp), %eax {{.*#+}} 4-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = sub i32 %a0, 1
  %3 = xor i32 %a0, -1
  %4 = or i32 %2, %3
  ret i32 %4
}

define i64 @stack_fold_blsic_u64(i64 %a0) {
  ;CHECK-LABEL: stack_fold_blsic_u64
  ;CHECK:       blsicq {{-?[0-9]*}}(%rsp), %rax {{.*#+}} 8-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = sub i64 %a0, 1
  %3 = xor i64 %a0, -1
  %4 = or i64 %2, %3
  ret i64 %4
}

define i32 @stack_fold_t1mskc_u32(i32 %a0) {
  ;CHECK-LABEL: stack_fold_t1mskc_u32
  ;CHECK:       t1mskcl {{-?[0-9]*}}(%rsp), %eax {{.*#+}} 4-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = add i32 %a0, 1
  %3 = xor i32 %a0, -1
  %4 = or i32 %2, %3
  ret i32 %4
}

define i64 @stack_fold_t1mskc_u64(i64 %a0) {
  ;CHECK-LABEL: stack_fold_t1mskc_u64
  ;CHECK:       t1mskcq {{-?[0-9]*}}(%rsp), %rax {{.*#+}} 8-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = add i64 %a0, 1
  %3 = xor i64 %a0, -1
  %4 = or i64 %2, %3
  ret i64 %4
}

define i32 @stack_fold_tzmsk_u32(i32 %a0) {
  ;CHECK-LABEL: stack_fold_tzmsk_u32
  ;CHECK:       tzmskl {{-?[0-9]*}}(%rsp), %eax {{.*#+}} 4-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = sub i32 %a0, 1
  %3 = xor i32 %a0, -1
  %4 = and i32 %2, %3
  ret i32 %4
}

define i64 @stack_fold_tzmsk_u64(i64 %a0) {
  ;CHECK-LABEL: stack_fold_tzmsk_u64
  ;CHECK:       tzmskq {{-?[0-9]*}}(%rsp), %rax {{.*#+}} 8-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = sub i64 %a0, 1
  %3 = xor i64 %a0, -1
  %4 = and i64 %2, %3
  ret i64 %4
}
