; RUN: llc -O3 -disable-peephole -mtriple=x86_64-unknown-unknown -mattr=+adx < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

; Stack reload folding tests.
;
; By including a nop call with sideeffects we can force a partial register spill of the
; relevant registers and check that the reload is correctly folded into the instruction.

define i8 @stack_fold_addcarry_u32(i8 %a0, i32 %a1, i32 %a2, i8* %a3) {
  ;CHECK-LABEL: stack_fold_addcarry_u32
  ;CHECK:       adcxl {{-?[0-9]*}}(%rsp), %ecx {{.*#+}} 4-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = tail call i8 @llvm.x86.addcarry.u32(i8 %a0, i32 %a1, i32 %a2, i8* %a3)
  ret i8 %2;
}
declare i8 @llvm.x86.addcarry.u32(i8, i32, i32, i8*)

define i8 @stack_fold_addcarry_u64(i8 %a0, i64 %a1, i64 %a2, i8* %a3) {
  ;CHECK-LABEL: stack_fold_addcarry_u64
  ;CHECK:       adcxq {{-?[0-9]*}}(%rsp), %rcx {{.*#+}} 8-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = tail call i8 @llvm.x86.addcarry.u64(i8 %a0, i64 %a1, i64 %a2, i8* %a3)
  ret i8 %2;
}
declare i8 @llvm.x86.addcarry.u64(i8, i64, i64, i8*)

define i8 @stack_fold_addcarryx_u32(i8 %a0, i32 %a1, i32 %a2, i8* %a3) {
  ;CHECK-LABEL: stack_fold_addcarryx_u32
  ;CHECK:       adcxl {{-?[0-9]*}}(%rsp), %ecx {{.*#+}} 4-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = tail call i8 @llvm.x86.addcarryx.u32(i8 %a0, i32 %a1, i32 %a2, i8* %a3)
  ret i8 %2;
}
declare i8 @llvm.x86.addcarryx.u32(i8, i32, i32, i8*)

define i8 @stack_fold_addcarryx_u64(i8 %a0, i64 %a1, i64 %a2, i8* %a3) {
  ;CHECK-LABEL: stack_fold_addcarryx_u64
  ;CHECK:       adcxq {{-?[0-9]*}}(%rsp), %rcx {{.*#+}} 8-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = tail call i8 @llvm.x86.addcarryx.u64(i8 %a0, i64 %a1, i64 %a2, i8* %a3)
  ret i8 %2;
}
declare i8 @llvm.x86.addcarryx.u64(i8, i64, i64, i8*)
