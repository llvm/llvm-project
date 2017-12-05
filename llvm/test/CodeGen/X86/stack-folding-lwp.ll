; RUN: llc -O3 -disable-peephole -mtriple=x86_64-unknown-unknown -mattr=+lwp < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

; Stack reload folding tests.
;
; By including a nop call with sideeffects we can force a partial register spill of the
; relevant registers and check that the reload is correctly folded into the instruction.

define i8 @stack_fold_lwpins_u32(i32 %a0, i32 %a1) {
; CHECK-LABEL: stack_fold_lwpins_u32
; CHECK:       # %bb.0:
; CHECK:       lwpins $2814, {{-?[0-9]*}}(%rsp), %eax {{.*#+}} 4-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = tail call i8 @llvm.x86.lwpins32(i32 %a0, i32 %a1, i32 2814)
  ret i8 %2
}
declare i8 @llvm.x86.lwpins32(i32, i32, i32)

define i8 @stack_fold_lwpins_u64(i64 %a0, i32 %a1) {
; CHECK-LABEL: stack_fold_lwpins_u64
; CHECK:       # %bb.0:
; CHECK:       lwpins $2814, {{-?[0-9]*}}(%rsp), %rax {{.*#+}} 4-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = tail call i8 @llvm.x86.lwpins64(i64 %a0, i32 %a1, i32 2814)
  ret i8 %2
}
declare i8 @llvm.x86.lwpins64(i64, i32, i32)

define void @stack_fold_lwpval_u32(i32 %a0, i32 %a1) {
; CHECK-LABEL: stack_fold_lwpval_u32
; CHECK:       # %bb.0:
; CHECK:       lwpval $2814, {{-?[0-9]*}}(%rsp), %eax {{.*#+}} 4-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  tail call void @llvm.x86.lwpval32(i32 %a0, i32 %a1, i32 2814)
  ret void
}
declare void @llvm.x86.lwpval32(i32, i32, i32)

define void @stack_fold_lwpval_u64(i64 %a0, i32 %a1) {
; CHECK-LABEL: stack_fold_lwpval_u64
; CHECK:       # %bb.0:
; CHECK:       lwpval $2814, {{-?[0-9]*}}(%rsp), %rax {{.*#+}} 4-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  tail call void @llvm.x86.lwpval64(i64 %a0, i32 %a1, i32 2814)
  ret void
}
declare void @llvm.x86.lwpval64(i64, i32, i32)
