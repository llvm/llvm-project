; RUN: llc -O3 -disable-peephole -mtriple=x86_64-unknown-unknown -mattr=+adx < %s | FileCheck %s --check-prefix=CHECK --check-prefix=ADX
; RUN: llc -O3 -disable-peephole -mtriple=x86_64-unknown-unknown -mattr=-adx < %s | FileCheck %s --check-prefix=CHECK --check-prefix=NOADX

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

; Stack reload folding tests.
;
; By including a nop call with sideeffects we can force a partial register spill of the
; relevant registers and check that the reload is correctly folded into the instruction.

define i8 @stack_fold_addcarry_u32(i8 %a0, i32 %a1, i32 %a2, i8* %a3) {
  ;CHECK-LABEL: stack_fold_addcarry_u32
  ;CHECK:       adcl {{-?[0-9]*}}(%rsp), %{{.*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = call { i8, i32 } @llvm.x86.addcarry.32(i8 %a0, i32 %a1, i32 %a2)
  %3 = extractvalue { i8, i32 } %2, 1
  %4 = bitcast i8* %a3 to i32*
  store i32 %3, i32* %4, align 1
  %5 = extractvalue { i8, i32 } %2, 0
  ret i8 %5
}

define i8 @stack_fold_addcarry_u64(i8 %a0, i64 %a1, i64 %a2, i8* %a3) {
  ;CHECK-LABEL: stack_fold_addcarry_u64
  ;CHECK:       adcq {{-?[0-9]*}}(%rsp), %{{.*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = call { i8, i64 } @llvm.x86.addcarry.64(i8 %a0, i64 %a1, i64 %a2)
  %3 = extractvalue { i8, i64 } %2, 1
  %4 = bitcast i8* %a3 to i64*
  store i64 %3, i64* %4, align 1
  %5 = extractvalue { i8, i64 } %2, 0
  ret i8 %5
}

define i8 @stack_fold_addcarryx_u32(i8 %a0, i32 %a1, i32 %a2, i8* %a3) {
  ;CHECK-LABEL: stack_fold_addcarryx_u32
  ;CHECK:       adcl {{-?[0-9]*}}(%rsp), %{{.*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = call { i8, i32 } @llvm.x86.addcarry.32(i8 %a0, i32 %a1, i32 %a2)
  %3 = extractvalue { i8, i32 } %2, 1
  %4 = bitcast i8* %a3 to i32*
  store i32 %3, i32* %4, align 1
  %5 = extractvalue { i8, i32 } %2, 0
  ret i8 %5
}

define i8 @stack_fold_addcarryx_u64(i8 %a0, i64 %a1, i64 %a2, i8* %a3) {
  ;CHECK-LABEL: stack_fold_addcarryx_u64
  ;CHECK:       adcq {{-?[0-9]*}}(%rsp), %{{.*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = call { i8, i64 } @llvm.x86.addcarry.64(i8 %a0, i64 %a1, i64 %a2)
  %3 = extractvalue { i8, i64 } %2, 1
  %4 = bitcast i8* %a3 to i64*
  store i64 %3, i64* %4, align 1
  %5 = extractvalue { i8, i64 } %2, 0
  ret i8 %5
}

define i8 @stack_fold_subborrow_u32(i8 %a0, i32 %a1, i32 %a2, i8* %a3) {
  ;CHECK-LABEL: stack_fold_subborrow_u32
  ;CHECK:       sbbl {{-?[0-9]*}}(%rsp), %{{.*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = call { i8, i32 } @llvm.x86.subborrow.32(i8 %a0, i32 %a1, i32 %a2)
  %3 = extractvalue { i8, i32 } %2, 1
  %4 = bitcast i8* %a3 to i32*
  store i32 %3, i32* %4, align 1
  %5 = extractvalue { i8, i32 } %2, 0
  ret i8 %5
}

define i8 @stack_fold_subborrow_u64(i8 %a0, i64 %a1, i64 %a2, i8* %a3) {
  ;CHECK-LABEL: stack_fold_subborrow_u64
  ;CHECK:       sbbq {{-?[0-9]*}}(%rsp), %{{.*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = call { i8, i64 } @llvm.x86.subborrow.64(i8 %a0, i64 %a1, i64 %a2)
  %3 = extractvalue { i8, i64 } %2, 1
  %4 = bitcast i8* %a3 to i64*
  store i64 %3, i64* %4, align 1
  %5 = extractvalue { i8, i64 } %2, 0
  ret i8 %5
}

declare { i8, i32 } @llvm.x86.addcarry.32(i8, i32, i32)
declare { i8, i64 } @llvm.x86.addcarry.64(i8, i64, i64)
declare { i8, i32 } @llvm.x86.subborrow.32(i8, i32, i32)
declare { i8, i64 } @llvm.x86.subborrow.64(i8, i64, i64)
