; RUN: llc < %s -O0 -frame-pointer=all -relocation-model=pic
; PR7313
;
; The inline asm in this function clobbers almost all allocatable registers.
; Make sure that the register allocator recovers.
;
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

declare void @snapshot()

define void @test_too_many_longs() nounwind {
entry:
  call void asm sideeffect "xor %rax, %rax\0A\09xor %rbx, %rbx\0A\09xor %rcx, %rcx\0A\09xor %rdx, %rdx\0A\09xor %rsi, %rsi\0A\09xor %rdi, %rdi\0A\09xor %r8, %r8\0A\09xor %r9, %r9\0A\09xor %r10, %r10\0A\09xor %r11, %r11\0A\09xor %r12, %r12\0A\09xor %r13, %r13\0A\09xor %r14, %r14\0A\09xor %r15, %r15\0A\09", "~{fpsr},~{flags},~{r15},~{r14},~{r13},~{r12},~{r11},~{r10},~{r9},~{r8},~{rdi},~{rsi},~{rdx},~{rcx},~{rbx},~{rax}"() nounwind
  call void bitcast (void ()* @snapshot to void (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)*)(i64 32, i64 33, i64 34, i64 35, i64 36, i64 37, i64 38, i64 39, i64 40, i64 41, i64 42, i64 43) nounwind
  ret void
}
