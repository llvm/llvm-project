  # RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu --x86-align-branch-boundary=32 --x86-align-branch=fused+jcc %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s

  # Exercise cases where fused instructions need to be aligned.

  .text
  .globl  foo
foo:
  .p2align  5
  .rept 30
  int3
  .endr
  # 'cmp  %rax, %rbp' is macro fused with 'jne foo',
  # so we need to align the pair.
  # CHECK:    20:          cmpq    %rax, %rbp
  # CHECK:    23:          jne
  cmp  %rax, %rbp
  jne foo
  int3

  .p2align  5
  .rept 28
  int3
  .endr
  # 'cmp  %rax, %rbp' is fusible but can not fused with `jo foo`,
  # so we only need to align 'jo foo'.
  # CHECK:    5c:          cmpq    %rax, %rbp
  cmp  %rax, %rbp
  # CHECK:    60:          jo
  jo foo
  int3

  .p2align  5
  .rept 26
  int3
  .endr
  # The second 'cmp  %rax, %rbp' is macro fused with 'jne foo'.
  cmp  %rax, %rbp
  # CHECK:    a0:          cmpq    %rax, %rbp
  # CHECK:    a3:          jne
  cmp  %rax, %rbp
  jne foo
  int3
