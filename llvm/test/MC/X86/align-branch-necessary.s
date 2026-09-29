# RUN: llvm-mc -filetype=obj -triple x86_64 --x86-align-branch-boundary=32 --x86-align-branch=fused+jcc+indirect+call+ret %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s

  # Check if no instruction crosses or is against the boundary,
  # there will be no extra padding.

  .text
  .globl  foo
  .p2align  5
foo:
  .p2align  5
  .rept 24
  int3
  .endr
  # CHECK:    18:          js
  js foo

  .p2align  5
  .rept 24
  int3
  .endr
  # CHECK:    38:          jmp
  jmp foo

  .p2align  5
  .rept 24
  int3
  .endr
  # CHECK:    58:          jmpq    *%rcx
  jmp  *%rcx


  .p2align  5
  .rept 24
  int3
  .endr
  # CHECK:    78:          callq
  call  foo

  .p2align  5
  .rept 27
  int3
  .endr
  # CHECK:    9b:          retq    $0
  ret $0


  .p2align  5
  .rept 21
  int3
  .endr
  # CHECK:    b5:          cmpq    %rax, %rbp
  # CHECK:    b8:          je
  cmp  %rax, %rbp
  je  foo
