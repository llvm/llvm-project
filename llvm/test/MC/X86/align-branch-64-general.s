  # RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu --x86-align-branch-boundary=32 --x86-align-branch=fused+jcc+jmp %s -o %t1
  # RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu --x86-branches-within-32B-boundaries %s -o %t2
  # RUN: cmp %t1 %t2

  # Check the general option --x86-branches-within-32B-boundaries is equivelent
  # to the fined options --x86-align-branch-boundary=32 --x86-align-branch=fused+jcc+jmp.

  .text
  .globl  foo
  .p2align  5
foo:
  .p2align  5
  .rept 30
  int3
  .endr
  js foo

  .p2align  5
  .rept 30
  int3
  .endr
  jmp foo

  .p2align  5
  .rept 30
  int3
  .endr
  jmp  *%rcx


  .p2align  5
  .rept 30
  int3
  .endr
  call  foo

  .p2align  5
  .rept 30
  int3
  .endr
  ret $0


  .p2align  5
  .rept 29
  int3
  .endr
  cmp  %rax, %rbp
  je  foo
