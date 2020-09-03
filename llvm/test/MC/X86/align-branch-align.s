# RUN: llvm-mc -filetype=obj -triple x86_64 --x86-align-branch-boundary=32 --x86-align-branch=fused+jcc+call %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s

  # Exercise some corner cases related to align directive.

  .text
  # Check the align directive between the macro fused pair
  # does not make code crazy.
  # CHECK:     0:          cmpq    %rax, %rbp
  # CHECK:     3:          nop
  # CHECK:     4:          jne
  cmp  %rax, %rbp
  .p2align  1
  jne bar

  .rept 24
  int3
  .endr
  .p2align  1
  # Check we can ensure this call not cross or end at boundary when there
  # is a align directive before it.
  # CHECK:    20:          callq    *%rax
  call  *%rax

  .type   bar,@function
bar:
  retq
