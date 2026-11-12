# RUN: llvm-mc -filetype=obj -triple x86_64 --x86-align-branch-boundary=32 --x86-align-branch=jcc+jmp+indirect+call+ret %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s

  # Exercise cases where mixed kinds of branch are asked to be aligned.

  .text
  .globl  labeled_mixed_test1
labeled_mixed_test1:
  .p2align  5
  .rept 30
  int3
  .endr
  # This call should have started at 0x1e and ends at 0x23, so two bytes of
  # padding are inserted before it.
  # CHECK:    20:          callq
  call  bar
  .rept 28
  int3
  .endr
  # If the previous call was not aligned, this jmp should have started at 0x3f
  # and need two bytes of padding. After the two bytes of padding are inserted
  # for the call, this jmp starts at 0xa1 and does not need padding.
  # CHECK:    41:          jmp
  jmp  *%rax

  .globl  labeled_mixed_test2
labeled_mixed_test2:
  .p2align  5
  .rept 30
  int3
  .endr
  # This jne should have started at 0x7e, so two bytes of padding are inserted
  # before it.
  # CHECK:    80:          jne
  jne bar
  .rept 28
  int3
  .endr
  # If the previous jne was not aligned, this jmp should have started at 0x3c.
  # After the two bytes of padding are inserted for the jne, this jmp should
  # have started at 0x9e, so two bytes of padding are inserted and it starts at
  # 0xa0.
  # CHECK:    a0:          jmp
  jmp bar

  .globl  labeled_mixed_test3
labeled_mixed_test3:
  .p2align 5
  .type   bar,@function
bar:
  # CHECK:    c0:          retq
  retq
