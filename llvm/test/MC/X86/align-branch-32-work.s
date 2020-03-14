  # RUN: llvm-mc -filetype=obj -triple i386-pc-linux-gnu --x86-align-branch-boundary=32 --x86-align-branch=jmp %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s

  # Check the branch align mechanism can work in 32-bit mode.

  .text
  .rept 31
  int3
  .endr
  # CHECK:    20:          jmp
  jmp bar


  .type   bar,@function
bar:
  ret
