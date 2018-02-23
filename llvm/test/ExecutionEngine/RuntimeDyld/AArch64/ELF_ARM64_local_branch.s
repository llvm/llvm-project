# RUN: llvm-mc -triple=arm64-none-linux-gnu -filetype=obj -o %T/branch.o %s
# RUN: llvm-rtdyld -triple=arm64-none-linux-gnu -verify -check=%s %T/branch.o

.globl _main
.weak _label1

.section .text.1,"ax"
_label1:
  nop
_main:
  b _label1

## Branch 1 instruction back from _main
# rtdyld-check: *{4}(_main) = 0x17ffffff
