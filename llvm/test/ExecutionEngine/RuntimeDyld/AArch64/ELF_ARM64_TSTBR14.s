# RUN: llvm-mc -triple=arm64-none-linux-gnu -filetype=obj -o %t %s
# RUN: llvm-rtdyld -triple=aarch64_be-none-linux-gnu -verify -check=%s %t

.section .text.1,"ax"
.globl foo
foo:
  ret

.globl _main
_main:
  tbnz x0, #1, foo

## Branch 1 instruction back from _main
# rtdyld-check: *{4}(_main) = 0x370FFFE0
