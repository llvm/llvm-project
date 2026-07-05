# RUN: llvm-mc -filetype=obj -triple=i386 %s | llvm-objdump - --headers | FileCheck %s

  # CHECK: .text1        00000005 00000000
  # CHECK: .text2        00000005 00000000

  .section .text1
 .skip after-before,0x0
.Lint80_keep_stack:

  .section .text2
before:
 jmp .Lint80_keep_stack
after:
