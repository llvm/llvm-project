// RUN: llvm-mc -triple i686-unknown-windows-msvc -filetype obj -o %t.o %s
// RUN: llvm-objdump -r %t.o | FileCheck %s

// CHECK: 00000004 IMAGE_REL_I386_REL32 twop32

  .section .rdata,"rd"
twop32:
  .quad 0x41f0000000000000

  .text
0:
  mulsd twop32-0b(%eax), %xmm1
