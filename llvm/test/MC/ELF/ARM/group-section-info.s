# RUN: llvm-mc %s -triple armv7-elf -filetype obj -o %t1.1.o
# RUN: llvm-readelf -Ss %t1.1.o | FileCheck %s
# This test checks the value of sh_info group section when the group signature
# is the same as a section name.

# CHECK: Section Headers
# CHECK: foo
# CHECK: .group {{.*}} 04 6 1
# CHECK: A {{.*}} AXG

# CHECK: Symbol table
# CHECK: 1: {{.*}} foo

  .section foo,"ax",%progbits
  .globl main
main:
  .section A,"axG",%progbits,foo
