# RUN: llvm-mc -triple=mipsel -mcpu=mips32r2 -mattr=+micromips -filetype=obj %s -o %t.le.o
# RUN: llvm-objdump -s -j .text.micromips %t.le.o | FileCheck %s --check-prefix=LE
# RUN: llvm-mc -triple=mips -mcpu=mips32r2 -mattr=+micromips -filetype=obj %s -o %t.be.o
# RUN: llvm-objdump -s -j .text.micromips %t.be.o | FileCheck %s --check-prefix=BE
# RUN: llvm-mc -filetype=obj  -triple=mipsel %s -o %t
.byte 1
.p2align 2
foo:

.section .text.micromips,"ax",@progbits
.set micromips
.set noreorder
  move $2, $3
  .p2align 2
  move $2, $3
  .p2align 3
  move $2, $3
  .p2align 3

## Emit one 16-bit NOP (0x0c00), then fill the remaining words with zero NOPs.
# LE: 0000 430c000c 430c000c 430c000c 00000000
# BE: 0000 0c430c00 0c430c00 0c430c00 00000000

## An odd-sized data item needs a zero byte before the NOPs.
  .byte 0xff
  .p2align 3

## Standard MIPS padding remains zero; .set pop restores microMIPS padding.
.set push
.set nomicromips
  nop
  .p2align 3
.set pop
  move $2, $3
  .p2align 3

# LE: 0010 ff00000c 00000000 00000000 00000000
# BE: 0010 ff000c00 00000000 00000000 00000000
# LE: 0020 430c000c 00000000
# BE: 0020 0c430c00 00000000
