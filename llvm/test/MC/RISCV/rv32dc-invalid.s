# RUN: not llvm-mc -triple=riscv32 -mattr=+c,+d < %s 2>&1 | FileCheck %s

## FPRC
c.fld  ft3, 8(a5) # CHECK: :[[@LINE]]:8: error: invalid operand for instruction

## uimm9_lsb000
c.fldsp  fs1, 512(sp) # CHECK: :[[@LINE]]:15: error: immediate must be a multiple of 8 bytes in the range [0, 504]
c.fsdsp  fs2, -8(sp) # CHECK: :[[@LINE]]:15: error: immediate must be a multiple of 8 bytes in the range [0, 504]

## uimm8_lsb000
c.fld  fs0, -8(sp) # CHECK: :[[@LINE]]:13: error: immediate must be a multiple of 8 bytes in the range [0, 248]
c.fsd  fs1, 256(sp) # CHECK: :[[@LINE]]:13: error: immediate must be a multiple of 8 bytes in the range [0, 248]
