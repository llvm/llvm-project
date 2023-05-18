# RUN: not llvm-mc -triple=riscv32 -mattr=+c,+f < %s 2>&1 | FileCheck %s

## FPRC
c.flw  ft3, 8(a5) # CHECK: :[[@LINE]]:8: error: invalid operand for instruction

## uimm8_lsb00
c.flwsp  fs1, 256(sp) # CHECK: :[[@LINE]]:15: error: immediate must be a multiple of 4 bytes in the range [0, 252]
c.fswsp  fs2, -4(sp) # CHECK: :[[@LINE]]:15: error: immediate must be a multiple of 4 bytes in the range [0, 252]

## uimm7_lsb00
c.flw  fs0, -4(sp) # CHECK: :[[@LINE]]:13: error: immediate must be a multiple of 4 bytes in the range [0, 124]
c.fsw  fs1, 128(sp) # CHECK: :[[@LINE]]:13: error: immediate must be a multiple of 4 bytes in the range [0, 124]
