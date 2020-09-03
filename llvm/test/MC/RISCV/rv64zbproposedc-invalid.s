# RUN: not llvm-mc -triple riscv64 -mattr=+c,+experimental-zbproposedc,+experimental-b < %s 2>&1 | FileCheck %s

# Too many operands
c.zext.w s0, s1 # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
