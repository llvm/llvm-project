# RUN: not llvm-mc -triple riscv32 -mattr=+c,+experimental-zbproposedc < %s 2>&1 | FileCheck %s

# Too many operands
c.not s0, s1 # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
c.neg s0, s1 # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
