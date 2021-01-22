# RUN: not llvm-mc -triple riscv32 -mattr=+c,+experimental-zbproposedc,+experimental-Zba < %s 2>&1 | FileCheck %s

# Too many operands
c.not s0, s1 # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
c.neg s0, s1 # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
c.zext.w s0 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
