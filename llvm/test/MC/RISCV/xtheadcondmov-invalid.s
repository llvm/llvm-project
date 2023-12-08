# RUN: not llvm-mc -triple riscv32 -mattr=+xtheadcondmov < %s 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple riscv64 -mattr=+xtheadcondmov < %s 2>&1 | FileCheck %s

th.mveqz a0,a1        # CHECK: :[[@LINE]]:1: error: too few operands for instruction
th.mveqz a0,a1,a2,a3  # CHECK: :[[@LINE]]:19: error: invalid operand for instruction
th.mveqz a0,a1,1      # CHECK: :[[@LINE]]:16: error: invalid operand for instruction
th.mvnez a0,a1        # CHECK: :[[@LINE]]:1: error: too few operands for instruction
th.mvnez a0,a1,a2,a3  # CHECK: :[[@LINE]]:19: error: invalid operand for instruction
th.mvnez a0,a1,1      # CHECK: :[[@LINE]]:16: error: invalid operand for instruction
