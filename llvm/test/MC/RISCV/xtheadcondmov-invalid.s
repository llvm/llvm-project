# RUN: not llvm-mc -triple riscv32 -mattr=+xtheadcondmov < %s 2>&1 | FileCheck --match-full-lines %s
# RUN: not llvm-mc -triple riscv64 -mattr=+xtheadcondmov < %s 2>&1 | FileCheck --match-full-lines %s

th.mveqz a0,a1        # CHECK: <stdin>:[[@LINE]]:15: error: too few operands for instruction
th.mveqz a0,a1,a2,a3  # CHECK: <stdin>:[[@LINE]]:19: error: unexpected extra operand for instruction
th.mveqz a0,a1,1      # CHECK: <stdin>:[[@LINE]]:16: error: register must be a GPR
th.mvnez a0,a1        # CHECK: <stdin>:[[@LINE]]:15: error: too few operands for instruction
th.mvnez a0,a1,a2,a3  # CHECK: <stdin>:[[@LINE]]:19: error: unexpected extra operand for instruction
th.mvnez a0,a1,1      # CHECK: <stdin>:[[@LINE]]:16: error: register must be a GPR
