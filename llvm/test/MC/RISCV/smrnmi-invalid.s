# RUN: not llvm-mc -triple riscv32 -mattr=+smrnmi < %s 2>&1 | FileCheck --match-full-lines %s
# RUN: not llvm-mc -triple riscv64 -mattr=+smrnmi < %s 2>&1 | FileCheck --match-full-lines %s

mnret 0x10 # CHECK: <stdin>:[[@LINE]]:7: error: unexpected extra operand for instruction
