# RUN: not llvm-mc -triple riscv32 -mattr=+smrnmi < %s 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple riscv64 -mattr=+smrnmi < %s 2>&1 | FileCheck %s

mnret 0x10 # CHECK: :[[@LINE]]:7: error: invalid operand for instruction
