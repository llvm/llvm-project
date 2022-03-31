# RUN: not llvm-mc -triple riscv64 -mattr=+zbb < %s 2>&1 | FileCheck %s

# Too many operands
clzw t0, t1, t2 # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
# Too many operands
ctzw t0, t1, t2 # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
# Too many operands
cpopw t0, t1, t2 # CHECK: :[[@LINE]]:15: error: invalid operand for instruction
