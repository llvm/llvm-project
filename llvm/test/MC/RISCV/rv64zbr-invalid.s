# RUN: not llvm-mc -triple riscv64 -mattr=+experimental-zbr < %s 2>&1 | FileCheck %s

# Too many operands
crc32.d t0, t1, t2 # CHECK: :[[@LINE]]:17: error: invalid operand for instruction
# Too many operands
crc32c.d t0, t1, t2 # CHECK: :[[@LINE]]:18: error: invalid operand for instruction
