# RUN: not llvm-mc -triple riscv32 -mattr=+experimental-zbr < %s 2>&1 | FileCheck %s

# Too many operands
crc32.b	t0, t1, t2 # CHECK: :[[@LINE]]:17: error: invalid operand for instruction
# Too many operands
crc32.h	t0, t1, t2 # CHECK: :[[@LINE]]:17: error: invalid operand for instruction
# Too many operands
crc32.w	t0, t1, t2 # CHECK: :[[@LINE]]:17: error: invalid operand for instruction
# Too many operands
crc32c.b t0, t1, t2 # CHECK: :[[@LINE]]:18: error: invalid operand for instruction
# Too many operands
crc32c.h t0, t1, t2 # CHECK: :[[@LINE]]:18: error: invalid operand for instruction
# Too many operands
crc32c.w t0, t1, t2 # CHECK: :[[@LINE]]:18: error: invalid operand for instruction
crc32.d t0, t1 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set{{$}}
crc32c.d t0, t1 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set{{$}}
