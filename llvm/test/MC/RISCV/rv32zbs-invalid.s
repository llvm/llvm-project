# RUN: not llvm-mc -triple riscv32 -mattr=+experimental-b,experimental-zbs < %s 2>&1 | FileCheck %s

# Too few operands
sbclr t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
sbset t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
sbinv t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
sbext t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
sbclri t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Immediate operand out of range
sbclri t0, t1, 32 # CHECK: :[[@LINE]]:16: error: immediate must be an integer in the range [0, 31]
sbclri t0, t1, -1 # CHECK: :[[@LINE]]:16: error: immediate must be an integer in the range [0, 31]
# Too few operands
sbseti t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Immediate operand out of range
sbseti t0, t1, 32 # CHECK: :[[@LINE]]:16: error: immediate must be an integer in the range [0, 31]
sbseti t0, t1, -1 # CHECK: :[[@LINE]]:16: error: immediate must be an integer in the range [0, 31]
# Too few operands
sbinvi t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Immediate operand out of range
sbinvi t0, t1, 32 # CHECK: :[[@LINE]]:16: error: immediate must be an integer in the range [0, 31]
sbinvi t0, t1, -1 # CHECK: :[[@LINE]]:16: error: immediate must be an integer in the range [0, 31]
# Too few operands
sbexti t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Immediate operand out of range
sbexti t0, t1, 32 # CHECK: :[[@LINE]]:16: error: immediate must be an integer in the range [0, 31]
sbexti t0, t1, -1 # CHECK: :[[@LINE]]:16: error: immediate must be an integer in the range [0, 31]
