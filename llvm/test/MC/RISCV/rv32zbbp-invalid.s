# RUN: not llvm-mc -triple riscv32 -mattr=+experimental-b,experimental-zbb,experimental-zbp < %s 2>&1 | FileCheck %s

# Too few operands
andn t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
orn t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
xnor t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
rol t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
ror t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
rori t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Immediate operand out of range
rori t0, t1, 32 # CHECK: :[[@LINE]]:14: error: immediate must be an integer in the range [0, 31]
rori t0, t1, -1 # CHECK: :[[@LINE]]:14: error: immediate must be an integer in the range [0, 31]
# Too few operands
pack t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
packu t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
packh t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
