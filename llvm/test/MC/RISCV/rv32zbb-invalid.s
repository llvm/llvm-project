# RUN: not llvm-mc -triple riscv32 -mattr=+experimental-b,experimental-zbb < %s 2>&1 | FileCheck %s

# Too few operands
slo t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
sro t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
sloi t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Immediate operand out of range
sloi t0, t1, 32 # CHECK: :[[@LINE]]:14: error: immediate must be an integer in the range [0, 31]
sloi t0, t1, -1 # CHECK: :[[@LINE]]:14: error: immediate must be an integer in the range [0, 31]
# Too few operands
sroi t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Immediate operand out of range
sroi t0, t1, 32 # CHECK: :[[@LINE]]:14: error: immediate must be an integer in the range [0, 31]
sroi t0, t1, -1 # CHECK: :[[@LINE]]:14: error: immediate must be an integer in the range [0, 31]
# Too many operands
clz t0, t1, t2 # CHECK: :[[@LINE]]:13: error: invalid operand for instruction
# Too many operands
ctz t0, t1, t2 # CHECK: :[[@LINE]]:13: error: invalid operand for instruction
# Too many operands
pcnt t0, t1, t2 # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
# Too many operands
sext.b t0, t1, t2 # CHECK: :[[@LINE]]:16: error: invalid operand for instruction
# Too many operands
sext.h t0, t1, t2 # CHECK: :[[@LINE]]:16: error: invalid operand for instruction
# Too few operands
min t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
max t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
minu t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
maxu t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
