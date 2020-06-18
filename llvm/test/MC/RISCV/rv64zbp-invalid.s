# RUN: not llvm-mc -triple riscv64 -mattr=+experimental-b,experimental-zbp < %s 2>&1 | FileCheck %s

# Too few operands
gorcw t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
grevw t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
gorciw t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Immediate operand out of range
gorciw t0, t1, 32 # CHECK: :[[@LINE]]:16: error: immediate must be an integer in the range [0, 31]
gorciw t0, t1, -1 # CHECK: :[[@LINE]]:16: error: immediate must be an integer in the range [0, 31]
# Too few operands
greviw t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Immediate operand out of range
greviw t0, t1, 32 # CHECK: :[[@LINE]]:16: error: immediate must be an integer in the range [0, 31]
greviw t0, t1, -1 # CHECK: :[[@LINE]]:16: error: immediate must be an integer in the range [0, 31]
# Too few operands
shflw t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
unshflw t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
