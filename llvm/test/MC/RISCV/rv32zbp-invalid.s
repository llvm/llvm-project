# RUN: not llvm-mc -triple riscv32 -mattr=+experimental-b,experimental-zbp < %s 2>&1 | FileCheck %s

# Too few operands
gorc t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
grev t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
gorci t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Immediate operand out of range
gorci t0, t1, 32 # CHECK: :[[@LINE]]:15: error: immediate must be an integer in the range [0, 31]
gorci t0, t1, -1 # CHECK: :[[@LINE]]:15: error: immediate must be an integer in the range [0, 31]
# Too few operands
grevi t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Immediate operand out of range
grevi t0, t1, 32 # CHECK: :[[@LINE]]:15: error: immediate must be an integer in the range [0, 31]
grevi t0, t1, -1 # CHECK: :[[@LINE]]:15: error: immediate must be an integer in the range [0, 31]
# Too few operands
shfl t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
unshfl t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
shfli t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Immediate operand out of range
shfli t0, t1, 16 # CHECK: :[[@LINE]]:15: error: immediate must be an integer in the range [0, 15]
shfli t0, t1, -1 # CHECK: :[[@LINE]]:15: error: immediate must be an integer in the range [0, 15]
# Too few operands
unshfli t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Immediate operand out of range
unshfli t0, t1, 16 # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 15]
unshfli t0, t1, -1 # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 15]
# Too few operands
pack t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
packu t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
packh t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
xperm.n t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
xperm.b t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
xperm.h t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
gorcw t0, t1, t2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
grevw t0, t1, t2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
gorciw t0, t1, 0 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
greviw t0, t1, 0 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
shflw t0, t1, t2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
unshflw t0, t1, t2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
packw t0, t1, t2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
packuw t0, t1, t2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
xperm.w t0, t1, t2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
