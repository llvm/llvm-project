# RUN: not llvm-mc -triple riscv32 -mattr=+d < %s 2>&1 | FileCheck %s

# Out of range immediates
## simm12
fld ft1, -2049(a0) # CHECK: :[[@LINE]]:10: error: operand must be a symbol with %lo/%pcrel_lo modifier or an integer in the range [-2048, 2047]
fsd ft2, 2048(a1) # CHECK: :[[@LINE]]:10: error: operand must be a symbol with %lo/%pcrel_lo modifier or an integer in the range [-2048, 2047]

# Memory operand not formatted correctly
fld ft1, a0, -200 # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
fsd ft2, a1, 100 # CHECK: :[[@LINE]]:14: error: invalid operand for instruction

# Invalid register names
fld ft15, 100(a0) # CHECK: :[[@LINE]]:5: error: invalid operand for instruction
fld ft1, 100(a10) # CHECK: :[[@LINE]]:14: error: expected register
fsgnjn.d fa100, fa2, fa3 # CHECK: :[[@LINE]]:10: error: invalid operand for instruction

# Integer registers where FP regs are expected
fadd.d a2, a1, a0 # CHECK: :[[@LINE]]:8: error: invalid operand for instruction

# FP registers where integer regs are expected
fcvt.wu.d ft2, a1 # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
