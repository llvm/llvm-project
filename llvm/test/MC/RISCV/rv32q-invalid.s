# RUN: not llvm-mc -triple riscv32 -mattr=+q < %s 2>&1 | FileCheck %s

# Out of range immediates
## simm12
flq ft1, -2049(a0) # CHECK: :[[@LINE]]:10: error: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]
fsq ft2, 2048(a1) # CHECK: :[[@LINE]]:10: error: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]

# Memory operand not formatted correctly
flq ft1, a0, -200 # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
fsq ft2, a1, 100 # CHECK: :[[@LINE]]:14: error: invalid operand for instruction

# Invalid register names
flq ft15, 100(a0) # CHECK: :[[@LINE]]:5: error: invalid operand for instruction
flq ft1, 100(a10) # CHECK: :[[@LINE]]:14: error: expected register
fsgnjn.q fa100, fa2, fa3 # CHECK: :[[@LINE]]:10: error: invalid operand for instruction

# Integer registers where FP regs are expected
fadd.q a2, a1, a0 # CHECK: :[[@LINE]]:8: error: invalid operand for instruction

# FP registers where integer regs are expected
fcvt.wu.q ft2, a1 # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
