# RUN: not llvm-mc -triple riscv64 -mattr=+zfh < %s 2>&1 | \
# RUN:   FileCheck %s

# Out of range immediates
## simm12
flh ft1, -2049(a0) # CHECK: :[[@LINE]]:10: error: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo modifier or an integer in the range [-2048, 2047]
fsh ft2, 2048(a1) # CHECK: :[[@LINE]]:10: error: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo modifier or an integer in the range [-2048, 2047]

# Memory operand not formatted correctly
flw ft1, a0, -200 # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
fsw ft2, a1, 100 # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
fld ft1, a0, -200 # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
fsd ft2, a1, 100 # CHECK: :[[@LINE]]:14: error: invalid operand for instruction

# Integer registers where FP regs are expected
fcvt.l.h ft0, a0 # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
fcvt.lu.h ft1, a1 # CHECK: :[[@LINE]]:11: error: invalid operand for instruction

# FP registers where integer regs are expected
fcvt.h.l a2, ft2 # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
fcvt.h.lu a3, ft3 # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
