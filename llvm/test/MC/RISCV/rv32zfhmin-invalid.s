# RUN: not llvm-mc -triple riscv32 -mattr=+experimental-zfhmin < %s 2>&1 | \
# RUN:   FileCheck %s

# Out of range immediates
## simm12
flh ft1, -2049(a0) # CHECK: :[[@LINE]]:10: error: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo modifier or an integer in the range [-2048, 2047]
fsh ft2, 2048(a1) # CHECK: :[[@LINE]]:10: error: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo modifier or an integer in the range [-2048, 2047]

# Memory operand not formatted correctly
flh ft1, a0, -200 # CHECK: :[[@LINE]]:14: error: invalid operand for instruction

# Invalid register names
flh ft15, 100(a0) # CHECK: :[[@LINE]]:5: error: invalid operand for instruction
flh ft1, 100(a10) # CHECK: :[[@LINE]]:14: error: expected register

# Integer registers where FP regs are expected
fmv.x.h fs7, a2 # CHECK: :[[@LINE]]:9: error: invalid operand for instruction

# FP registers where integer regs are expected
fmv.h.x a8, ft2 # CHECK: :[[@LINE]]:9: error: invalid operand for instruction

# Zfh instructions
fmadd.h f10, f11, f12, f13, dyn # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'Zfh' (Half-Precision Floating-Point)
