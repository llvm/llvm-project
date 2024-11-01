# RUN: not llvm-mc -triple riscv32 -mattr=+experimental-zfbfmin,+d < %s 2>&1 | \
# RUN:   FileCheck %s
# RUN: not llvm-mc -triple riscv64 -mattr=+experimental-zfbfmin,+d < %s 2>&1 | \
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

# Attempting to use fcvt instructions from zfhmin
fcvt.s.h fa0, ft0 # CHECK: [[@LINE]]:1: error: instruction requires the following: 'Zfh' (Half-Precision Floating-Point) or 'Zfhmin' (Half-Precision Floating-Point Minimal)
fcvt.h.s ft2, fa2 # CHECK: [[@LINE]]:1: error: instruction requires the following: 'Zfh' (Half-Precision Floating-Point) or 'Zfhmin' (Half-Precision Floating-Point Minimal)
fcvt.d.h fa0, ft0 # CHECK: [[@LINE]]:1: error: instruction requires the following: 'Zfh' (Half-Precision Floating-Point) or 'Zfhmin' (Half-Precision Floating-Point Minimal)
fcvt.h.d ft2, fa2 # CHECK: [[@LINE]]:1: error: instruction requires the following: 'Zfh' (Half-Precision Floating-Point) or 'Zfhmin' (Half-Precision Floating-Point Minimal)
