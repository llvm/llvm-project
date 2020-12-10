# RUN: not llvm-mc -triple riscv32 -mattr=+experimental-zfh < %s 2>&1 | \
# RUN:   FileCheck %s

# Out of range immediates
## simm12
flh ft1, -2049(a0) # CHECK: :[[@LINE]]:10: error: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo modifier or an integer in the range [-2048, 2047]
fsh ft2, 2048(a1) # CHECK: :[[@LINE]]:10: error: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo modifier or an integer in the range [-2048, 2047]

# Memory operand not formatted correctly
flh ft1, a0, -200 # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
fsw ft2, a1, 100 # CHECK: :[[@LINE]]:14: error: invalid operand for instruction

# Invalid register names
flh ft15, 100(a0) # CHECK: :[[@LINE]]:5: error: invalid operand for instruction
flh ft1, 100(a10) # CHECK: :[[@LINE]]:14: error: expected register
fsgnjn.h fa100, fa2, fa3 # CHECK: :[[@LINE]]:10: error: invalid operand for instruction

# Integer registers where FP regs are expected
fmv.x.h fs7, a2 # CHECK: :[[@LINE]]:9: error: invalid operand for instruction

# FP registers where integer regs are expected
fmv.h.x a8, ft2 # CHECK: :[[@LINE]]:9: error: invalid operand for instruction

# Rounding mode when a register is expected
fmadd.h f10, f11, f12, ree # CHECK: :[[@LINE]]:24: error: invalid operand for instruction

# Invalid rounding modes
fmadd.h f10, f11, f12, f13, ree # CHECK: :[[@LINE]]:29: error: operand must be a valid floating point rounding mode mnemonic
fmsub.h f14, f15, f16, f17, 0 # CHECK: :[[@LINE]]:29: error: operand must be a valid floating point rounding mode mnemonic
fnmsub.h f18, f19, f20, f21, 0b111 # CHECK: :[[@LINE]]:30: error: operand must be a valid floating point rounding mode mnemonic

# Integer registers where FP regs are expected
fadd.h a2, a1, a0 # CHECK: :[[@LINE]]:8: error: invalid operand for instruction

# FP registers where integer regs are expected
fcvt.wu.h ft2, a1 # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
