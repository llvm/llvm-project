# RUN: not llvm-mc -triple riscv64 -mattr=+zdinx %s 2>&1 | FileCheck %s

# Not support float registers
fld fa4, 12(sp) # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'D' (Double-Precision Floating-Point)
ld a0, -2049(a1) # CHECK: :[[@LINE]]:8: error: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo modifier or an integer in the range [-2048, 2047]

# Invalid instructions
fsd a5, 12(sp) # CHECK: :[[@LINE]]:5: error: invalid operand for instruction
sd fa4, 64(sp) # CHECK: :[[@LINE]]:4: error: invalid operand for instruction
fmv.x.d t2, a2 # CHECK: :[[@LINE]]:13: error: invalid operand for instruction
fmv.d.x a5, t5 # CHECK: :[[@LINE]]:9:  error: invalid operand for instruction

# FP registers where integer regs are expected
fcvt.d.l a3, ft3 # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
fcvt.d.lu a4, ft4 # CHECK: :[[@LINE]]:15: error: invalid operand for instruction
