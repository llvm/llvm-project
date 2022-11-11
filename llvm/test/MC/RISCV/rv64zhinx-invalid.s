# RUN: not llvm-mc -triple riscv64 -mattr=+zhinx %s 2>&1 | FileCheck %s

# Not support float registers
flh fa4, 12(sp) # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'Zfh' (Half-Precision Floating-Point) or 'Zfhmin' (Half-Precision Floating-Point Minimal){{$}}

# Invalid instructions
fsh a5, 12(sp) # CHECK: :[[@LINE]]:5: error: invalid operand for instruction
fmv.x.h t2, a2 # CHECK: :[[@LINE]]:13: error: invalid operand for instruction
fmv.h.x a5, t5 # CHECK: :[[@LINE]]:9: error: invalid operand for instruction

# FP registers where integer regs are expected
fcvt.h.l a2, ft2 # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
fcvt.h.lu a3, ft3 # CHECK: :[[@LINE]]:15: error: invalid operand for instruction
