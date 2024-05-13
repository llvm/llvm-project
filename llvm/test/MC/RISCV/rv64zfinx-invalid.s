# RUN: not llvm-mc -triple riscv64 -mattr=+zfinx %s 2>&1 | FileCheck %s

# Not support float registers
flw fa4, 12(sp) # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'F' (Single-Precision Floating-Point){{$}}

# Invalid instructions
fsw a5, 12(sp) # CHECK: :[[@LINE]]:5: error: invalid operand for instruction
fmv.x.w t2, a2 # CHECK: :[[@LINE]]:13: error: invalid operand for instruction
fmv.w.x a5, t5 # CHECK: :[[@LINE]]:9: error: invalid operand for instruction

# FP registers where integer regs are expected
fcvt.s.l a2, ft2 # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
fcvt.s.lu a3, ft3 # CHECK: :[[@LINE]]:15: error: invalid operand for instruction
