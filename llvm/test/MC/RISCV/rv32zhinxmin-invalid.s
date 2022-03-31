# RUN: not llvm-mc -triple riscv32 -mattr=+zhinxmin %s 2>&1 | FileCheck %s

# Not support float registers
flw fa4, 12(sp) # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'F' (Single-Precision Floating-Point)
fcvt.h.s fa0, fa1 # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'Zfh' (Half-Precision Floating-Point) or 'Zfhmin' (Half-Precision Floating-Point Minimal)

# Invalid instructions
fsw a5, 12(sp) # CHECK: :[[@LINE]]:5: error: invalid operand for instruction
fmv.x.h s0, s1 # CHECK: :[[@LINE]]:13: error: invalid operand for instruction

# Invalid register names
fcvt.h.s a100, a1 # CHECK: :[[@LINE]]:10: error: invalid operand for instruction

# Valid in Zhinx
fmadd.h x10, x11, x12, x13, dyn # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'Zhinx' (Half Float in Integer)
