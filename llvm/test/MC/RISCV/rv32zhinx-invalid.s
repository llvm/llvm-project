# RUN: not llvm-mc -triple riscv32 -mattr=+zhinx %s 2>&1 | FileCheck %s

# Not support float registers
flw fa4, 12(sp) # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'F' (Single-Precision Floating-Point){{$}}
fadd.h fa0, fa1, fa2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'Zfh' (Half-Precision Floating-Point){{$}}

# Invalid instructions
fsw a5, 12(sp) # CHECK: :[[@LINE]]:5: error: invalid operand for instruction
fmv.x.h s0, s1 # CHECK: :[[@LINE]]:13: error: invalid operand for instruction

# Invalid register names
fadd.h a100, a2, a3 # CHECK: :[[@LINE]]:8: error: invalid operand for instruction
fsgnjn.h a100, a2, a3 # CHECK: :[[@LINE]]:10: error: invalid operand for instruction

# Rounding mode when a register is expected
fmadd.h x10, x11, x12, ree # CHECK: :[[@LINE]]:24: error: invalid operand for instruction

# Invalid rounding modes
fmadd.h x10, x11, x12, x13, ree # CHECK: :[[@LINE]]:29: error: operand must be a valid floating point rounding mode mnemonic
fmsub.h x14, x15, x16, x17, 0 # CHECK: :[[@LINE]]:29: error: operand must be a valid floating point rounding mode mnemonic
fnmsub.h x18, x19, x20, x21, 0b111 # CHECK: :[[@LINE]]:30: error: operand must be a valid floating point rounding mode mnemonic

# FP registers where integer regs are expected
fcvt.wu.h ft2, a1 # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
