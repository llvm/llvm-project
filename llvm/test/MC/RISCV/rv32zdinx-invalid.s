# RUN: not llvm-mc -triple riscv32 -mattr=+zdinx %s 2>&1 | FileCheck %s

# Unsupport Odd Registers in RV32
fadd.d a0, a1, a2 # CHECK: :[[@LINE]]:1: error: double precision floating point operands must use even numbered X register

# Not support float registers
flw fa4, 12(sp) # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'F' (Single-Precision Floating-Point){{$}}
fadd.d fa0, fa1, fa2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'D' (Double-Precision Floating-Point){{$}}

# Invalid instructions
fsw a5, 12(sp) # CHECK: :[[@LINE]]:5: error: invalid operand for instruction
fmv.x.w s0, s1 # CHECK: :[[@LINE]]:13: error: invalid operand for instruction

# Invalid register names
fadd.d a100, a2, a3 # CHECK: :[[@LINE]]:8: error: invalid operand for instruction
fsgnjn.d a100, a2, a3 # CHECK: :[[@LINE]]:10: error: invalid operand for instruction

# Rounding mode when a register is expected
fmadd.d x10, x12, x14, ree # CHECK: :[[@LINE]]:24: error: invalid operand for instruction

# Invalid rounding modes
fmadd.d x10, x12, x14, x16, ree # CHECK: :[[@LINE]]:29: error: operand must be a valid floating point rounding mode mnemonic
fmsub.d x10, x12, x14, x16, 0 # CHECK: :[[@LINE]]:29: error: operand must be a valid floating point rounding mode mnemonic
fnmsub.d x10, x12, x14, x16, 0b111 # CHECK: :[[@LINE]]:30: error: operand must be a valid floating point rounding mode mnemonic

# FP registers where integer regs are expected
fcvt.wu.d ft2, a1 # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
