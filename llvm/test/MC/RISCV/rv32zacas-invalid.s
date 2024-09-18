# RUN: not llvm-mc -triple riscv32 -mattr=+a,+experimental-zacas < %s 2>&1 | FileCheck %s

# Non-zero offsets not supported for the third operand (rs1).
amocas.w a1, a3, 1(a5) # CHECK: :[[@LINE]]:18: error: optional integer offset must be 0
amocas.d a0, a2, 2(a5) # CHECK: :[[@LINE]]:18: error: optional integer offset must be 0

# First and second operands (rd and rs2) of amocas.d must be even for RV32.
amocas.d a1, a2, (a1) # CHECK: :[[@LINE]]:10: error: register must be even
amocas.d a0, a1, (a1) # CHECK: :[[@LINE]]:14: error: register must be even
amocas.d.aq a1, a2, (a1) # CHECK: :[[@LINE]]:13: error: register must be even
amocas.d.aq a0, a1, (a1) # CHECK: :[[@LINE]]:17: error: register must be even
amocas.d.rl a1, a2, (a1) # CHECK: :[[@LINE]]:13: error: register must be even
amocas.d.rl a0, a1, (a1) # CHECK: :[[@LINE]]:17: error: register must be even
amocas.d.aqrl a1, a2, (a1) # CHECK: :[[@LINE]]:15: error: register must be even
amocas.d.aqrl a0, a1, (a1) # CHECK: :[[@LINE]]:19: error: register must be even

# amocas.q is not supported for RV32.
amocas.q a0, a0, (a1) # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set{{$}}
