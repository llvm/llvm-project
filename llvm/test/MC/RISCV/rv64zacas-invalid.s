# RUN: not llvm-mc -triple riscv64 -mattr=+experimental-zacas < %s 2>&1 | FileCheck %s

# Non-zero offsets not supported for the third operand (rs1).
amocas.w a1, a3, 1(a5) # CHECK: :[[@LINE]]:18: error: optional integer offset must be 0
amocas.d a1, a3, 2(a5) # CHECK: :[[@LINE]]:18: error: optional integer offset must be 0
amocas.q a1, a3, 3(a5) # CHECK: :[[@LINE]]:18: error: optional integer offset must be 0

# First and second operands (rd and rs2) of amocas.q must be even.
amocas.q a1, a2, (a1) # CHECK: :[[@LINE]]:10: error: The destination register must be even.
amocas.q a0, a1, (a1) # CHECK: :[[@LINE]]:14: error: The source register must be even.
amocas.q.aq a1, a2, (a1) # CHECK: :[[@LINE]]:13: error: The destination register must be even.
amocas.q.aq a0, a1, (a1) # CHECK: :[[@LINE]]:17: error: The source register must be even.
amocas.q.rl a1, a2, (a1) # CHECK: :[[@LINE]]:13: error: The destination register must be even.
amocas.q.rl a0, a1, (a1) # CHECK: :[[@LINE]]:17: error: The source register must be even.
amocas.q.aqrl a1, a2, (a1) # CHECK: :[[@LINE]]:15: error: The destination register must be even.
amocas.q.aqrl a0, a1, (a1) # CHECK: :[[@LINE]]:19: error: The source register must be even.
