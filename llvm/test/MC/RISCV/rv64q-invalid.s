# RUN: not llvm-mc -triple riscv64 -mattr=+q < %s 2>&1 | FileCheck %s

# Integer registers where FP regs are expected
fcvt.l.q ft0, a0 # CHECK: :[[@LINE]]:1: error: invalid instruction
fcvt.lu.q ft1, a1 # CHECK: :[[@LINE]]:1: error: invalid instruction

# FP registers where integer regs are expected
fcvt.q.l a3, ft3 # CHECK: :[[@LINE]]:1: error: invalid instruction
fcvt.q.lu a4, ft4 # CHECK: :[[@LINE]]:1: error: invalid instruction
