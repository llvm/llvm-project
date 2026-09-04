# RUN: not llvm-mc -triple riscv64 -mattr=+f < %s 2>&1 | FileCheck %s

# Integer registers where FP regs are expected
fcvt.l.s ft0, a0 # CHECK: :[[@LINE]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-1]]:10: note: register must be a GPR
# CHECK: :[[@LINE-2]]:1: note: instruction requires the following: 'Zfinx' (Float in Integer)
fcvt.lu.s ft1, a1 # CHECK: :[[@LINE]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-1]]:11: note: register must be a GPR
# CHECK: :[[@LINE-2]]:1: note: instruction requires the following: 'Zfinx' (Float in Integer)

# FP registers where integer regs are expected
fcvt.s.l a2, ft2 # CHECK: :[[@LINE]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-1]]:10: note: register must be a FPR
# CHECK: :[[@LINE-2]]:1: note: instruction requires the following: 'Zfinx' (Float in Integer)

fcvt.s.lu a3, ft3 # CHECK: :[[@LINE]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-1]]:11: note: register must be a FPR
# CHECK: :[[@LINE-2]]:1: note: instruction requires the following: 'Zfinx' (Float in Integer)
