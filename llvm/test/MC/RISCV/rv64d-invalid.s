# RUN: not llvm-mc -triple riscv64 -mattr=+d < %s 2>&1 | FileCheck --match-full-lines %s

# Integer registers where FP regs are expected
fcvt.l.d ft0, a0 # CHECK: <stdin>:[[@LINE]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: <stdin>:[[@LINE-1]]:10: note: register must be a GPR
# CHECK: <stdin>:[[@LINE-2]]:1: note: instruction requires the following: 'Zdinx' (Double in Integer)
fcvt.lu.d ft1, a1 # CHECK: <stdin>:[[@LINE]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: <stdin>:[[@LINE-1]]:11: note: register must be a GPR
# CHECK: <stdin>:[[@LINE-2]]:1: note: instruction requires the following: 'Zdinx' (Double in Integer)
fmv.x.d ft2, a2 # CHECK: <stdin>:[[@LINE]]:9: error: register must be a GPR

# FP registers where integer regs are expected
fcvt.d.l a3, ft3 # CHECK: <stdin>:[[@LINE]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: <stdin>:[[@LINE-1]]:10: note: register must be a FPR
# CHECK: <stdin>:[[@LINE-2]]:1: note: instruction requires the following: 'Zdinx' (Double in Integer)

fcvt.d.lu a4, ft4 # CHECK: <stdin>:[[@LINE]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: <stdin>:[[@LINE-1]]:11: note: register must be a FPR
# CHECK: <stdin>:[[@LINE-2]]:1: note: instruction requires the following: 'Zdinx' (Double in Integer)

fmv.d.x a5, ft5 # CHECK: <stdin>:[[@LINE]]:9: error: register must be a FPR
