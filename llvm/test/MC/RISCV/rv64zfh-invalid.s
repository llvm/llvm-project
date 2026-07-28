# RUN: not llvm-mc -triple riscv64 -mattr=+zfh < %s 2>&1 | \
# RUN:   FileCheck %s

# Out of range immediates
## simm12
flh ft1, -2049(a0) # CHECK: :[[@LINE]]:10: error: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]
fsh ft2, 2048(a1) # CHECK: :[[@LINE]]:10: error: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]

# Integer registers where FP regs are expected
fcvt.l.h ft0, a0 # CHECK: :[[@LINE]]:10: error: register must be a GPR
fcvt.lu.h ft1, a1 # CHECK: :[[@LINE]]:11: error: register must be a GPR

# FP registers where integer regs are expected
fcvt.h.l a2, ft2 # CHECK: :[[@LINE]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-1]]:10: note: register must be a FPR
# CHECK: :[[@LINE-2]]:10: note: register must be a GPR when used as FPR
fcvt.h.lu a3, ft3 # CHECK: :[[@LINE]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-1]]:11: note: register must be a FPR
# CHECK: :[[@LINE-2]]:11: note: register must be a GPR when used as FPR
