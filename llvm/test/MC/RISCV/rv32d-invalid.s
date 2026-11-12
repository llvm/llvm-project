# RUN: not llvm-mc -triple riscv32 -mattr=+d < %s 2>&1 | FileCheck --match-full-lines %s

# Out of range immediates
## simm12
fld ft1, -2049(a0) # CHECK: <stdin>:[[@LINE]]:10: error: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]
fsd ft2, 2048(a1) # CHECK: <stdin>:[[@LINE]]:10: error: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]

# Memory operand not formatted correctly
fld ft1, a0, -200 # CHECK: <stdin>:[[@LINE]]:14: error: register must be a GPR excluding zero (x0)
fsd ft2, a1, 100 # CHECK: <stdin>:[[@LINE]]:14: error: register must be a GPR excluding zero (x0)
fld ft1, a_symbol, x0 # CHECK: <stdin>:[[@LINE]]:20: error: register must be a GPR excluding zero (x0)
fsd ft2, a_symbol, x0 # CHECK: <stdin>:[[@LINE]]:20: error: register must be a GPR excluding zero (x0)

# Invalid register names
fld ft15, 100(a0) # CHECK: <stdin>:[[@LINE]]:5: error: register must be a FPR
fld ft1, 100(a10) # CHECK: <stdin>:[[@LINE]]:14: error: expected register
fsgnjn.d fa100, fa2, fa3 # CHECK: <stdin>:[[@LINE]]:10: error: register must be a FPR

# Integer registers where FP regs are expected
fadd.d a2, a1, a0 
# CHECK: <stdin>:[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: <stdin>:[[@LINE-2]]:8: note: register must be a FPR
# CHECK: <stdin>:[[@LINE-3]]:1: note: instruction requires the following: 'Zdinx' (Double in Integer), RV64I Base Instruction Set
# CHECK: <stdin>:[[@LINE-4]]:1: note: instruction requires the following: 'Zdinx' (Double in Integer)

# FP registers where integer regs are expected
fcvt.wu.d ft2, a1 
# CHECK: <stdin>:[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: <stdin>:[[@LINE-2]]:11: note: register must be a GPR
# CHECK: <stdin>:[[@LINE-3]]:1: note: instruction requires the following: 'Zdinx' (Double in Integer), RV64I Base Instruction Set
# CHECK: <stdin>:[[@LINE-4]]:1: note: instruction requires the following: 'Zdinx' (Double in Integer)
