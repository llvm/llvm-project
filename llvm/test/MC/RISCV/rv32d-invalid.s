# RUN: llvm-mc --verify -triple riscv32 -mattr=+d < %s

# Out of range immediates
## simm12
fld ft1, -2049(a0) # expected-error:10 {{operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]}}
fsd ft2, 2048(a1) # expected-error:10 {{operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]}}

# Memory operand not formatted correctly
fld ft1, a0, -200 # expected-error:14 {{register must be a GPR excluding zero (x0)}}
fsd ft2, a1, 100 # expected-error:14 {{register must be a GPR excluding zero (x0)}}
fld ft1, a_symbol, x0 # expected-error:20 {{register must be a GPR excluding zero (x0)}}
fsd ft2, a_symbol, x0 # expected-error:20 {{register must be a GPR excluding zero (x0)}}

# Invalid register names
fld ft15, 100(a0) # expected-error:5 {{register must be a FPR}}
fld ft1, 100(a10) # expected-error:14 {{expected register}}
fsgnjn.d fa100, fa2, fa3 # expected-error:10 {{register must be a FPR}}

# Integer registers where FP regs are expected
fadd.d a2, a1, a0 
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:8 {{register must be a FPR}}
# expected-note@-3:1 {{instruction requires the following: 'Zdinx' (Double in Integer), RV64I Base Instruction Set}}
# expected-note@-4:1 {{instruction requires the following: 'Zdinx' (Double in Integer)}}

# FP registers where integer regs are expected
fcvt.wu.d ft2, a1 
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:11 {{register must be a GPR}}
# expected-note@-3:1 {{instruction requires the following: 'Zdinx' (Double in Integer), RV64I Base Instruction Set}}
# expected-note@-4:1 {{instruction requires the following: 'Zdinx' (Double in Integer)}}
