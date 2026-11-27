# RUN: llvm-mc --verify -triple riscv64 -mattr=+zfh < %s

# Out of range immediates
## simm12
flh ft1, -2049(a0) # expected-error:10 {{operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]}}
fsh ft2, 2048(a1) # expected-error:10 {{operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]}}

# Integer registers where FP regs are expected
fcvt.l.h ft0, a0 
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:10 {{register must be a GPR}}
# expected-note@-3:1 {{instruction requires the following: 'Zhinx' (Half Float in Integer)}}

fcvt.lu.h ft1, a1 
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:11 {{register must be a GPR}}
# expected-note@-3:1 {{instruction requires the following: 'Zhinx' (Half Float in Integer)}}

# FP registers where integer regs are expected
fcvt.h.l a2, ft2 
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:10 {{register must be a FPR}}
# expected-note@-3:1 {{instruction requires the following: 'Zhinx' (Half Float in Integer)}}

fcvt.h.lu a3, ft3 
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:11 {{register must be a FPR}}
# expected-note@-3:1 {{instruction requires the following: 'Zhinx' (Half Float in Integer)}}
