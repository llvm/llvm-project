# RUN: llvm-mc --verify -triple riscv64 -mattr=+f < %s

# Integer registers where FP regs are expected
fcvt.l.s ft0, a0 # expected-error:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-1:10 {{register must be a GPR}}
# expected-note@-2:1 {{instruction requires the following: 'Zfinx' (Float in Integer)}}
fcvt.lu.s ft1, a1 # expected-error:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-1:11 {{register must be a GPR}}
# expected-note@-2:1 {{instruction requires the following: 'Zfinx' (Float in Integer)}}

# FP registers where integer regs are expected
fcvt.s.l a2, ft2 # expected-error:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-1:10 {{register must be a FPR}}
# expected-note@-2:1 {{instruction requires the following: 'Zfinx' (Float in Integer)}}

fcvt.s.lu a3, ft3 # expected-error:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-1:11 {{register must be a FPR}}
# expected-note@-2:1 {{instruction requires the following: 'Zfinx' (Float in Integer)}}
