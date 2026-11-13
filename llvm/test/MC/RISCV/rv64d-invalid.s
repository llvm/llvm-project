# RUN: llvm-mc --verify -triple riscv64 -mattr=+d < %s

# Integer registers where FP regs are expected
fcvt.l.d ft0, a0 # expected-error:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-1:10 {{register must be a GPR}}
# expected-note@-2:1 {{instruction requires the following: 'Zdinx' (Double in Integer)}}
fcvt.lu.d ft1, a1 # expected-error:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-1:11 {{register must be a GPR}}
# expected-note@-2:1 {{instruction requires the following: 'Zdinx' (Double in Integer)}}
fmv.x.d ft2, a2 # expected-error:9 {{register must be a GPR}}

# FP registers where integer regs are expected
fcvt.d.l a3, ft3 # expected-error:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-1:10 {{register must be a FPR}}
# expected-note@-2:1 {{instruction requires the following: 'Zdinx' (Double in Integer)}}

fcvt.d.lu a4, ft4 # expected-error:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-1:11 {{register must be a FPR}}
# expected-note@-2:1 {{instruction requires the following: 'Zdinx' (Double in Integer)}}

fmv.d.x a5, ft5 # expected-error:9 {{register must be a FPR}}
