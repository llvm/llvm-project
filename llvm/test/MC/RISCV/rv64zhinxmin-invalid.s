# RUN: llvm-mc --verify -triple riscv64 -mattr=+zhinxmin %s

# Not support float registers
flh fa4, 12(sp) # expected-error:1 {{instruction requires the following: 'Zfh' (Half-Precision Floating-Point) or 'Zfhmin' (Half-Precision Floating-Point Minimal) or 'Zfbfmin' (Scalar BF16 Converts)}}

# Invalid instructions
fsh a5, 12(sp) # expected-error:1 {{instruction requires the following: 'Zfh' (Half-Precision Floating-Point) or 'Zfhmin' (Half-Precision Floating-Point Minimal) or 'Zfbfmin' (Scalar BF16 Converts)}}
fmv.x.h t2, a2 # expected-error:1 {{invalid instruction}}
fmv.h.x a5, t5 # expected-error:1 {{invalid instruction}}

# FP registers where integer regs are expected
fcvt.d.h a0, fa2 # expected-error:1 {{instruction requires the following: 'Zdinx' (Double in Integer), RV32I Base Instruction Set}}
fcvt.h.d a0, fa2 # expected-error:1 {{invalid instruction}}
