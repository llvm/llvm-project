# RUN: llvm-mc --verify -triple riscv32 -mattr=+zhinxmin %s

# Not support float registers
flw fa4, 12(sp) # expected-error:1 {{instruction requires the following: 'F' (Single-Precision Floating-Point)}}
fcvt.h.s fa0, fa1 # expected-error:1 {{instruction requires the following: 'Zfh' (Half-Precision Floating-Point) or 'Zfhmin' (Half-Precision Floating-Point Minimal)}}

# Invalid instructions
fsw a5, 12(sp) # expected-error:1 {{instruction requires the following: 'F' (Single-Precision Floating-Point)}}
fmv.x.h s0, s1 # expected-error:1 {{invalid instruction}}

# Invalid register names
fcvt.h.s a100, a1 # expected-error:10 {{register must be a GPR when used as an FP operand}}

# Valid in Zhinx
fmadd.h x10, x11, x12, x13, dyn # expected-error:1 {{instruction requires the following: 'Zhinx' (Half Float in Integer)}}
