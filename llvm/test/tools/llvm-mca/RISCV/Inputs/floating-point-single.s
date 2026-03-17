# Input instructions for the 'F' extension.

# Floating-Point Load and Store Instructions
## Single-Precision
flw ft0, 0(a0)
fsw ft0, 0(a0)

# Floating-Point Computational Instructions
## Single-Precision
fadd.s f26, f27, f28
fsub.s f29, f30, f31
fmul.s ft0, ft1, ft2
fdiv.s ft3, ft4, ft5
fsqrt.s ft6, ft7
fmin.s fa5, fa6, fa7
fmax.s fs2, fs3, fs4
fmadd.s f10, f11, f12, f31
fmsub.s f14, f15, f16, f17
fnmsub.s f18, f19, f20, f21
fnmadd.s f22, f23, f24, f25

# Floating-Point Conversion and Move Instructions
## Single-Precision
fcvt.w.s a0, fs5
fcvt.wu.s a1, fs6
fcvt.s.w ft11, a4
fcvt.s.wu ft0, a5

fcvt.l.s a0, ft0
fcvt.lu.s a1, ft1
fcvt.s.l ft2, a2
fcvt.s.lu ft3, a3

fmv.x.w a2, fs7
fmv.w.x ft1, a6

fsgnj.s fs1, fa0, fa1
fsgnjn.s fa1, fa3, fa4

# Floating-Point Compare Instructions
## Single-Precision
feq.s a1, fs8, fs9
flt.s a2, fs10, fs11
fle.s a3, ft8, ft9

# Floating-Point Classify Instruction
## Single-Precision
fclass.s a3, ft10
