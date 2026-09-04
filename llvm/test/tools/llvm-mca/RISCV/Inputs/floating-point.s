# Input instructions for the 'F' and 'D' extensions.

# Floating-Point Load and Store Instructions
## Single-Precision
flw ft0, 0(a0)
fsw ft0, 0(a0)

## Double-Precision
fld ft0, 0(a0)
fsd ft0, 0(a0)

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

## Double-Precision
fadd.d f26, f27, f28
fsub.d f29, f30, f31
fmul.d ft0, ft1, ft2
fdiv.d ft3, ft4, ft5
fsqrt.d ft6, ft7
fmin.d fa5, fa6, fa7
fmax.d fs2, fs3, fs4
fmadd.d f10, f11, f12, f31
fmsub.d f14, f15, f16, f17
fnmsub.d f18, f19, f20, f21
fnmadd.d f22, f23, f24, f25

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

## Double-Precision
fcvt.wu.d a4, ft11
fcvt.w.d a4, ft11
fcvt.d.w ft0, a5
fcvt.d.wu ft1, a6

fcvt.s.d fs5, fs6
fcvt.d.s fs7, fs8

fcvt.l.d a0, ft0
fcvt.lu.d a1, ft1
fcvt.d.l ft3, a3
fcvt.d.lu ft4, a4

fmv.x.d a2, ft2
fmv.d.x ft5, a5

fsgnj.d fs1, fa0, fa1
fsgnjn.d fa1, fa3, fa4

# Floating-Point Compare Instructions
## Single-Precision
feq.s a1, fs8, fs9
flt.s a2, fs10, fs11
fle.s a3, ft8, ft9

## Double-Precision
feq.d a1, fs8, fs9
flt.d a2, fs10, fs11
fle.d a3, ft8, ft9

# Floating-Point Classify Instruction
## Single-Precision
fclass.s a3, ft10
## Double-Precision
fclass.d a3, ft10
