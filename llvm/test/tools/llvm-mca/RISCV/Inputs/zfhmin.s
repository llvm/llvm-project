# Input instructions for the 'Zfhmin' extension.

# Floating-Point Load and Store Instructions
flh ft0, 0(a0)
fsh ft0, 0(a0)

# Floating-Point Conversion and Move Instructions
fmv.x.h a2, fs7
fmv.h.x ft1, a6

fcvt.s.h fa0, ft0
fcvt.s.h fa0, ft0, rup

fcvt.h.s ft2, fa2
fcvt.d.h fa0, ft0

fcvt.d.h fa0, ft0, rup
fcvt.h.d ft2, fa2

