# Input instructions for the 'Zfh' extension.

# Floating-Point Computational Instructions
fadd.h f26, f27, f28
fsub.h f29, f30, f31
fmul.h ft0, ft1, ft2
fdiv.h ft3, ft4, ft5
fsqrt.h ft6, ft7
fmin.h fa5, fa6, fa7
fmax.h fs2, fs3, fs4
fmadd.h f10, f11, f12, f31
fmsub.h f14, f15, f16, f17
fnmsub.h f18, f19, f20, f21
fnmadd.h f22, f23, f24, f25

# Floating-Point Compare Instructions
feq.h a1, fs8, fs9
flt.h a2, fs10, fs11
fle.h a3, ft8, ft9

# Floating-Point Classify Instruction
fclass.h a3, ft10
