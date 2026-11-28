# RUN: llvm-mc --verify -triple riscv32 -mattr=+zvfbfwma,+d < %s
# RUN: llvm-mc --verify -triple riscv64 -mattr=+zvfbfwma,+d < %s

# Attempting to use fcvt instructions from zfhmin
fcvt.s.h fa0, ft0 # expected-error:1 {{instruction requires the following: 'Zfh' (Half-Precision Floating-Point) or 'Zfhmin' (Half-Precision Floating-Point Minimal)}}
fcvt.h.s ft2, fa2 # expected-error:1 {{instruction requires the following: 'Zfh' (Half-Precision Floating-Point) or 'Zfhmin' (Half-Precision Floating-Point Minimal)}}
fcvt.d.h fa0, ft0 # expected-error:1 {{instruction requires the following: 'Zfh' (Half-Precision Floating-Point) or 'Zfhmin' (Half-Precision Floating-Point Minimal)}}
fcvt.h.d ft2, fa2 # expected-error:1 {{instruction requires the following: 'Zfh' (Half-Precision Floating-Point) or 'Zfhmin' (Half-Precision Floating-Point Minimal)}}
