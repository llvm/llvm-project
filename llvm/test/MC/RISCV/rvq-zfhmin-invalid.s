# RUN: llvm-mc --verify=no-zfhmin -triple=riscv64 -mattr=+q < %s
# RUN: llvm-mc --verify=no-q -triple=riscv64 -mattr=+zfhmin < %s

fcvt.q.h fa0, ft0
# no-zfhmin-error@-1:1 {{instruction requires the following: 'Zfh' (Half-Precision Floating-Point) or 'Zfhmin' (Half-Precision Floating-Point Minimal)}}
# no-q-error@-2:1 {{instruction requires the following: 'Q' (Quad-Precision Floating-Point)}}

fcvt.h.q ft2, fa2
# no-zfhmin-error@-1:1 {{instruction requires the following: 'Zfh' (Half-Precision Floating-Point) or 'Zfhmin' (Half-Precision Floating-Point Minimal)}}
# no-q-error@-2:1 {{instruction requires the following: 'Q' (Quad-Precision Floating-Point)}}
