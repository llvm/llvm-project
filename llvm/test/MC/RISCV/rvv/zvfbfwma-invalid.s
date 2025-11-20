# RUN: not llvm-mc -triple riscv32 -mattr=+zvfbfwma,+d < %s 2>&1 | \
# RUN:   FileCheck %s
# RUN: not llvm-mc -triple riscv64 -mattr=+zvfbfwma,+d < %s 2>&1 | \
# RUN:   FileCheck %s

# Attempting to use fcvt instructions from zfhmin
fcvt.s.h fa0, ft0 # CHECK: [[@LINE]]:1: error: instruction requires the following: 'Zfh' (Half-Precision Floating-Point) or 'Zfhmin' (Half-Precision Floating-Point Minimal)
fcvt.h.s ft2, fa2 # CHECK: [[@LINE]]:1: error: instruction requires the following: 'Zfh' (Half-Precision Floating-Point) or 'Zfhmin' (Half-Precision Floating-Point Minimal)
fcvt.d.h fa0, ft0 # CHECK: [[@LINE]]:1: error: instruction requires the following: 'Zfh' (Half-Precision Floating-Point) or 'Zfhmin' (Half-Precision Floating-Point Minimal)
fcvt.h.d ft2, fa2 # CHECK: [[@LINE]]:1: error: instruction requires the following: 'Zfh' (Half-Precision Floating-Point) or 'Zfhmin' (Half-Precision Floating-Point Minimal)
