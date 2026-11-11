# RUN: not llvm-mc -triple=riscv64 -mattr=+q < %s 2>&1 \
# RUN:     | FileCheck --check-prefix=CHECK-NO-ZFHMIN %s
# RUN: not llvm-mc -triple=riscv64 -mattr=+zfhmin < %s 2>&1 \
# RUN:     | FileCheck --check-prefix=CHECK-NO-Q %s

# CHECK-NO-ZFHMIN: :[[@LINE+2]]:1: error: instruction requires the following: 'Zfh' (Half-Precision Floating-Point) or 'Zfhmin' (Half-Precision Floating-Point Minimal){{$}}
# CHECK-NO-Q: :[[@LINE+1]]:1: error: instruction requires the following: 'Q' (Quad-Precision Floating-Point){{$}}
fcvt.q.h fa0, ft0

# CHECK-NO-ZFHMIN: :[[@LINE+2]]:1: error: instruction requires the following: 'Zfh' (Half-Precision Floating-Point) or 'Zfhmin' (Half-Precision Floating-Point Minimal){{$}}
# CHECK-NO-Q: :[[@LINE+1]]:1: error: instruction requires the following: 'Q' (Quad-Precision Floating-Point){{$}}
fcvt.h.q ft2, fa2
