# RUN: not llvm-mc -triple riscv32 -mattr=+zfa,+d \
# RUN:     -M no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXTZFH %s
# RUN: not llvm-mc -triple riscv64 -mattr=+zfa,+d \
# RUN:     -M no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXTZFH %s

# CHECK-NO-EXTZFH: error: instruction requires the following: 'Zfh' (Half-Precision Floating-Point){{$}}
fminm.h fa0, fa1, fa2

# CHECK-NO-EXTZFH: error: instruction requires the following: 'Zfh' (Half-Precision Floating-Point){{$}}
fmaxm.h fs3, fs4, fs5

# CHECK-NO-EXTZFH: error: instruction requires the following: 'Zfh' (Half-Precision Floating-Point){{$}}
fround.h fs1, fs2

# CHECK-NO-EXTZFH: error: instruction requires the following: 'Zfh' (Half-Precision Floating-Point){{$}}
fround.h fs1, fs2, dyn

# CHECK-NO-EXTZFH: error: instruction requires the following: 'Zfh' (Half-Precision Floating-Point){{$}}
fround.h fs1, fs2, rtz

# CHECK-NO-EXTZFH: error: instruction requires the following: 'Zfh' (Half-Precision Floating-Point){{$}}
fround.h fs1, fs2, rne

# CHECK-NO-EXTZFH: error: instruction requires the following: 'Zfh' (Half-Precision Floating-Point){{$}}
froundnx.h fs1, fs2

# CHECK-NO-EXTZFH: error: instruction requires the following: 'Zfh' (Half-Precision Floating-Point){{$}}
froundnx.h fs1, fs2, dyn

# CHECK-NO-EXTZFH: error: instruction requires the following: 'Zfh' (Half-Precision Floating-Point){{$}}
froundnx.h fs1, fs2, rtz

# CHECK-NO-EXTZFH: error: instruction requires the following: 'Zfh' (Half-Precision Floating-Point){{$}}
froundnx.h fs1, fs2, rne

# CHECK-NO-EXTZFH: error: instruction requires the following: 'Zfh' (Half-Precision Floating-Point){{$}}
fltq.h a1, fs1, fs2

# CHECK-NO-EXTZFH: error: instruction requires the following: 'Zfh' (Half-Precision Floating-Point){{$}}
fleq.h a1, ft1, ft2
