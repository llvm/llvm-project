# RUN: not llvm-mc -triple riscv32 -mattr=+zfa,+zfh \
# RUN:     -M no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXTQ %s
# RUN: not llvm-mc -triple riscv64 -mattr=+zfa,+zfh \
# RUN:     -M no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXTQ %s

# CHECK-NO-EXTQ: error: instruction requires the following: 'Q' (Quad-Precision Floating-Point){{$}}
fminm.q fa0, fa1, fa2

# CHECK-NO-EXTQ: error: instruction requires the following: 'Q' (Quad-Precision Floating-Point){{$}}
fmaxm.q fs3, fs4, fs5

# CHECK-NO-EXTQ: error: instruction requires the following: 'Q' (Quad-Precision Floating-Point){{$}}
fround.q fs1, fs2

# CHECK-NO-EXTQ: error: instruction requires the following: 'Q' (Quad-Precision Floating-Point){{$}}
fround.q fs1, fs2, dyn

# CHECK-NO-EXTQ: error: instruction requires the following: 'Q' (Quad-Precision Floating-Point){{$}}
fround.q fs1, fs2, rtz

# CHECK-NO-EXTQ: error: instruction requires the following: 'Q' (Quad-Precision Floating-Point){{$}}
fround.q fs1, fs2, rne

# CHECK-NO-EXTQ: error: instruction requires the following: 'Q' (Quad-Precision Floating-Point){{$}}
froundnx.q fs1, fs2

# CHECK-NO-EXTQ: error: instruction requires the following: 'Q' (Quad-Precision Floating-Point){{$}}
froundnx.q fs1, fs2, dyn

# CHECK-NO-EXTQ: error: instruction requires the following: 'Q' (Quad-Precision Floating-Point){{$}}
froundnx.q fs1, fs2, rtz

# CHECK-NO-EXTQ: error: instruction requires the following: 'Q' (Quad-Precision Floating-Point){{$}}
froundnx.q fs1, fs2, rne

# CHECK-NO-EXTQ: error: instruction requires the following: 'Q' (Quad-Precision Floating-Point){{$}}
fltq.q a1, fs1, fs2

# CHECK-NO-EXTQ: error: instruction requires the following: 'Q' (Quad-Precision Floating-Point){{$}}
fleq.q a1, ft1, ft2
