# RUN: not llvm-mc -triple riscv32 -mattr=+zfa,+zfh \
# RUN:     -riscv-no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXTD %s
# RUN: not llvm-mc -triple riscv64 -mattr=+zfa,+zfh \
# RUN:     -riscv-no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXTD %s

# CHECK-NO-EXTD: error: instruction requires the following: 'D' (Double-Precision Floating-Point){{$}}
fminm.d fa0, fa1, fa2

# CHECK-NO-EXTD: error: instruction requires the following: 'D' (Double-Precision Floating-Point){{$}}
fmaxm.d fs3, fs4, fs5

# CHECK-NO-EXTD: error: instruction requires the following: 'D' (Double-Precision Floating-Point){{$}}
fround.d fs1, fs2

# CHECK-NO-EXTD: error: instruction requires the following: 'D' (Double-Precision Floating-Point){{$}}
fround.d fs1, fs2, dyn

# CHECK-NO-EXTD: error: instruction requires the following: 'D' (Double-Precision Floating-Point){{$}}
fround.d fs1, fs2, rtz

# CHECK-NO-EXTD: error: instruction requires the following: 'D' (Double-Precision Floating-Point){{$}}
fround.d fs1, fs2, rne

# CHECK-NO-EXTD: error: instruction requires the following: 'D' (Double-Precision Floating-Point){{$}}
froundnx.d fs1, fs2

# CHECK-NO-EXTD: error: instruction requires the following: 'D' (Double-Precision Floating-Point){{$}}
froundnx.d fs1, fs2, dyn

# CHECK-NO-EXTD: error: instruction requires the following: 'D' (Double-Precision Floating-Point){{$}}
froundnx.d fs1, fs2, rtz

# CHECK-NO-EXTD: error: instruction requires the following: 'D' (Double-Precision Floating-Point){{$}}
froundnx.d fs1, fs2, rne

# CHECK-NO-EXTD: error: instruction requires the following: 'D' (Double-Precision Floating-Point){{$}}
fcvtmod.w.d a1, ft1, rtz

# CHECK-NO-EXTD: error: instruction requires the following: 'D' (Double-Precision Floating-Point){{$}}
fltq.d a1, fs1, fs2

# CHECK-NO-EXTD: error: instruction requires the following: 'D' (Double-Precision Floating-Point){{$}}
fleq.d a1, ft1, ft2
