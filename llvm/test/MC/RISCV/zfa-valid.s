# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-zfa,+d,+zfh -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-zfa,+d,+zfh -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+experimental-zfa,+d,+zfh < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zfa,+d,+zfh -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-zfa,+d,+zfh < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zfa,+d,+zfh -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
#
# RUN: not llvm-mc -triple riscv32 -mattr=+d,+zfh \
# RUN:     -riscv-no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXT %s
# RUN: not llvm-mc -triple riscv64 -mattr=+d,+zfh \
# RUN:     -riscv-no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXT %s

# CHECK-ASM-AND-OBJ: fminm.s fa0, fa1, fa2
# CHECK-ASM: encoding: [0x53,0xa5,0xc5,0x28]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fminm.s fa0, fa1, fa2

# CHECK-ASM-AND-OBJ: fmaxm.s fs3, fs4, fs5
# CHECK-ASM: encoding: [0xd3,0x39,0x5a,0x29]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fmaxm.s fs3, fs4, fs5

# CHECK-ASM-AND-OBJ: fminm.d fa0, fa1, fa2
# CHECK-ASM: encoding: [0x53,0xa5,0xc5,0x2a]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fminm.d fa0, fa1, fa2

# CHECK-ASM-AND-OBJ: fmaxm.d fs3, fs4, fs5
# CHECK-ASM: encoding: [0xd3,0x39,0x5a,0x2b]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fmaxm.d fs3, fs4, fs5

# CHECK-ASM-AND-OBJ: fminm.h fa0, fa1, fa2
# CHECK-ASM: encoding: [0x53,0xa5,0xc5,0x2c]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fminm.h fa0, fa1, fa2

# CHECK-ASM-AND-OBJ: fmaxm.h fs3, fs4, fs5
# CHECK-ASM: encoding: [0xd3,0x39,0x5a,0x2d]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fmaxm.h fs3, fs4, fs5

# CHECK-ASM-AND-OBJ: fround.s fs1, fs2, dyn
# CHECK-ASM: encoding: [0xd3,0x74,0x49,0x40]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fround.s fs1, fs2

# CHECK-ASM-AND-OBJ: fround.s fs1, fs2, dyn
# CHECK-ASM: encoding: [0xd3,0x74,0x49,0x40]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fround.s fs1, fs2, dyn

# CHECK-ASM-AND-OBJ: fround.s fs1, fs2, rtz
# CHECK-ASM: encoding: [0xd3,0x14,0x49,0x40]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fround.s fs1, fs2, rtz

# CHECK-ASM-AND-OBJ: fround.s fs1, fs2, rne
# CHECK-ASM: encoding: [0xd3,0x04,0x49,0x40]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fround.s fs1, fs2, rne

# CHECK-ASM-AND-OBJ: froundnx.s fs1, fs2, dyn
# CHECK-ASM: encoding: [0xd3,0x74,0x59,0x40]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
froundnx.s fs1, fs2

# CHECK-ASM-AND-OBJ: froundnx.s fs1, fs2, dyn
# CHECK-ASM: encoding: [0xd3,0x74,0x59,0x40]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
froundnx.s fs1, fs2, dyn

# CHECK-ASM-AND-OBJ: froundnx.s fs1, fs2, rtz
# CHECK-ASM: encoding: [0xd3,0x14,0x59,0x40]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
froundnx.s fs1, fs2, rtz

# CHECK-ASM-AND-OBJ: froundnx.s fs1, fs2, rne
# CHECK-ASM: encoding: [0xd3,0x04,0x59,0x40]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
froundnx.s fs1, fs2, rne

# CHECK-ASM-AND-OBJ: fround.d fs1, fs2, dyn
# CHECK-ASM: encoding: [0xd3,0x74,0x49,0x42]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fround.d fs1, fs2

# CHECK-ASM-AND-OBJ: fround.d fs1, fs2, dyn
# CHECK-ASM: encoding: [0xd3,0x74,0x49,0x42]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fround.d fs1, fs2, dyn

# CHECK-ASM-AND-OBJ: fround.d fs1, fs2, rtz
# CHECK-ASM: encoding: [0xd3,0x14,0x49,0x42]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fround.d fs1, fs2, rtz

# CHECK-ASM-AND-OBJ: fround.d fs1, fs2, rne
# CHECK-ASM: encoding: [0xd3,0x04,0x49,0x42]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fround.d fs1, fs2, rne

# CHECK-ASM-AND-OBJ: froundnx.d fs1, fs2, dyn
# CHECK-ASM: encoding: [0xd3,0x74,0x59,0x42]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
froundnx.d fs1, fs2

# CHECK-ASM-AND-OBJ: froundnx.d fs1, fs2, dyn
# CHECK-ASM: encoding: [0xd3,0x74,0x59,0x42]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
froundnx.d fs1, fs2, dyn

# CHECK-ASM-AND-OBJ: froundnx.d fs1, fs2, rtz
# CHECK-ASM: encoding: [0xd3,0x14,0x59,0x42]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
froundnx.d fs1, fs2, rtz

# CHECK-ASM-AND-OBJ: froundnx.d fs1, fs2, rne
# CHECK-ASM: encoding: [0xd3,0x04,0x59,0x42]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
froundnx.d fs1, fs2, rne

# CHECK-ASM-AND-OBJ: fround.h ft1, fa1, dyn
# CHECK-ASM: encoding: [0xd3,0xf0,0x45,0x44]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fround.h ft1, fa1

# CHECK-ASM-AND-OBJ: fround.h ft1, fa1, dyn
# CHECK-ASM: encoding: [0xd3,0xf0,0x45,0x44]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fround.h ft1, fa1, dyn

# CHECK-ASM-AND-OBJ: fround.h ft1, fa1, rtz
# CHECK-ASM: encoding: [0xd3,0x90,0x45,0x44]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fround.h ft1, fa1, rtz

# CHECK-ASM-AND-OBJ: fround.h fs1, fs2, rne
# CHECK-ASM: encoding: [0xd3,0x04,0x49,0x44]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fround.h fs1, fs2, rne

# CHECK-ASM-AND-OBJ: froundnx.h ft1, fa1, dyn
# CHECK-ASM: encoding: [0xd3,0xf0,0x55,0x44]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
froundnx.h ft1, fa1

# CHECK-ASM-AND-OBJ: froundnx.h ft1, fa1, dyn
# CHECK-ASM: encoding: [0xd3,0xf0,0x55,0x44]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
froundnx.h ft1, fa1, dyn

# CHECK-ASM-AND-OBJ: froundnx.h ft1, fa1, rtz
# CHECK-ASM: encoding: [0xd3,0x90,0x55,0x44]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
froundnx.h ft1, fa1, rtz

# CHECK-ASM-AND-OBJ: froundnx.h fs1, fs2, rne
# CHECK-ASM: encoding: [0xd3,0x04,0x59,0x44]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
froundnx.h fs1, fs2, rne

# CHECK-ASM-AND-OBJ: fcvtmod.w.d a1, ft1, rtz
# CHECK-ASM: encoding: [0xd3,0x95,0x80,0xc2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fcvtmod.w.d a1, ft1, rtz

# CHECK-ASM-AND-OBJ: fltq.s a1, fs1, fs2
# CHECK-ASM: encoding: [0xd3,0xd5,0x24,0xa1]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fltq.s a1, fs1, fs2

# CHECK-ASM-AND-OBJ: fleq.s a1, ft1, ft2
# CHECK-ASM: encoding: [0xd3,0xc5,0x20,0xa0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fleq.s a1, ft1, ft2

# CHECK-ASM-AND-OBJ: fltq.s a1, fs2, fs1
# CHECK-ASM: encoding: [0xd3,0x55,0x99,0xa0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fgtq.s a1, fs1, fs2

# CHECK-ASM-AND-OBJ: fleq.s a1, ft2, ft1
# CHECK-ASM: encoding: [0xd3,0x45,0x11,0xa0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fgeq.s a1, ft1, ft2

# CHECK-ASM-AND-OBJ: fltq.d a1, fs1, fs2
# CHECK-ASM: encoding: [0xd3,0xd5,0x24,0xa3]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fltq.d a1, fs1, fs2

# CHECK-ASM-AND-OBJ: fleq.d a1, ft1, ft2
# CHECK-ASM: encoding: [0xd3,0xc5,0x20,0xa2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fleq.d a1, ft1, ft2

# CHECK-ASM-AND-OBJ: fltq.d a1, fs2, fs1
# CHECK-ASM: encoding: [0xd3,0x55,0x99,0xa2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fgtq.d a1, fs1, fs2

# CHECK-ASM-AND-OBJ: fleq.d a1, ft2, ft1
# CHECK-ASM: encoding: [0xd3,0x45,0x11,0xa2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fgeq.d a1, ft1, ft2

# CHECK-ASM-AND-OBJ: fltq.h a1, fs1, fs2
# CHECK-ASM: encoding: [0xd3,0xd5,0x24,0xa5]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fltq.h a1, fs1, fs2

# CHECK-ASM-AND-OBJ: fleq.h a1, ft1, ft2
# CHECK-ASM: encoding: [0xd3,0xc5,0x20,0xa4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fleq.h a1, ft1, ft2

# CHECK-ASM-AND-OBJ: fltq.h a1, fs2, fs1
# CHECK-ASM: encoding: [0xd3,0x55,0x99,0xa4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fgtq.h a1, fs1, fs2

# CHECK-ASM-AND-OBJ: fleq.h a1, ft2, ft1
# CHECK-ASM: encoding: [0xd3,0x45,0x11,0xa4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fgeq.h a1, ft1, ft2
