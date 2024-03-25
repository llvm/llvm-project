# RUN: llvm-mc %s -triple=riscv32 -mattr=+zfa,+d,+zfh -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zfa,+d,+zfh -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+zfa,+d,+zfh < %s \
# RUN:     | llvm-objdump --mattr=+zfa,+d,+zfh -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zfa,+d,+zfh < %s \
# RUN:     | llvm-objdump --mattr=+zfa,+d,+zfh -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
#
# RUN: not llvm-mc -triple riscv32 -mattr=+d,+zfh \
# RUN:     -riscv-no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXT %s
# RUN: not llvm-mc -triple riscv64 -mattr=+d,+zfh \
# RUN:     -riscv-no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXT %s

# CHECK-ASM-AND-OBJ: fli.s ft1, -1.0
# CHECK-ASM: encoding: [0xd3,0x00,0x10,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, -1.000000e+00

# CHECK-ASM-AND-OBJ: fli.s ft1, -1.0
# CHECK-ASM: encoding: [0xd3,0x00,0x10,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, -0x1p+0

# CHECK-ASM-AND-OBJ: fli.s ft1, min
# CHECK-ASM: encoding: [0xd3,0x80,0x10,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, min

# CHECK-ASM-AND-OBJ: fli.s ft1, 1.52587890625e-05
# CHECK-ASM: encoding: [0xd3,0x00,0x11,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 1.52587890625e-05

# CHECK-ASM-AND-OBJ: fli.s ft1, 1.52587890625e-05
# CHECK-ASM: encoding: [0xd3,0x00,0x11,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 0x1p-16

# CHECK-ASM-AND-OBJ: fli.s ft1, 3.0517578125e-05
# CHECK-ASM: encoding: [0xd3,0x80,0x11,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 3.0517578125e-05

# CHECK-ASM-AND-OBJ: fli.s ft1, 3.0517578125e-05
# CHECK-ASM: encoding: [0xd3,0x80,0x11,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 0x1p-15

# CHECK-ASM-AND-OBJ: fli.s ft1, 0.00390625
# CHECK-ASM: encoding: [0xd3,0x00,0x12,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 3.906250e-03

# CHECK-ASM-AND-OBJ: fli.s ft1, 0.00390625
# CHECK-ASM: encoding: [0xd3,0x00,0x12,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 0x1p-8

# CHECK-ASM-AND-OBJ: fli.s ft1, 0.0078125
# CHECK-ASM: encoding: [0xd3,0x80,0x12,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 7.812500e-03

# CHECK-ASM-AND-OBJ: fli.s ft1, 0.0078125
# CHECK-ASM: encoding: [0xd3,0x80,0x12,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 0x1p-7

# CHECK-ASM-AND-OBJ: fli.s ft1, 0.0625
# CHECK-ASM: encoding: [0xd3,0x00,0x13,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 6.250000e-02

# CHECK-ASM-AND-OBJ: fli.s ft1, 0.0625
# CHECK-ASM: encoding: [0xd3,0x00,0x13,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 0x1p-4

# CHECK-ASM-AND-OBJ: fli.s ft1, 0.125
# CHECK-ASM: encoding: [0xd3,0x80,0x13,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 1.250000e-01

# CHECK-ASM-AND-OBJ: fli.s ft1, 0.125
# CHECK-ASM: encoding: [0xd3,0x80,0x13,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 0x1p-3

# CHECK-ASM-AND-OBJ: fli.s ft1, 0.25
# CHECK-ASM: encoding: [0xd3,0x00,0x14,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 2.500000e-01

# CHECK-ASM-AND-OBJ: fli.s ft1, 0.25
# CHECK-ASM: encoding: [0xd3,0x00,0x14,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 0x1p-2

# CHECK-ASM-AND-OBJ: fli.s ft1, 0.3125
# CHECK-ASM: encoding: [0xd3,0x80,0x14,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 3.125000e-01

# CHECK-ASM-AND-OBJ: fli.s ft1, 0.3125
# CHECK-ASM: encoding: [0xd3,0x80,0x14,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 0x1.4p-2

# CHECK-ASM-AND-OBJ: fli.s ft1, 0.375
# CHECK-ASM: encoding: [0xd3,0x00,0x15,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 3.750000e-01

# CHECK-ASM-AND-OBJ: fli.s ft1, 0.375
# CHECK-ASM: encoding: [0xd3,0x00,0x15,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 0x1.8p-2

# CHECK-ASM-AND-OBJ: fli.s ft1, 0.4375
# CHECK-ASM: encoding: [0xd3,0x80,0x15,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 4.375000e-01

# CHECK-ASM-AND-OBJ: fli.s ft1, 0.4375
# CHECK-ASM: encoding: [0xd3,0x80,0x15,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 0x1.cp-2

# CHECK-ASM-AND-OBJ: fli.s ft1, 0.5
# CHECK-ASM: encoding: [0xd3,0x00,0x16,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 5.000000e-01

# CHECK-ASM-AND-OBJ: fli.s ft1, 0.5
# CHECK-ASM: encoding: [0xd3,0x00,0x16,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 0x1p-1

# CHECK-ASM-AND-OBJ: fli.s ft1, 0.625
# CHECK-ASM: encoding: [0xd3,0x80,0x16,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 6.250000e-01

# CHECK-ASM-AND-OBJ: fli.s ft1, 0.625
# CHECK-ASM: encoding: [0xd3,0x80,0x16,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 0x1.4p-1

# CHECK-ASM-AND-OBJ: fli.s ft1, 0.75
# CHECK-ASM: encoding: [0xd3,0x00,0x17,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 7.500000e-01

# CHECK-ASM-AND-OBJ: fli.s ft1, 0.75
# CHECK-ASM: encoding: [0xd3,0x00,0x17,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 0x1.8p-1

# CHECK-ASM-AND-OBJ: fli.s ft1, 0.875
# CHECK-ASM: encoding: [0xd3,0x80,0x17,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 8.750000e-01

# CHECK-ASM-AND-OBJ: fli.s ft1, 0.875
# CHECK-ASM: encoding: [0xd3,0x80,0x17,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 0x1.cp-1

# CHECK-ASM-AND-OBJ: fli.s ft1, 1.0
# CHECK-ASM: encoding: [0xd3,0x00,0x18,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 1.000000e+00

# CHECK-ASM-AND-OBJ: fli.s ft1, 1.0
# CHECK-ASM: encoding: [0xd3,0x00,0x18,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 0x1p+0

# CHECK-ASM-AND-OBJ: fli.s ft1, 1.25
# CHECK-ASM: encoding: [0xd3,0x80,0x18,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 1.250000e+00

# CHECK-ASM-AND-OBJ: fli.s ft1, 1.25
# CHECK-ASM: encoding: [0xd3,0x80,0x18,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 0x1.4p+0

# CHECK-ASM-AND-OBJ: fli.s ft1, 1.5
# CHECK-ASM: encoding: [0xd3,0x00,0x19,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 1.500000e+00

# CHECK-ASM-AND-OBJ: fli.s ft1, 1.5
# CHECK-ASM: encoding: [0xd3,0x00,0x19,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 0x1.8p+0

# CHECK-ASM-AND-OBJ: fli.s ft1, 1.75
# CHECK-ASM: encoding: [0xd3,0x80,0x19,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 1.750000e+00

# CHECK-ASM-AND-OBJ: fli.s ft1, 1.75
# CHECK-ASM: encoding: [0xd3,0x80,0x19,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 0x1.cp+0

# CHECK-ASM-AND-OBJ: fli.s ft1, 2.0
# CHECK-ASM: encoding: [0xd3,0x00,0x1a,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 2.000000e+00

# CHECK-ASM-AND-OBJ: fli.s ft1, 2.0
# CHECK-ASM: encoding: [0xd3,0x00,0x1a,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 0x1p+1

# CHECK-ASM-AND-OBJ: fli.s ft1, 2.5
# CHECK-ASM: encoding: [0xd3,0x80,0x1a,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 2.500000e+00

# CHECK-ASM-AND-OBJ: fli.s ft1, 2.5
# CHECK-ASM: encoding: [0xd3,0x80,0x1a,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 0x1.4p+1

# CHECK-ASM-AND-OBJ: fli.s ft1, 3.0
# CHECK-ASM: encoding: [0xd3,0x00,0x1b,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 3.000000e+00

# CHECK-ASM-AND-OBJ: fli.s ft1, 3.0
# CHECK-ASM: encoding: [0xd3,0x00,0x1b,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 0x1.8p+1

# CHECK-ASM-AND-OBJ: fli.s ft1, 4.0
# CHECK-ASM: encoding: [0xd3,0x80,0x1b,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 4.000000e+00

# CHECK-ASM-AND-OBJ: fli.s ft1, 4.0
# CHECK-ASM: encoding: [0xd3,0x80,0x1b,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 0x1p+2

# CHECK-ASM-AND-OBJ: fli.s ft1, 8.0
# CHECK-ASM: encoding: [0xd3,0x00,0x1c,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 8.000000e+00

# CHECK-ASM-AND-OBJ: fli.s ft1, 8.0
# CHECK-ASM: encoding: [0xd3,0x00,0x1c,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 0x1p+3

# CHECK-ASM-AND-OBJ: fli.s ft1, 16.0
# CHECK-ASM: encoding: [0xd3,0x80,0x1c,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 1.600000e+01

# CHECK-ASM-AND-OBJ: fli.s ft1, 16.0
# CHECK-ASM: encoding: [0xd3,0x80,0x1c,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 0x1p+4

# CHECK-ASM-AND-OBJ: fli.s ft1, 128.0
# CHECK-ASM: encoding: [0xd3,0x00,0x1d,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 1.280000e+02

# CHECK-ASM-AND-OBJ: fli.s ft1, 128.0
# CHECK-ASM: encoding: [0xd3,0x00,0x1d,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 0x1p+7

# CHECK-ASM-AND-OBJ: fli.s ft1, 256.0
# CHECK-ASM: encoding: [0xd3,0x80,0x1d,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 2.560000e+02

# CHECK-ASM-AND-OBJ: fli.s ft1, 256.0
# CHECK-ASM: encoding: [0xd3,0x80,0x1d,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 0x1p+8

# CHECK-ASM-AND-OBJ: fli.s ft1, 32768.0
# CHECK-ASM: encoding: [0xd3,0x00,0x1e,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 3.276800e+04

# CHECK-ASM-AND-OBJ: fli.s ft1, 32768.0
# CHECK-ASM: encoding: [0xd3,0x00,0x1e,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 0x1p+15

# CHECK-ASM-AND-OBJ: fli.s ft1, 65536.0
# CHECK-ASM: encoding: [0xd3,0x80,0x1e,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 6.553600e+04

# CHECK-ASM-AND-OBJ: fli.s ft1, 65536.0
# CHECK-ASM: encoding: [0xd3,0x80,0x1e,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 0x1p+16

# CHECK-ASM-AND-OBJ: fli.s ft1, inf
# CHECK-ASM: encoding: [0xd3,0x00,0x1f,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, INF

# CHECK-ASM-AND-OBJ: fli.s ft1, nan
# CHECK-ASM: encoding: [0xd3,0x80,0x1f,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, nan

# CHECK-ASM-AND-OBJ: fli.d ft1, -1.0
# CHECK-ASM: encoding: [0xd3,0x00,0x10,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, -1.000000e+00

# CHECK-ASM-AND-OBJ: fli.d ft1, -1.0
# CHECK-ASM: encoding: [0xd3,0x00,0x10,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, -0x1p+0

# CHECK-ASM-AND-OBJ: fli.d ft1, min
# CHECK-ASM: encoding: [0xd3,0x80,0x10,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, min

# CHECK-ASM-AND-OBJ: fli.d ft1, 1.52587890625e-05
# CHECK-ASM: encoding: [0xd3,0x00,0x11,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 1.52587890625e-05

# CHECK-ASM-AND-OBJ: fli.d ft1, 1.52587890625e-05
# CHECK-ASM: encoding: [0xd3,0x00,0x11,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 0x1p-16

# CHECK-ASM-AND-OBJ: fli.d ft1, 3.0517578125e-05
# CHECK-ASM: encoding: [0xd3,0x80,0x11,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 3.0517578125e-05

# CHECK-ASM-AND-OBJ: fli.d ft1, 3.0517578125e-05
# CHECK-ASM: encoding: [0xd3,0x80,0x11,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 0x1p-15

# CHECK-ASM-AND-OBJ: fli.d ft1, 0.00390625
# CHECK-ASM: encoding: [0xd3,0x00,0x12,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 3.906250e-03

# CHECK-ASM-AND-OBJ: fli.d ft1, 0.00390625
# CHECK-ASM: encoding: [0xd3,0x00,0x12,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 0x1p-8

# CHECK-ASM-AND-OBJ: fli.d ft1, 0.0078125
# CHECK-ASM: encoding: [0xd3,0x80,0x12,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 7.812500e-03

# CHECK-ASM-AND-OBJ: fli.d ft1, 0.0078125
# CHECK-ASM: encoding: [0xd3,0x80,0x12,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 0x1p-7

# CHECK-ASM-AND-OBJ: fli.d ft1, 0.0625
# CHECK-ASM: encoding: [0xd3,0x00,0x13,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 6.250000e-02

# CHECK-ASM-AND-OBJ: fli.d ft1, 0.0625
# CHECK-ASM: encoding: [0xd3,0x00,0x13,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 0x1p-4

# CHECK-ASM-AND-OBJ: fli.d ft1, 0.125
# CHECK-ASM: encoding: [0xd3,0x80,0x13,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 1.250000e-01

# CHECK-ASM-AND-OBJ: fli.d ft1, 0.125
# CHECK-ASM: encoding: [0xd3,0x80,0x13,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 0x1p-3

# CHECK-ASM-AND-OBJ: fli.d ft1, 0.25
# CHECK-ASM: encoding: [0xd3,0x00,0x14,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 2.500000e-01

# CHECK-ASM-AND-OBJ: fli.d ft1, 0.25
# CHECK-ASM: encoding: [0xd3,0x00,0x14,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 0x1p-2

# CHECK-ASM-AND-OBJ: fli.d ft1, 0.3125
# CHECK-ASM: encoding: [0xd3,0x80,0x14,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 3.125000e-01

# CHECK-ASM-AND-OBJ: fli.d ft1, 0.3125
# CHECK-ASM: encoding: [0xd3,0x80,0x14,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 0x1.4p-2

# CHECK-ASM-AND-OBJ: fli.d ft1, 0.375
# CHECK-ASM: encoding: [0xd3,0x00,0x15,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 3.750000e-01

# CHECK-ASM-AND-OBJ: fli.d ft1, 0.375
# CHECK-ASM: encoding: [0xd3,0x00,0x15,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 0x1.8p-2

# CHECK-ASM-AND-OBJ: fli.d ft1, 0.4375
# CHECK-ASM: encoding: [0xd3,0x80,0x15,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 4.375000e-01

# CHECK-ASM-AND-OBJ: fli.d ft1, 0.4375
# CHECK-ASM: encoding: [0xd3,0x80,0x15,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 0x1.cp-2

# CHECK-ASM-AND-OBJ: fli.d ft1, 0.5
# CHECK-ASM: encoding: [0xd3,0x00,0x16,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 5.000000e-01

# CHECK-ASM-AND-OBJ: fli.d ft1, 0.5
# CHECK-ASM: encoding: [0xd3,0x00,0x16,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 0x1p-1

# CHECK-ASM-AND-OBJ: fli.d ft1, 0.625
# CHECK-ASM: encoding: [0xd3,0x80,0x16,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 6.250000e-01

# CHECK-ASM-AND-OBJ: fli.d ft1, 0.625
# CHECK-ASM: encoding: [0xd3,0x80,0x16,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 0x1.4p-1

# CHECK-ASM-AND-OBJ: fli.d ft1, 0.75
# CHECK-ASM: encoding: [0xd3,0x00,0x17,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 7.500000e-01

# CHECK-ASM-AND-OBJ: fli.d ft1, 0.75
# CHECK-ASM: encoding: [0xd3,0x00,0x17,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 0x1.8p-1

# CHECK-ASM-AND-OBJ: fli.d ft1, 0.875
# CHECK-ASM: encoding: [0xd3,0x80,0x17,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 8.750000e-01

# CHECK-ASM-AND-OBJ: fli.d ft1, 0.875
# CHECK-ASM: encoding: [0xd3,0x80,0x17,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 0x1.cp-1

# CHECK-ASM-AND-OBJ: fli.d ft1, 1.0
# CHECK-ASM: encoding: [0xd3,0x00,0x18,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 1.000000e+00

# CHECK-ASM-AND-OBJ: fli.d ft1, 1.0
# CHECK-ASM: encoding: [0xd3,0x00,0x18,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 0x1p+0

# CHECK-ASM-AND-OBJ: fli.d ft1, 1.25
# CHECK-ASM: encoding: [0xd3,0x80,0x18,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 1.250000e+00


# CHECK-ASM-AND-OBJ: fli.d ft1, 1.25
# CHECK-ASM: encoding: [0xd3,0x80,0x18,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 0x1.4p+0

# CHECK-ASM-AND-OBJ: fli.d ft1, 1.5
# CHECK-ASM: encoding: [0xd3,0x00,0x19,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 1.500000e+00


# CHECK-ASM-AND-OBJ: fli.d ft1, 1.5
# CHECK-ASM: encoding: [0xd3,0x00,0x19,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 0x1.8p+0

# CHECK-ASM-AND-OBJ: fli.d ft1, 1.75
# CHECK-ASM: encoding: [0xd3,0x80,0x19,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 1.750000e+00

# CHECK-ASM-AND-OBJ: fli.d ft1, 1.75
# CHECK-ASM: encoding: [0xd3,0x80,0x19,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 0x1.cp+0

# CHECK-ASM-AND-OBJ: fli.d ft1, 2.0
# CHECK-ASM: encoding: [0xd3,0x00,0x1a,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 2.000000e+00

# CHECK-ASM-AND-OBJ: fli.d ft1, 2.0
# CHECK-ASM: encoding: [0xd3,0x00,0x1a,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 0x1p+1

# CHECK-ASM-AND-OBJ: fli.d ft1, 2.5
# CHECK-ASM: encoding: [0xd3,0x80,0x1a,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 2.500000e+00

# CHECK-ASM-AND-OBJ: fli.d ft1, 2.5
# CHECK-ASM: encoding: [0xd3,0x80,0x1a,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 0x1.4p+1

# CHECK-ASM-AND-OBJ: fli.d ft1, 3.0
# CHECK-ASM: encoding: [0xd3,0x00,0x1b,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 3.000000e+00

# CHECK-ASM-AND-OBJ: fli.d ft1, 3.0
# CHECK-ASM: encoding: [0xd3,0x00,0x1b,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 0x1.8p+1

# CHECK-ASM-AND-OBJ: fli.d ft1, 4.0
# CHECK-ASM: encoding: [0xd3,0x80,0x1b,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 4.000000e+00

# CHECK-ASM-AND-OBJ: fli.d ft1, 4.0
# CHECK-ASM: encoding: [0xd3,0x80,0x1b,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 0x1p+2

# CHECK-ASM-AND-OBJ: fli.d ft1, 8.0
# CHECK-ASM: encoding: [0xd3,0x00,0x1c,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 8.000000e+00

# CHECK-ASM-AND-OBJ: fli.d ft1, 8.0
# CHECK-ASM: encoding: [0xd3,0x00,0x1c,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 0x1p+3

# CHECK-ASM-AND-OBJ: fli.d ft1, 16.0
# CHECK-ASM: encoding: [0xd3,0x80,0x1c,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 1.600000e+01

# CHECK-ASM-AND-OBJ: fli.d ft1, 16.0
# CHECK-ASM: encoding: [0xd3,0x80,0x1c,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 0x1p+4

# CHECK-ASM-AND-OBJ: fli.d ft1, 128.0
# CHECK-ASM: encoding: [0xd3,0x00,0x1d,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 1.280000e+02

# CHECK-ASM-AND-OBJ: fli.d ft1, 128.0
# CHECK-ASM: encoding: [0xd3,0x00,0x1d,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 0x1p+7

# CHECK-ASM-AND-OBJ: fli.d ft1, 256.0
# CHECK-ASM: encoding: [0xd3,0x80,0x1d,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 2.560000e+02

# CHECK-ASM-AND-OBJ: fli.d ft1, 256.0
# CHECK-ASM: encoding: [0xd3,0x80,0x1d,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 0x1p+8

# CHECK-ASM-AND-OBJ: fli.d ft1, 32768.0
# CHECK-ASM: encoding: [0xd3,0x00,0x1e,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 3.276800e+04

# CHECK-ASM-AND-OBJ: fli.d ft1, 32768.0
# CHECK-ASM: encoding: [0xd3,0x00,0x1e,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 0x1p+15

# CHECK-ASM-AND-OBJ: fli.d ft1, 65536.0
# CHECK-ASM: encoding: [0xd3,0x80,0x1e,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 6.553600e+04

# CHECK-ASM-AND-OBJ: fli.d ft1, 65536.0
# CHECK-ASM: encoding: [0xd3,0x80,0x1e,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 0x1p+16

# CHECK-ASM-AND-OBJ: fli.d ft1, inf
# CHECK-ASM: encoding: [0xd3,0x00,0x1f,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, INF

# CHECK-ASM-AND-OBJ: fli.d ft1, nan
# CHECK-ASM: encoding: [0xd3,0x80,0x1f,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, nan

# CHECK-ASM-AND-OBJ: fli.h ft1, -1.0
# CHECK-ASM: encoding: [0xd3,0x00,0x10,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, -1.000000e+00

# CHECK-ASM-AND-OBJ: fli.h ft1, -1.0
# CHECK-ASM: encoding: [0xd3,0x00,0x10,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, -0x1p+0

# CHECK-ASM-AND-OBJ: fli.h ft1, min
# CHECK-ASM: encoding: [0xd3,0x80,0x10,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, min

# CHECK-ASM-AND-OBJ: fli.h ft1, 1.52587890625e-05
# CHECK-ASM: encoding: [0xd3,0x00,0x11,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 1.52587890625e-05

# CHECK-ASM-AND-OBJ: fli.h ft1, 1.52587890625e-05
# CHECK-ASM: encoding: [0xd3,0x00,0x11,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 0x1p-16

# CHECK-ASM-AND-OBJ: fli.h ft1, 3.0517578125e-05
# CHECK-ASM: encoding: [0xd3,0x80,0x11,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 3.0517578125e-05

# CHECK-ASM-AND-OBJ: fli.h ft1, 3.0517578125e-05
# CHECK-ASM: encoding: [0xd3,0x80,0x11,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 0x1p-15

# CHECK-ASM-AND-OBJ: fli.h ft1, 0.00390625
# CHECK-ASM: encoding: [0xd3,0x00,0x12,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 3.906250e-03

# CHECK-ASM-AND-OBJ: fli.h ft1, 0.00390625
# CHECK-ASM: encoding: [0xd3,0x00,0x12,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 0x1p-8

# CHECK-ASM-AND-OBJ: fli.h ft1, 0.0078125
# CHECK-ASM: encoding: [0xd3,0x80,0x12,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 7.812500e-03

# CHECK-ASM-AND-OBJ: fli.h ft1, 0.0078125
# CHECK-ASM: encoding: [0xd3,0x80,0x12,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 0x1p-7

# CHECK-ASM-AND-OBJ: fli.h ft1, 0.0625
# CHECK-ASM: encoding: [0xd3,0x00,0x13,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 6.250000e-02

# CHECK-ASM-AND-OBJ: fli.h ft1, 0.0625
# CHECK-ASM: encoding: [0xd3,0x00,0x13,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 0x1p-4

# CHECK-ASM-AND-OBJ: fli.h ft1, 0.125
# CHECK-ASM: encoding: [0xd3,0x80,0x13,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 1.250000e-01

# CHECK-ASM-AND-OBJ: fli.h ft1, 0.125
# CHECK-ASM: encoding: [0xd3,0x80,0x13,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 0x1p-3

# CHECK-ASM-AND-OBJ: fli.h ft1, 0.25
# CHECK-ASM: encoding: [0xd3,0x00,0x14,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 2.500000e-01

# CHECK-ASM-AND-OBJ: fli.h ft1, 0.25
# CHECK-ASM: encoding: [0xd3,0x00,0x14,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 0x1p-2

# CHECK-ASM-AND-OBJ: fli.h ft1, 0.3125
# CHECK-ASM: encoding: [0xd3,0x80,0x14,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 3.125000e-01

# CHECK-ASM-AND-OBJ: fli.h ft1, 0.3125
# CHECK-ASM: encoding: [0xd3,0x80,0x14,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 0x1.4p-2

# CHECK-ASM-AND-OBJ: fli.h ft1, 0.375
# CHECK-ASM: encoding: [0xd3,0x00,0x15,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 3.750000e-01

# CHECK-ASM-AND-OBJ: fli.h ft1, 0.375
# CHECK-ASM: encoding: [0xd3,0x00,0x15,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 0x1.8p-2

# CHECK-ASM-AND-OBJ: fli.h ft1, 0.4375
# CHECK-ASM: encoding: [0xd3,0x80,0x15,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 4.375000e-01

# CHECK-ASM-AND-OBJ: fli.h ft1, 0.4375
# CHECK-ASM: encoding: [0xd3,0x80,0x15,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 0x1.cp-2

# CHECK-ASM-AND-OBJ: fli.h ft1, 0.5
# CHECK-ASM: encoding: [0xd3,0x00,0x16,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 5.000000e-01

# CHECK-ASM-AND-OBJ: fli.h ft1, 0.5
# CHECK-ASM: encoding: [0xd3,0x00,0x16,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 0x1p-1

# CHECK-ASM-AND-OBJ: fli.h ft1, 0.625
# CHECK-ASM: encoding: [0xd3,0x80,0x16,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 6.250000e-01

# CHECK-ASM-AND-OBJ: fli.h ft1, 0.625
# CHECK-ASM: encoding: [0xd3,0x80,0x16,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 0x1.4p-1

# CHECK-ASM-AND-OBJ: fli.h ft1, 0.75
# CHECK-ASM: encoding: [0xd3,0x00,0x17,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 7.500000e-01

# CHECK-ASM-AND-OBJ: fli.h ft1, 0.75
# CHECK-ASM: encoding: [0xd3,0x00,0x17,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 0x1.8p-1

# CHECK-ASM-AND-OBJ: fli.h ft1, 0.875
# CHECK-ASM: encoding: [0xd3,0x80,0x17,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 8.750000e-01

# CHECK-ASM-AND-OBJ: fli.h ft1, 0.875
# CHECK-ASM: encoding: [0xd3,0x80,0x17,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 0x1.cp-1

# CHECK-ASM-AND-OBJ: fli.h ft1, 1.0
# CHECK-ASM: encoding: [0xd3,0x00,0x18,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 1.000000e+00

# CHECK-ASM-AND-OBJ: fli.h ft1, 1.0
# CHECK-ASM: encoding: [0xd3,0x00,0x18,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 0x1p+0

# CHECK-ASM-AND-OBJ: fli.h ft1, 1.25
# CHECK-ASM: encoding: [0xd3,0x80,0x18,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 1.250000e+00

# CHECK-ASM-AND-OBJ: fli.h ft1, 1.25
# CHECK-ASM: encoding: [0xd3,0x80,0x18,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 0x1.4p+0

# CHECK-ASM-AND-OBJ: fli.h ft1, 1.5
# CHECK-ASM: encoding: [0xd3,0x00,0x19,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 1.500000e+00

# CHECK-ASM-AND-OBJ: fli.h ft1, 1.5
# CHECK-ASM: encoding: [0xd3,0x00,0x19,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 0x1.8p+0

# CHECK-ASM-AND-OBJ: fli.h ft1, 1.75
# CHECK-ASM: encoding: [0xd3,0x80,0x19,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 1.750000e+00

# CHECK-ASM-AND-OBJ: fli.h ft1, 1.75
# CHECK-ASM: encoding: [0xd3,0x80,0x19,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 0x1.cp+0

# CHECK-ASM-AND-OBJ: fli.h ft1, 2.0
# CHECK-ASM: encoding: [0xd3,0x00,0x1a,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 2.000000e+00

# CHECK-ASM-AND-OBJ: fli.h ft1, 2.0
# CHECK-ASM: encoding: [0xd3,0x00,0x1a,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 0x1p+1

# CHECK-ASM-AND-OBJ: fli.h ft1, 2.5
# CHECK-ASM: encoding: [0xd3,0x80,0x1a,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 2.500000e+00

# CHECK-ASM-AND-OBJ: fli.h ft1, 2.5
# CHECK-ASM: encoding: [0xd3,0x80,0x1a,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 0x1.4p+1

# CHECK-ASM-AND-OBJ: fli.h ft1, 3.0
# CHECK-ASM: encoding: [0xd3,0x00,0x1b,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 3.000000e+00

# CHECK-ASM-AND-OBJ: fli.h ft1, 3.0
# CHECK-ASM: encoding: [0xd3,0x00,0x1b,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 0x1.8p+1

# CHECK-ASM-AND-OBJ: fli.h ft1, 4.0
# CHECK-ASM: encoding: [0xd3,0x80,0x1b,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 4.000000e+00

# CHECK-ASM-AND-OBJ: fli.h ft1, 4.0
# CHECK-ASM: encoding: [0xd3,0x80,0x1b,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 0x1p+2

# CHECK-ASM-AND-OBJ: fli.h ft1, 8.0
# CHECK-ASM: encoding: [0xd3,0x00,0x1c,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 8.000000e+00

# CHECK-ASM-AND-OBJ: fli.h ft1, 8.0
# CHECK-ASM: encoding: [0xd3,0x00,0x1c,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 0x1p+3

# CHECK-ASM-AND-OBJ: fli.h ft1, 16.0
# CHECK-ASM: encoding: [0xd3,0x80,0x1c,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 1.600000e+01

# CHECK-ASM-AND-OBJ: fli.h ft1, 16.0
# CHECK-ASM: encoding: [0xd3,0x80,0x1c,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 0x1p+4

# CHECK-ASM-AND-OBJ: fli.h ft1, 128.0
# CHECK-ASM: encoding: [0xd3,0x00,0x1d,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 1.280000e+02

# CHECK-ASM-AND-OBJ: fli.h ft1, 128.0
# CHECK-ASM: encoding: [0xd3,0x00,0x1d,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 0x1p+7

# CHECK-ASM-AND-OBJ: fli.h ft1, 256.0
# CHECK-ASM: encoding: [0xd3,0x80,0x1d,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 2.560000e+02

# CHECK-ASM-AND-OBJ: fli.h ft1, 256.0
# CHECK-ASM: encoding: [0xd3,0x80,0x1d,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 0x1p+8

# CHECK-ASM-AND-OBJ: fli.h ft1, 32768.0
# CHECK-ASM: encoding: [0xd3,0x00,0x1e,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 3.276800e+04

# CHECK-ASM-AND-OBJ: fli.h ft1, 32768.0
# CHECK-ASM: encoding: [0xd3,0x00,0x1e,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 0x1p+15

# CHECK-ASM-AND-OBJ: fli.h ft1, 65536.0
# CHECK-ASM: encoding: [0xd3,0x80,0x1e,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 6.553600e+04

# CHECK-ASM-AND-OBJ: fli.h ft1, 65536.0
# CHECK-ASM: encoding: [0xd3,0x80,0x1e,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 0x1p+16

# CHECK-ASM-AND-OBJ: fli.h ft1, inf
# CHECK-ASM: encoding: [0xd3,0x00,0x1f,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, INF

# CHECK-ASM-AND-OBJ: fli.h ft1, nan
# CHECK-ASM: encoding: [0xd3,0x80,0x1f,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, nan

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
