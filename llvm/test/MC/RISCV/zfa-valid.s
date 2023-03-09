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

# CHECK-ASM-AND-OBJ: fli.s ft1, -1.000000e+00
# CHECK-ASM: encoding: [0xd3,0x00,0x10,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, -1.000000e+00

# CHECK-ASM-AND-OBJ: fli.s ft1, min
# CHECK-ASM: encoding: [0xd3,0x80,0x10,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 1.1754943508222875079687365372222456778186655567720875215087517062784172594547271728515625e-38

# CHECK-ASM-AND-OBJ: fli.s ft1, min
# CHECK-ASM: encoding: [0xd3,0x80,0x10,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, min

# CHECK-ASM-AND-OBJ: fli.s ft1, 1.525879e-05
# CHECK-ASM: encoding: [0xd3,0x00,0x11,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 1.525879e-05

# CHECK-ASM-AND-OBJ: fli.s ft1, 3.051758e-05
# CHECK-ASM: encoding: [0xd3,0x80,0x11,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 3.051758e-05

# CHECK-ASM-AND-OBJ: fli.s ft1, 3.906250e-03
# CHECK-ASM: encoding: [0xd3,0x00,0x12,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 3.906250e-03

# CHECK-ASM-AND-OBJ: fli.s ft1, 7.812500e-03
# CHECK-ASM: encoding: [0xd3,0x80,0x12,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 7.812500e-03

# CHECK-ASM-AND-OBJ: fli.s ft1, 6.250000e-02
# CHECK-ASM: encoding: [0xd3,0x00,0x13,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 6.250000e-02

# CHECK-ASM-AND-OBJ: fli.s ft1, 1.250000e-01
# CHECK-ASM: encoding: [0xd3,0x80,0x13,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 1.250000e-01

# CHECK-ASM-AND-OBJ: fli.s ft1, 2.500000e-01
# CHECK-ASM: encoding: [0xd3,0x00,0x14,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 2.500000e-01

# CHECK-ASM-AND-OBJ: fli.s ft1, 3.125000e-01
# CHECK-ASM: encoding: [0xd3,0x80,0x14,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 3.125000e-01

# CHECK-ASM-AND-OBJ: fli.s ft1, 3.750000e-01
# CHECK-ASM: encoding: [0xd3,0x00,0x15,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 3.750000e-01

# CHECK-ASM-AND-OBJ: fli.s ft1, 4.375000e-01
# CHECK-ASM: encoding: [0xd3,0x80,0x15,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 4.375000e-01

# CHECK-ASM-AND-OBJ: fli.s ft1, 5.000000e-01
# CHECK-ASM: encoding: [0xd3,0x00,0x16,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 5.000000e-01

# CHECK-ASM-AND-OBJ: fli.s ft1, 6.250000e-01
# CHECK-ASM: encoding: [0xd3,0x80,0x16,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 6.250000e-01

# CHECK-ASM-AND-OBJ: fli.s ft1, 7.500000e-01
# CHECK-ASM: encoding: [0xd3,0x00,0x17,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 7.500000e-01

# CHECK-ASM-AND-OBJ: fli.s ft1, 8.750000e-01
# CHECK-ASM: encoding: [0xd3,0x80,0x17,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 8.750000e-01

# CHECK-ASM-AND-OBJ: fli.s ft1, 1.000000e+00
# CHECK-ASM: encoding: [0xd3,0x00,0x18,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 1.000000e+00

# CHECK-ASM-AND-OBJ: fli.s ft1, 1.250000e+00
# CHECK-ASM: encoding: [0xd3,0x80,0x18,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 1.250000e+00

# CHECK-ASM-AND-OBJ: fli.s ft1, 1.500000e+00
# CHECK-ASM: encoding: [0xd3,0x00,0x19,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 1.500000e+00

# CHECK-ASM-AND-OBJ: fli.s ft1, 1.750000e+00
# CHECK-ASM: encoding: [0xd3,0x80,0x19,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 1.750000e+00

# CHECK-ASM-AND-OBJ: fli.s ft1, 2.000000e+00
# CHECK-ASM: encoding: [0xd3,0x00,0x1a,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 2.000000e+00

# CHECK-ASM-AND-OBJ: fli.s ft1, 2.500000e+00
# CHECK-ASM: encoding: [0xd3,0x80,0x1a,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 2.500000e+00

# CHECK-ASM-AND-OBJ: fli.s ft1, 3.000000e+00
# CHECK-ASM: encoding: [0xd3,0x00,0x1b,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 3.000000e+00

# CHECK-ASM-AND-OBJ: fli.s ft1, 4.000000e+00
# CHECK-ASM: encoding: [0xd3,0x80,0x1b,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 4.000000e+00

# CHECK-ASM-AND-OBJ: fli.s ft1, 8.000000e+00
# CHECK-ASM: encoding: [0xd3,0x00,0x1c,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 8.000000e+00

# CHECK-ASM-AND-OBJ: fli.s ft1, 1.600000e+01
# CHECK-ASM: encoding: [0xd3,0x80,0x1c,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 1.600000e+01

# CHECK-ASM-AND-OBJ: fli.s ft1, 1.280000e+02
# CHECK-ASM: encoding: [0xd3,0x00,0x1d,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 1.280000e+02

# CHECK-ASM-AND-OBJ: fli.s ft1, 2.560000e+02
# CHECK-ASM: encoding: [0xd3,0x80,0x1d,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 2.560000e+02

# CHECK-ASM-AND-OBJ: fli.s ft1, 3.276800e+04
# CHECK-ASM: encoding: [0xd3,0x00,0x1e,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 3.276800e+04

# CHECK-ASM-AND-OBJ: fli.s ft1, 6.553600e+04
# CHECK-ASM: encoding: [0xd3,0x80,0x1e,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 6.553600e+04

# CHECK-ASM-AND-OBJ: fli.s ft1, 6.553600e+04
# CHECK-ASM: encoding: [0xd3,0x80,0x1e,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, 29

# CHECK-ASM-AND-OBJ: fli.s ft1, inf
# CHECK-ASM: encoding: [0xd3,0x00,0x1f,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, INF

# CHECK-ASM-AND-OBJ: fli.s ft1, nan
# CHECK-ASM: encoding: [0xd3,0x80,0x1f,0xf0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.s ft1, nan

# CHECK-ASM-AND-OBJ: fli.d ft1, -1.000000e+00
# CHECK-ASM: encoding: [0xd3,0x00,0x10,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, -1.000000e+00

# CHECK-ASM-AND-OBJ: fli.d ft1, min
# CHECK-ASM: encoding: [0xd3,0x80,0x10,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 2.225073858507201383090232717332404064219215980462331830553327416887204434813918195854283159012511020564067339731035811005152434161553460108856012385377718821130777993532002330479610147442583636071921565046942503734208375250806650616658158948720491179968591639648500635908770118304874799780887753749949451580451605050915399856582470818645113537935804992115981085766051992433352114352390148795699609591288891602992641511063466313393663477586513029371762047325631781485664350872122828637642044846811407613911477062801689853244110024161447421618567166150540154285084716752901903161322778896729707373123334086988983175067838846926092773977972858659654941091369095406136467568702398678315290680984617210924625396728515625e-308

# CHECK-ASM-AND-OBJ: fli.d ft1, min
# CHECK-ASM: encoding: [0xd3,0x80,0x10,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, min

# CHECK-ASM-AND-OBJ: fli.d ft1, 1.525879e-05
# CHECK-ASM: encoding: [0xd3,0x00,0x11,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 1.525879e-05

# CHECK-ASM-AND-OBJ: fli.d ft1, 3.051758e-05
# CHECK-ASM: encoding: [0xd3,0x80,0x11,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 3.051758e-05

# CHECK-ASM-AND-OBJ: fli.d ft1, 3.906250e-03
# CHECK-ASM: encoding: [0xd3,0x00,0x12,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 3.906250e-03

# CHECK-ASM-AND-OBJ: fli.d ft1, 7.812500e-03
# CHECK-ASM: encoding: [0xd3,0x80,0x12,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 7.812500e-03

# CHECK-ASM-AND-OBJ: fli.d ft1, 6.250000e-02
# CHECK-ASM: encoding: [0xd3,0x00,0x13,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 6.250000e-02

# CHECK-ASM-AND-OBJ: fli.d ft1, 1.250000e-01
# CHECK-ASM: encoding: [0xd3,0x80,0x13,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 1.250000e-01

# CHECK-ASM-AND-OBJ: fli.d ft1, 2.500000e-01
# CHECK-ASM: encoding: [0xd3,0x00,0x14,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 2.500000e-01

# CHECK-ASM-AND-OBJ: fli.d ft1, 2.500000e-01
# CHECK-ASM: encoding: [0xd3,0x00,0x14,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 8

# CHECK-ASM-AND-OBJ: fli.d ft1, 3.125000e-01
# CHECK-ASM: encoding: [0xd3,0x80,0x14,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 3.125000e-01

# CHECK-ASM-AND-OBJ: fli.d ft1, 3.750000e-01
# CHECK-ASM: encoding: [0xd3,0x00,0x15,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 3.750000e-01

# CHECK-ASM-AND-OBJ: fli.d ft1, 4.375000e-01
# CHECK-ASM: encoding: [0xd3,0x80,0x15,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 4.375000e-01

# CHECK-ASM-AND-OBJ: fli.d ft1, 5.000000e-01
# CHECK-ASM: encoding: [0xd3,0x00,0x16,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 5.000000e-01

# CHECK-ASM-AND-OBJ: fli.d ft1, 6.250000e-01
# CHECK-ASM: encoding: [0xd3,0x80,0x16,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 6.250000e-01

# CHECK-ASM-AND-OBJ: fli.d ft1, 7.500000e-01
# CHECK-ASM: encoding: [0xd3,0x00,0x17,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 7.500000e-01

# CHECK-ASM-AND-OBJ: fli.d ft1, 8.750000e-01
# CHECK-ASM: encoding: [0xd3,0x80,0x17,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 8.750000e-01

# CHECK-ASM-AND-OBJ: fli.d ft1, 1.000000e+00
# CHECK-ASM: encoding: [0xd3,0x00,0x18,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 1.000000e+00

# CHECK-ASM-AND-OBJ: fli.d ft1, 1.250000e+00
# CHECK-ASM: encoding: [0xd3,0x80,0x18,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 1.250000e+00

# CHECK-ASM-AND-OBJ: fli.d ft1, 1.500000e+00
# CHECK-ASM: encoding: [0xd3,0x00,0x19,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 1.500000e+00

# CHECK-ASM-AND-OBJ: fli.d ft1, 1.750000e+00
# CHECK-ASM: encoding: [0xd3,0x80,0x19,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 1.750000e+00

# CHECK-ASM-AND-OBJ: fli.d ft1, 2.000000e+00
# CHECK-ASM: encoding: [0xd3,0x00,0x1a,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 2.000000e+00

# CHECK-ASM-AND-OBJ: fli.d ft1, 2.500000e+00
# CHECK-ASM: encoding: [0xd3,0x80,0x1a,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 2.500000e+00

# CHECK-ASM-AND-OBJ: fli.d ft1, 3.000000e+00
# CHECK-ASM: encoding: [0xd3,0x00,0x1b,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 3.000000e+00

# CHECK-ASM-AND-OBJ: fli.d ft1, 4.000000e+00
# CHECK-ASM: encoding: [0xd3,0x80,0x1b,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 4.000000e+00

# CHECK-ASM-AND-OBJ: fli.d ft1, 8.000000e+00
# CHECK-ASM: encoding: [0xd3,0x00,0x1c,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 8.000000e+00

# CHECK-ASM-AND-OBJ: fli.d ft1, 1.600000e+01
# CHECK-ASM: encoding: [0xd3,0x80,0x1c,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 1.600000e+01

# CHECK-ASM-AND-OBJ: fli.d ft1, 1.280000e+02
# CHECK-ASM: encoding: [0xd3,0x00,0x1d,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 1.280000e+02

# CHECK-ASM-AND-OBJ: fli.d ft1, 2.560000e+02
# CHECK-ASM: encoding: [0xd3,0x80,0x1d,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 2.560000e+02

# CHECK-ASM-AND-OBJ: fli.d ft1, 3.276800e+04
# CHECK-ASM: encoding: [0xd3,0x00,0x1e,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 3.276800e+04

# CHECK-ASM-AND-OBJ: fli.d ft1, 6.553600e+04
# CHECK-ASM: encoding: [0xd3,0x80,0x1e,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 6.553600e+04

# CHECK-ASM-AND-OBJ: fli.d ft1, 6.553600e+04
# CHECK-ASM: encoding: [0xd3,0x80,0x1e,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, 29

# CHECK-ASM-AND-OBJ: fli.d ft1, inf
# CHECK-ASM: encoding: [0xd3,0x00,0x1f,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, INF

# CHECK-ASM-AND-OBJ: fli.d ft1, nan
# CHECK-ASM: encoding: [0xd3,0x80,0x1f,0xf2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.d ft1, nan

# CHECK-ASM-AND-OBJ: fli.h ft1, -1.000000e+00
# CHECK-ASM: encoding: [0xd3,0x00,0x10,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, -1.000000e+00

# CHECK-ASM-AND-OBJ: fli.h ft1, min
# CHECK-ASM: encoding: [0xd3,0x80,0x10,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 6.103516e-05

# CHECK-ASM-AND-OBJ: fli.h ft1, min
# CHECK-ASM: encoding: [0xd3,0x80,0x10,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, min

# CHECK-ASM-AND-OBJ: fli.h ft1, 1.525879e-05
# CHECK-ASM: encoding: [0xd3,0x00,0x11,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 1.525879e-05

# CHECK-ASM-AND-OBJ: fli.h ft1, 3.051758e-05
# CHECK-ASM: encoding: [0xd3,0x80,0x11,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 3.051758e-05

# CHECK-ASM-AND-OBJ: fli.h ft1, 3.906250e-03
# CHECK-ASM: encoding: [0xd3,0x00,0x12,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 3.906250e-03

# CHECK-ASM-AND-OBJ: fli.h ft1, 7.812500e-03
# CHECK-ASM: encoding: [0xd3,0x80,0x12,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 7.812500e-03

# CHECK-ASM-AND-OBJ: fli.h ft1, 6.250000e-02
# CHECK-ASM: encoding: [0xd3,0x00,0x13,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 6.250000e-02

# CHECK-ASM-AND-OBJ: fli.h ft1, 1.250000e-01
# CHECK-ASM: encoding: [0xd3,0x80,0x13,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 1.250000e-01

# CHECK-ASM-AND-OBJ: fli.h ft1, 2.500000e-01
# CHECK-ASM: encoding: [0xd3,0x00,0x14,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 2.500000e-01

# CHECK-ASM-AND-OBJ: fli.h ft1, 3.125000e-01
# CHECK-ASM: encoding: [0xd3,0x80,0x14,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 3.125000e-01

# CHECK-ASM-AND-OBJ: fli.h ft1, 3.750000e-01
# CHECK-ASM: encoding: [0xd3,0x00,0x15,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 3.750000e-01

# CHECK-ASM-AND-OBJ: fli.h ft1, 4.375000e-01
# CHECK-ASM: encoding: [0xd3,0x80,0x15,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 4.375000e-01

# CHECK-ASM-AND-OBJ: fli.h ft1, 5.000000e-01
# CHECK-ASM: encoding: [0xd3,0x00,0x16,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 5.000000e-01

# CHECK-ASM-AND-OBJ: fli.h ft1, 6.250000e-01
# CHECK-ASM: encoding: [0xd3,0x80,0x16,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 6.250000e-01

# CHECK-ASM-AND-OBJ: fli.h ft1, 7.500000e-01
# CHECK-ASM: encoding: [0xd3,0x00,0x17,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 7.500000e-01

# CHECK-ASM-AND-OBJ: fli.h ft1, 8.750000e-01
# CHECK-ASM: encoding: [0xd3,0x80,0x17,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 8.750000e-01

# CHECK-ASM-AND-OBJ: fli.h ft1, 1.000000e+00
# CHECK-ASM: encoding: [0xd3,0x00,0x18,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 1.000000e+00

# CHECK-ASM-AND-OBJ: fli.h ft1, 1.250000e+00
# CHECK-ASM: encoding: [0xd3,0x80,0x18,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 1.250000e+00

# CHECK-ASM-AND-OBJ: fli.h ft1, 1.500000e+00
# CHECK-ASM: encoding: [0xd3,0x00,0x19,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 1.500000e+00

# CHECK-ASM-AND-OBJ: fli.h ft1, 1.750000e+00
# CHECK-ASM: encoding: [0xd3,0x80,0x19,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 1.750000e+00

# CHECK-ASM-AND-OBJ: fli.h ft1, 2.000000e+00
# CHECK-ASM: encoding: [0xd3,0x00,0x1a,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 2.000000e+00

# CHECK-ASM-AND-OBJ: fli.h ft1, 2.500000e+00
# CHECK-ASM: encoding: [0xd3,0x80,0x1a,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 2.500000e+00

# CHECK-ASM-AND-OBJ: fli.h ft1, 3.000000e+00
# CHECK-ASM: encoding: [0xd3,0x00,0x1b,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 3.000000e+00

# CHECK-ASM-AND-OBJ: fli.h ft1, 4.000000e+00
# CHECK-ASM: encoding: [0xd3,0x80,0x1b,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 4.000000e+00

# CHECK-ASM-AND-OBJ: fli.h ft1, 8.000000e+00
# CHECK-ASM: encoding: [0xd3,0x00,0x1c,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 8.000000e+00

# CHECK-ASM-AND-OBJ: fli.h ft1, 1.600000e+01
# CHECK-ASM: encoding: [0xd3,0x80,0x1c,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 1.600000e+01

# CHECK-ASM-AND-OBJ: fli.h ft1, 1.280000e+02
# CHECK-ASM: encoding: [0xd3,0x00,0x1d,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 1.280000e+02

# CHECK-ASM-AND-OBJ: fli.h ft1, 2.560000e+02
# CHECK-ASM: encoding: [0xd3,0x80,0x1d,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 2.560000e+02

# CHECK-ASM-AND-OBJ: fli.h ft1, 3.276800e+04
# CHECK-ASM: encoding: [0xd3,0x00,0x1e,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 3.276800e+04

# CHECK-ASM-AND-OBJ: fli.h ft1, 6.553600e+04
# CHECK-ASM: encoding: [0xd3,0x80,0x1e,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 6.553600e+04

# CHECK-ASM-AND-OBJ: fli.h ft1, 6.553600e+04
# CHECK-ASM: encoding: [0xd3,0x80,0x1e,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fli.h ft1, 29

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
