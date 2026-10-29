# RUN: llvm-mc -triple=riscv32 -show-encoding --mattr=+v --mattr=+experimental-zvabd %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv32 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+v --mattr=+experimental-zvabd %s \
# RUN:        | llvm-objdump -d --mattr=+v --mattr=+experimental-zvabd --no-print-imm-hex  - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

vabs.v v9, v8
# CHECK-INST: vabs.v v9, v8
# CHECK-ENCODING: [0xd7,0x64,0x80,0x56]
# CHECK-ERROR: instruction requires the following: 'Zvabd' (Vector Absolute Difference){{$}}

vabs.v v9, v8, v0.t
# CHECK-INST: vabs.v v9, v8, v0.t
# CHECK-ENCODING: [0xd7,0x64,0x80,0x54]
# CHECK-ERROR: instruction requires the following: 'Zvabd' (Vector Absolute Difference){{$}}

vabd.vv v10, v9, v8
# CHECK-INST: vabd.vv v10, v9, v8
# CHECK-ENCODING: [0x57,0x25,0x94,0x56]
# CHECK-ERROR: instruction requires the following: 'Zvabd' (Vector Absolute Difference){{$}}

vabd.vv v10, v9, v8, v0.t
# CHECK-INST: vabd.vv v10, v9, v8, v0.t
# CHECK-ENCODING: [0x57,0x25,0x94,0x54]
# CHECK-ERROR: instruction requires the following: 'Zvabd' (Vector Absolute Difference){{$}}

vabd.vx v10, v9, a0
# CHECK-INST: vabd.vx v10, v9, a0
# CHECK-ENCODING: [0x57,0x65,0x95,0x56]
# CHECK-ERROR: instruction requires the following: 'Zvabd' (Vector Absolute Difference){{$}}

vabd.vx v10, v9, a0, v0.t
# CHECK-INST: vabd.vx v10, v9, a0, v0.t
# CHECK-ENCODING: [0x57,0x65,0x95,0x54]
# CHECK-ERROR: instruction requires the following: 'Zvabd' (Vector Absolute Difference){{$}}

vabdu.vv v10, v9, v8
# CHECK-INST: vabdu.vv v10, v9, v8
# CHECK-ENCODING: [0x57,0x25,0x94,0x5a]
# CHECK-ERROR: instruction requires the following: 'Zvabd' (Vector Absolute Difference){{$}}

vabdu.vv v10, v9, v8, v0.t
# CHECK-INST: vabdu.vv v10, v9, v8, v0.t
# CHECK-ENCODING: [0x57,0x25,0x94,0x58]
# CHECK-ERROR: instruction requires the following: 'Zvabd' (Vector Absolute Difference){{$}}

vabdu.vx v10, v9, a0
# CHECK-INST: vabdu.vx v10, v9, a0
# CHECK-ENCODING: [0x57,0x65,0x95,0x5a]
# CHECK-ERROR: instruction requires the following: 'Zvabd' (Vector Absolute Difference){{$}}

vabdu.vx v10, v9, a0, v0.t
# CHECK-INST: vabdu.vx v10, v9, a0, v0.t
# CHECK-ENCODING: [0x57,0x65,0x95,0x58]
# CHECK-ERROR: instruction requires the following: 'Zvabd' (Vector Absolute Difference){{$}}

vwabda.vv v10, v9, v8
# CHECK-INST: vwabda.vv v10, v9, v8
# CHECK-ENCODING: [0x57,0x05,0x94,0xf6]
# CHECK-ERROR: instruction requires the following: 'Zvabd' (Vector Absolute Difference){{$}}

vwabda.vv v10, v9, v8, v0.t
# CHECK-INST: vwabda.vv v10, v9, v8, v0.t
# CHECK-ENCODING: [0x57,0x05,0x94,0xf4]
# CHECK-ERROR: instruction requires the following: 'Zvabd' (Vector Absolute Difference){{$}}

vwabda.vx v10, v9, a0
# CHECK-INST: vwabda.vx v10, v9, a0
# CHECK-ENCODING: [0x57,0x45,0x95,0xf6]
# CHECK-ERROR: instruction requires the following: 'Zvabd' (Vector Absolute Difference){{$}}

vwabda.vx v10, v9, a0, v0.t
# CHECK-INST: vwabda.vx v10, v9, a0, v0.t
# CHECK-ENCODING: [0x57,0x45,0x95,0xf4]
# CHECK-ERROR: instruction requires the following: 'Zvabd' (Vector Absolute Difference){{$}}

vwabdau.vv v10, v9, v8
# CHECK-INST: vwabdau.vv v10, v9, v8
# CHECK-ENCODING: [0x57,0x05,0x94,0xfa]
# CHECK-ERROR: instruction requires the following: 'Zvabd' (Vector Absolute Difference){{$}}

vwabdau.vv v10, v9, v8, v0.t
# CHECK-INST: vwabdau.vv v10, v9, v8, v0.t
# CHECK-ENCODING: [0x57,0x05,0x94,0xf8]
# CHECK-ERROR: instruction requires the following: 'Zvabd' (Vector Absolute Difference){{$}}

vwabdau.vx v10, v9, a0
# CHECK-INST: vwabdau.vx v10, v9, a0
# CHECK-ENCODING: [0x57,0x45,0x95,0xfa]
# CHECK-ERROR: instruction requires the following: 'Zvabd' (Vector Absolute Difference){{$}}

vwabdau.vx v10, v9, a0, v0.t
# CHECK-INST: vwabdau.vx v10, v9, a0, v0.t
# CHECK-ENCODING: [0x57,0x45,0x95,0xf8]
# CHECK-ERROR: instruction requires the following: 'Zvabd' (Vector Absolute Difference){{$}}
