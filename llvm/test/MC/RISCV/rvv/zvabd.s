# RUN: llvm-mc -triple=riscv32 -show-encoding --mattr=+v --mattr=+experimental-zvabd %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv32 -show-encoding %s
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+v --mattr=+experimental-zvabd %s \
# RUN:        | llvm-objdump -d --mattr=+v --mattr=+experimental-zvabd --no-print-imm-hex  - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

vabs.v v9, v8 # expected-error {{instruction requires the following: 'Zvabd' (Vector Absolute Difference)}}
# CHECK-INST: vabs.v v9, v8
# CHECK-ENCODING: [0xd7,0x64,0x80,0x56]

vabs.v v9, v8, v0.t # expected-error {{instruction requires the following: 'Zvabd' (Vector Absolute Difference)}}
# CHECK-INST: vabs.v v9, v8, v0.t
# CHECK-ENCODING: [0xd7,0x64,0x80,0x54]

vabd.vv v10, v9, v8 # expected-error {{instruction requires the following: 'Zvabd' (Vector Absolute Difference)}}
# CHECK-INST: vabd.vv v10, v9, v8
# CHECK-ENCODING: [0x57,0x25,0x94,0x56]

vabd.vv v10, v9, v8, v0.t # expected-error {{instruction requires the following: 'Zvabd' (Vector Absolute Difference)}}
# CHECK-INST: vabd.vv v10, v9, v8, v0.t
# CHECK-ENCODING: [0x57,0x25,0x94,0x54]

vabd.vx v10, v9, a0 # expected-error {{instruction requires the following: 'Zvabd' (Vector Absolute Difference)}}
# CHECK-INST: vabd.vx v10, v9, a0
# CHECK-ENCODING: [0x57,0x65,0x95,0x56]

vabd.vx v10, v9, a0, v0.t # expected-error {{instruction requires the following: 'Zvabd' (Vector Absolute Difference)}}
# CHECK-INST: vabd.vx v10, v9, a0, v0.t
# CHECK-ENCODING: [0x57,0x65,0x95,0x54]

vabdu.vv v10, v9, v8 # expected-error {{instruction requires the following: 'Zvabd' (Vector Absolute Difference)}}
# CHECK-INST: vabdu.vv v10, v9, v8
# CHECK-ENCODING: [0x57,0x25,0x94,0x5a]

vabdu.vv v10, v9, v8, v0.t # expected-error {{instruction requires the following: 'Zvabd' (Vector Absolute Difference)}}
# CHECK-INST: vabdu.vv v10, v9, v8, v0.t
# CHECK-ENCODING: [0x57,0x25,0x94,0x58]

vabdu.vx v10, v9, a0 # expected-error {{instruction requires the following: 'Zvabd' (Vector Absolute Difference)}}
# CHECK-INST: vabdu.vx v10, v9, a0
# CHECK-ENCODING: [0x57,0x65,0x95,0x5a]

vabdu.vx v10, v9, a0, v0.t # expected-error {{instruction requires the following: 'Zvabd' (Vector Absolute Difference)}}
# CHECK-INST: vabdu.vx v10, v9, a0, v0.t
# CHECK-ENCODING: [0x57,0x65,0x95,0x58]

vwabda.vv v10, v9, v8 # expected-error {{instruction requires the following: 'Zvabd' (Vector Absolute Difference)}}
# CHECK-INST: vwabda.vv v10, v9, v8
# CHECK-ENCODING: [0x57,0x05,0x94,0xf6]

vwabda.vv v10, v9, v8, v0.t # expected-error {{instruction requires the following: 'Zvabd' (Vector Absolute Difference)}}
# CHECK-INST: vwabda.vv v10, v9, v8, v0.t
# CHECK-ENCODING: [0x57,0x05,0x94,0xf4]

vwabda.vx v10, v9, a0 # expected-error {{instruction requires the following: 'Zvabd' (Vector Absolute Difference)}}
# CHECK-INST: vwabda.vx v10, v9, a0
# CHECK-ENCODING: [0x57,0x45,0x95,0xf6]

vwabda.vx v10, v9, a0, v0.t # expected-error {{instruction requires the following: 'Zvabd' (Vector Absolute Difference)}}
# CHECK-INST: vwabda.vx v10, v9, a0, v0.t
# CHECK-ENCODING: [0x57,0x45,0x95,0xf4]

vwabdau.vv v10, v9, v8 # expected-error {{instruction requires the following: 'Zvabd' (Vector Absolute Difference)}}
# CHECK-INST: vwabdau.vv v10, v9, v8
# CHECK-ENCODING: [0x57,0x05,0x94,0xfa]

vwabdau.vv v10, v9, v8, v0.t # expected-error {{instruction requires the following: 'Zvabd' (Vector Absolute Difference)}}
# CHECK-INST: vwabdau.vv v10, v9, v8, v0.t
# CHECK-ENCODING: [0x57,0x05,0x94,0xf8]

vwabdau.vx v10, v9, a0 # expected-error {{instruction requires the following: 'Zvabd' (Vector Absolute Difference)}}
# CHECK-INST: vwabdau.vx v10, v9, a0
# CHECK-ENCODING: [0x57,0x45,0x95,0xfa]

vwabdau.vx v10, v9, a0, v0.t # expected-error {{instruction requires the following: 'Zvabd' (Vector Absolute Difference)}}
# CHECK-INST: vwabdau.vx v10, v9, a0, v0.t
# CHECK-ENCODING: [0x57,0x45,0x95,0xf8]
