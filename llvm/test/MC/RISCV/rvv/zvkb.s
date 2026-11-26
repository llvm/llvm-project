# RUN: llvm-mc -triple=riscv32 -show-encoding --mattr=+zve32x --mattr=+zvkb %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv32 -show-encoding %s
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+zve32x --mattr=+zvkb %s \
# RUN:        | llvm-objdump --no-print-imm-hex -d --mattr=+zve32x --mattr=+zvkb  - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

vandn.vv v10, v9, v8, v0.t # expected-error {{instruction requires the following: 'Zvkb' (Vector Bit-manipulation used in Cryptography)}}
# CHECK-INST: vandn.vv v10, v9, v8, v0.t
# CHECK-ENCODING: [0x57,0x05,0x94,0x04]

vandn.vx v10, v9, a0, v0.t # expected-error {{instruction requires the following: 'Zvkb' (Vector Bit-manipulation used in Cryptography)}}
# CHECK-INST: vandn.vx v10, v9, a0, v0.t
# CHECK-ENCODING: [0x57,0x45,0x95,0x04]

vbrev8.v v10, v9, v0.t # expected-error {{instruction requires the following: 'Zvkb' (Vector Bit-manipulation used in Cryptography)}}
# CHECK-INST: vbrev8.v v10, v9, v0.t
# CHECK-ENCODING: [0x57,0x25,0x94,0x48]

vrev8.v v10, v9, v0.t # expected-error {{instruction requires the following: 'Zvkb' (Vector Bit-manipulation used in Cryptography)}}
# CHECK-INST: vrev8.v v10, v9, v0.t
# CHECK-ENCODING: [0x57,0xa5,0x94,0x48]

vrol.vv v10, v9, v8, v0.t # expected-error {{instruction requires the following: 'Zvkb' (Vector Bit-manipulation used in Cryptography)}}
# CHECK-INST: vrol.vv v10, v9, v8, v0.t
# CHECK-ENCODING: [0x57,0x05,0x94,0x54]

vrol.vx v10, v9, a0, v0.t # expected-error {{instruction requires the following: 'Zvkb' (Vector Bit-manipulation used in Cryptography)}}
# CHECK-INST: vrol.vx v10, v9, a0, v0.t
# CHECK-ENCODING: [0x57,0x45,0x95,0x54]

vror.vv v10, v9, v8, v0.t # expected-error {{instruction requires the following: 'Zvkb' (Vector Bit-manipulation used in Cryptography)}}
# CHECK-INST: vror.vv v10, v9, v8, v0.t
# CHECK-ENCODING: [0x57,0x05,0x94,0x50]

vror.vx v10, v9, a0, v0.t # expected-error {{instruction requires the following: 'Zvkb' (Vector Bit-manipulation used in Cryptography)}}
# CHECK-INST: vror.vx v10, v9, a0, v0.t
# CHECK-ENCODING: [0x57,0x45,0x95,0x50]

vror.vi v10, v9, 33, v0.t # expected-error {{instruction requires the following: 'Zvkb' (Vector Bit-manipulation used in Cryptography)}}
# CHECK-INST: vror.vi v10, v9, 33, v0.t
# CHECK-ENCODING: [0x57,0xb5,0x90,0x54]
