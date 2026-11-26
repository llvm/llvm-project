# RUN: llvm-mc -triple=riscv32 -show-encoding --mattr=+zve32x --mattr=+zvbb %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv32 -show-encoding %s
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+zve32x --mattr=+zvbb %s \
# RUN:        | llvm-objdump -d --mattr=+zve32x --mattr=+zvbb --no-print-imm-hex  - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

vbrev.v v10, v9, v0.t # expected-error {{instruction requires the following: 'Zvbb' (Vector basic bit-manipulation instructions)}}
# CHECK-INST: vbrev.v v10, v9, v0.t
# CHECK-ENCODING: [0x57,0x25,0x95,0x48]

vclz.v v10, v9, v0.t # expected-error {{instruction requires the following: 'Zvbb' (Vector basic bit-manipulation instructions)}}
# CHECK-INST: vclz.v v10, v9, v0.t
# CHECK-ENCODING: [0x57,0x25,0x96,0x48]

vcpop.v v10, v9, v0.t # expected-error {{instruction requires the following: 'Zvbb' (Vector basic bit-manipulation instructions)}}
# CHECK-INST: vcpop.v v10, v9, v0.t
# CHECK-ENCODING: [0x57,0x25,0x97,0x48]

vctz.v v10, v9, v0.t # expected-error {{instruction requires the following: 'Zvbb' (Vector basic bit-manipulation instructions)}}
# CHECK-INST: vctz.v v10, v9, v0.t
# CHECK-ENCODING: [0x57,0xa5,0x96,0x48]

vwsll.vv v10, v9, v8, v0.t # expected-error {{instruction requires the following: 'Zvbb' (Vector basic bit-manipulation instructions)}}
# CHECK-INST: vwsll.vv v10, v9, v8, v0.t
# CHECK-ENCODING: [0x57,0x05,0x94,0xd4]

vwsll.vx v10, v9, a0, v0.t # expected-error {{instruction requires the following: 'Zvbb' (Vector basic bit-manipulation instructions)}}
# CHECK-INST: vwsll.vx v10, v9, a0, v0.t
# CHECK-ENCODING: [0x57,0x45,0x95,0xd4]

vwsll.vi v10, v9, 29, v0.t # expected-error {{instruction requires the following: 'Zvbb' (Vector basic bit-manipulation instructions)}}
# CHECK-INST: vwsll.vi v10, v9, 29, v0.t
# CHECK-ENCODING: [0x57,0xb5,0x9e,0xd4]
