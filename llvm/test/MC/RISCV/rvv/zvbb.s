# RUN: llvm-mc -triple=riscv32 -show-encoding --mattr=+zve32x --mattr=+zvbb %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv32 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+zve32x --mattr=+zvbb %s \
# RUN:        | llvm-objdump -d --mattr=+zve32x --mattr=+zvbb --no-print-imm-hex  - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+zve32x --mattr=+zvbb %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vbrev.v v10, v9, v0.t
# CHECK-INST: vbrev.v v10, v9, v0.t
# CHECK-ENCODING: [0x57,0x25,0x95,0x48]
# CHECK-ERROR: instruction requires the following: 'Zvbb' (Vector basic bit-manipulation instructions){{$}}
# CHECK-UNKNOWN: 57 25 95 48   <unknown>

vclz.v v10, v9, v0.t
# CHECK-INST: vclz.v v10, v9, v0.t
# CHECK-ENCODING: [0x57,0x25,0x96,0x48]
# CHECK-ERROR: instruction requires the following: 'Zvbb' (Vector basic bit-manipulation instructions){{$}}
# CHECK-UNKNOWN: 57 25 96 48   <unknown>

vcpop.v v10, v9, v0.t
# CHECK-INST: vcpop.v v10, v9, v0.t
# CHECK-ENCODING: [0x57,0x25,0x97,0x48]
# CHECK-ERROR: instruction requires the following: 'Zvbb' (Vector basic bit-manipulation instructions){{$}}
# CHECK-UNKNOWN: 57 25 97 48   <unknown>

vctz.v v10, v9, v0.t
# CHECK-INST: vctz.v v10, v9, v0.t
# CHECK-ENCODING: [0x57,0xa5,0x96,0x48]
# CHECK-ERROR: instruction requires the following: 'Zvbb' (Vector basic bit-manipulation instructions){{$}}
# CHECK-UNKNOWN: 57 a5 96 48   <unknown>

vwsll.vv v10, v9, v8, v0.t
# CHECK-INST: vwsll.vv v10, v9, v8, v0.t
# CHECK-ENCODING: [0x57,0x05,0x94,0xd4]
# CHECK-ERROR: instruction requires the following: 'Zvbb' (Vector basic bit-manipulation instructions){{$}}
# CHECK-UNKNOWN: 57 05 94 d4   <unknown>

vwsll.vx v10, v9, a0, v0.t
# CHECK-INST: vwsll.vx v10, v9, a0, v0.t
# CHECK-ENCODING: [0x57,0x45,0x95,0xd4]
# CHECK-ERROR: instruction requires the following: 'Zvbb' (Vector basic bit-manipulation instructions){{$}}
# CHECK-UNKNOWN: 57 45 95 d4   <unknown>

vwsll.vi v10, v9, 29, v0.t
# CHECK-INST: vwsll.vi v10, v9, 29, v0.t
# CHECK-ENCODING: [0x57,0xb5,0x9e,0xd4]
# CHECK-ERROR: instruction requires the following: 'Zvbb' (Vector basic bit-manipulation instructions){{$}}
# CHECK-UNKNOWN: 57 b5 9e d4   <unknown>
