# RUN: llvm-mc -triple=riscv32 -show-encoding --mattr=+zve32x --mattr=+zvkb %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv32 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+zve32x --mattr=+zvkb %s \
# RUN:        | llvm-objdump --no-print-imm-hex -d --mattr=+zve32x --mattr=+zvkb  - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+zve32x --mattr=+zvkb %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vandn.vv v10, v9, v8, v0.t
# CHECK-INST: vandn.vv v10, v9, v8, v0.t
# CHECK-ENCODING: [0x57,0x05,0x94,0x04]
# CHECK-ERROR: instruction requires the following: 'Zvkb' (Vector Bit-manipulation used in Cryptography){{$}}
# CHECK-UNKNOWN: 57 05 94 04   <unknown>

vandn.vx v10, v9, a0, v0.t
# CHECK-INST: vandn.vx v10, v9, a0, v0.t
# CHECK-ENCODING: [0x57,0x45,0x95,0x04]
# CHECK-ERROR: instruction requires the following: 'Zvkb' (Vector Bit-manipulation used in Cryptography){{$}}
# CHECK-UNKNOWN: 57 45 95 04   <unknown>

vbrev8.v v10, v9, v0.t
# CHECK-INST: vbrev8.v v10, v9, v0.t
# CHECK-ENCODING: [0x57,0x25,0x94,0x48]
# CHECK-ERROR: instruction requires the following: 'Zvkb' (Vector Bit-manipulation used in Cryptography){{$}}
# CHECK-UNKNOWN: 57 25 94 48   <unknown>

vrev8.v v10, v9, v0.t
# CHECK-INST: vrev8.v v10, v9, v0.t
# CHECK-ENCODING: [0x57,0xa5,0x94,0x48]
# CHECK-ERROR: instruction requires the following: 'Zvkb' (Vector Bit-manipulation used in Cryptography){{$}}
# CHECK-UNKNOWN: 57 a5 94 48   <unknown>

vrol.vv v10, v9, v8, v0.t
# CHECK-INST: vrol.vv v10, v9, v8, v0.t
# CHECK-ENCODING: [0x57,0x05,0x94,0x54]
# CHECK-ERROR: instruction requires the following: 'Zvkb' (Vector Bit-manipulation used in Cryptography){{$}}
# CHECK-UNKNOWN: 57 05 94 54   <unknown>

vrol.vx v10, v9, a0, v0.t
# CHECK-INST: vrol.vx v10, v9, a0, v0.t
# CHECK-ENCODING: [0x57,0x45,0x95,0x54]
# CHECK-ERROR: instruction requires the following: 'Zvkb' (Vector Bit-manipulation used in Cryptography){{$}}
# CHECK-UNKNOWN: 57 45 95 54   <unknown>

vror.vv v10, v9, v8, v0.t
# CHECK-INST: vror.vv v10, v9, v8, v0.t
# CHECK-ENCODING: [0x57,0x05,0x94,0x50]
# CHECK-ERROR: instruction requires the following: 'Zvkb' (Vector Bit-manipulation used in Cryptography){{$}}
# CHECK-UNKNOWN: 57 05 94 50   <unknown>

vror.vx v10, v9, a0, v0.t
# CHECK-INST: vror.vx v10, v9, a0, v0.t
# CHECK-ENCODING: [0x57,0x45,0x95,0x50]
# CHECK-ERROR: instruction requires the following: 'Zvkb' (Vector Bit-manipulation used in Cryptography){{$}}
# CHECK-UNKNOWN: 57 45 95 50   <unknown>

vror.vi v10, v9, 33, v0.t
# CHECK-INST: vror.vi v10, v9, 33, v0.t
# CHECK-ENCODING: [0x57,0xb5,0x90,0x54]
# CHECK-ERROR: instruction requires the following: 'Zvkb' (Vector Bit-manipulation used in Cryptography){{$}}
# CHECK-UNKNOWN: 57 b5 90 54   <unknown>
