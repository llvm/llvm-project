// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+streaming-sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

cdot  z0.s, z1.b, z31.b, #0
// CHECK-INST: cdot	z0.s, z1.b, z31.b, #0
// CHECK-ENCODING: [0x20,0x10,0x9f,0x44]
// CHECK-ERROR: instruction requires: streaming-sve or sve2
// CHECK-UNKNOWN: 20 10 9f 44 <unknown>

cdot  z0.d, z1.h, z31.h, #0
// CHECK-INST: cdot	z0.d, z1.h, z31.h, #0
// CHECK-ENCODING: [0x20,0x10,0xdf,0x44]
// CHECK-ERROR: instruction requires: streaming-sve or sve2
// CHECK-UNKNOWN: 20 10 df 44 <unknown>

cdot  z0.d, z1.h, z31.h, #90
// CHECK-INST: cdot	z0.d, z1.h, z31.h, #90
// CHECK-ENCODING: [0x20,0x14,0xdf,0x44]
// CHECK-ERROR: instruction requires: streaming-sve or sve2
// CHECK-UNKNOWN: 20 14 df 44 <unknown>

cdot  z0.d, z1.h, z31.h, #180
// CHECK-INST: cdot	z0.d, z1.h, z31.h, #180
// CHECK-ENCODING: [0x20,0x18,0xdf,0x44]
// CHECK-ERROR: instruction requires: streaming-sve or sve2
// CHECK-UNKNOWN: 20 18 df 44 <unknown>

cdot  z0.d, z1.h, z31.h, #270
// CHECK-INST: cdot	z0.d, z1.h, z31.h, #270
// CHECK-ENCODING: [0x20,0x1c,0xdf,0x44]
// CHECK-ERROR: instruction requires: streaming-sve or sve2
// CHECK-UNKNOWN: 20 1c df 44 <unknown>

cdot  z0.s, z1.b, z7.b[3], #0
// CHECK-INST: cdot	z0.s, z1.b, z7.b[3], #0
// CHECK-ENCODING: [0x20,0x40,0xbf,0x44]
// CHECK-ERROR: instruction requires: streaming-sve or sve2
// CHECK-UNKNOWN: 20 40 bf 44 <unknown>

cdot  z0.d, z1.h, z15.h[1], #0
// CHECK-INST: cdot	z0.d, z1.h, z15.h[1], #0
// CHECK-ENCODING: [0x20,0x40,0xff,0x44]
// CHECK-ERROR: instruction requires: streaming-sve or sve2
// CHECK-UNKNOWN: 20 40 ff 44 <unknown>

cdot  z5.d, z6.h, z3.h[0], #90
// CHECK-INST: cdot	z5.d, z6.h, z3.h[0], #90
// CHECK-ENCODING: [0xc5,0x44,0xe3,0x44]
// CHECK-ERROR: instruction requires: streaming-sve or sve2
// CHECK-UNKNOWN: c5 44 e3 44 <unknown>

cdot  z29.d, z30.h, z0.h[0], #180
// CHECK-INST: cdot z29.d, z30.h, z0.h[0], #180
// CHECK-ENCODING: [0xdd,0x4b,0xe0,0x44]
// CHECK-ERROR: instruction requires: streaming-sve or sve2
// CHECK-UNKNOWN: dd 4b e0 44 <unknown>

cdot  z31.d, z30.h, z7.h[1], #270
// CHECK-INST: cdot z31.d, z30.h, z7.h[1], #270
// CHECK-ENCODING: [0xdf,0x4f,0xf7,0x44]
// CHECK-ERROR: instruction requires: streaming-sve or sve2
// CHECK-UNKNOWN: df 4f f7 44 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 bc 20 04 <unknown>

cdot  z0.d, z1.h, z31.h, #0
// CHECK-INST: cdot	z0.d, z1.h, z31.h, #0
// CHECK-ENCODING: [0x20,0x10,0xdf,0x44]
// CHECK-ERROR: instruction requires: streaming-sve or sve2
// CHECK-UNKNOWN: 20 10 df 44 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 bc 20 04 <unknown>

cdot  z0.d, z1.h, z15.h[1], #0
// CHECK-INST: cdot z0.d, z1.h, z15.h[1], #0
// CHECK-ENCODING: [0x20,0x40,0xff,0x44]
// CHECK-ERROR: instruction requires: streaming-sve or sve2
// CHECK-UNKNOWN: 20 40 ff 44 <unknown>
