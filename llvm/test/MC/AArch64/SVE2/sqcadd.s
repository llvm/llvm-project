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

sqcadd   z0.b, z0.b, z0.b, #90
// CHECK-INST: sqcadd   z0.b, z0.b, z0.b, #90
// CHECK-ENCODING: [0x00,0xd8,0x01,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 00 d8 01 45 <unknown>

sqcadd   z0.h, z0.h, z0.h, #90
// CHECK-INST: sqcadd   z0.h, z0.h, z0.h, #90
// CHECK-ENCODING: [0x00,0xd8,0x41,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 00 d8 41 45 <unknown>

sqcadd   z0.s, z0.s, z0.s, #90
// CHECK-INST: sqcadd   z0.s, z0.s, z0.s, #90
// CHECK-ENCODING: [0x00,0xd8,0x81,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 00 d8 81 45 <unknown>

sqcadd   z0.d, z0.d, z0.d, #90
// CHECK-INST: sqcadd   z0.d, z0.d, z0.d, #90
// CHECK-ENCODING: [0x00,0xd8,0xc1,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 00 d8 c1 45 <unknown>

sqcadd   z31.b, z31.b, z31.b, #270
// CHECK-INST: sqcadd   z31.b, z31.b, z31.b, #270
// CHECK-ENCODING: [0xff,0xdf,0x01,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff df 01 45 <unknown>

sqcadd   z31.h, z31.h, z31.h, #270
// CHECK-INST: sqcadd   z31.h, z31.h, z31.h, #270
// CHECK-ENCODING: [0xff,0xdf,0x41,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff df 41 45 <unknown>

sqcadd   z31.s, z31.s, z31.s, #270
// CHECK-INST: sqcadd   z31.s, z31.s, z31.s, #270
// CHECK-ENCODING: [0xff,0xdf,0x81,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff df 81 45 <unknown>

sqcadd   z31.d, z31.d, z31.d, #270
// CHECK-INST: sqcadd   z31.d, z31.d, z31.d, #270
// CHECK-ENCODING: [0xff,0xdf,0xc1,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff df c1 45 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z4, z6
// CHECK-INST: movprfx	z4, z6
// CHECK-ENCODING: [0xc4,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c4 bc 20 04 <unknown>

sqcadd   z4.d, z4.d, z31.d, #270
// CHECK-INST: sqcadd	z4.d, z4.d, z31.d, #270
// CHECK-ENCODING: [0xe4,0xdf,0xc1,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: e4 df c1 45 <unknown>
