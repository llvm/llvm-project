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

sqshl z0.b, p0/m, z0.b, z1.b
// CHECK-INST: sqshl z0.b, p0/m, z0.b, z1.b
// CHECK-ENCODING: [0x20,0x80,0x08,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 20 80 08 44 <unknown>

sqshl z0.h, p0/m, z0.h, z1.h
// CHECK-INST: sqshl z0.h, p0/m, z0.h, z1.h
// CHECK-ENCODING: [0x20,0x80,0x48,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 20 80 48 44 <unknown>

sqshl z29.s, p7/m, z29.s, z30.s
// CHECK-INST: sqshl z29.s, p7/m, z29.s, z30.s
// CHECK-ENCODING: [0xdd,0x9f,0x88,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: dd 9f 88 44 <unknown>

sqshl z31.d, p7/m, z31.d, z30.d
// CHECK-INST: sqshl z31.d, p7/m, z31.d, z30.d
// CHECK-ENCODING: [0xdf,0x9f,0xc8,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: df 9f c8 44 <unknown>

sqshl z0.b, p0/m, z0.b, #0
// CHECK-INST: sqshl z0.b, p0/m, z0.b, #0
// CHECK-ENCODING: [0x00,0x81,0x06,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 00 81 06 04 <unknown>

sqshl z31.b, p0/m, z31.b, #7
// CHECK-INST: sqshl z31.b, p0/m, z31.b, #7
// CHECK-ENCODING: [0xff,0x81,0x06,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff 81 06 04 <unknown>

sqshl z0.h, p0/m, z0.h, #0
// CHECK-INST: sqshl z0.h, p0/m, z0.h, #0
// CHECK-ENCODING: [0x00,0x82,0x06,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 00 82 06 04 <unknown>

sqshl z31.h, p0/m, z31.h, #15
// CHECK-INST: sqshl z31.h, p0/m, z31.h, #15
// CHECK-ENCODING: [0xff,0x83,0x06,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff 83 06 04 <unknown>

sqshl z0.s, p0/m, z0.s, #0
// CHECK-INST: sqshl z0.s, p0/m, z0.s, #0
// CHECK-ENCODING: [0x00,0x80,0x46,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 00 80 46 04 <unknown>

sqshl z31.s, p0/m, z31.s, #31
// CHECK-INST: sqshl z31.s, p0/m, z31.s, #31
// CHECK-ENCODING: [0xff,0x83,0x46,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff 83 46 04 <unknown>

sqshl z0.d, p0/m, z0.d, #0
// CHECK-INST: sqshl z0.d, p0/m, z0.d, #0
// CHECK-ENCODING: [0x00,0x80,0x86,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 00 80 86 04 <unknown>

sqshl z31.d, p0/m, z31.d, #63
// CHECK-INST: sqshl z31.d, p0/m, z31.d, #63
// CHECK-ENCODING: [0xff,0x83,0xc6,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff 83 c6 04 <unknown>

// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z31.d, p0/z, z6.d
// CHECK-INST: movprfx z31.d, p0/z, z6.d
// CHECK-ENCODING: [0xdf,0x20,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: df 20 d0 04 <unknown>

sqshl z31.d, p0/m, z31.d, z30.d
// CHECK-INST: sqshl z31.d, p0/m, z31.d, z30.d
// CHECK-ENCODING: [0xdf,0x83,0xc8,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: df 83 c8 44 <unknown>

movprfx z31, z6
// CHECK-INST: movprfx z31, z6
// CHECK-ENCODING: [0xdf,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: df bc 20 04 <unknown>

sqshl z31.d, p7/m, z31.d, z30.d
// CHECK-INST: sqshl z31.d, p7/m, z31.d, z30.d
// CHECK-ENCODING: [0xdf,0x9f,0xc8,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: df 9f c8 44 <unknown>

movprfx z31.d, p0/z, z6.d
// CHECK-INST: movprfx z31.d, p0/z, z6.d
// CHECK-ENCODING: [0xdf,0x20,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: df 20 d0 04 <unknown>

sqshl z31.d, p0/m, z31.d, #63
// CHECK-INST: sqshl z31.d, p0/m, z31.d, #63
// CHECK-ENCODING: [0xff,0x83,0xc6,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff 83 c6 04 <unknown>

movprfx z31, z6
// CHECK-INST: movprfx z31, z6
// CHECK-ENCODING: [0xdf,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: df bc 20 04 <unknown>

sqshl z31.d, p0/m, z31.d, #63
// CHECK-INST: sqshl z31.d, p0/m, z31.d, #63
// CHECK-ENCODING: [0xff,0x83,0xc6,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff 83 c6 04 <unknown>
