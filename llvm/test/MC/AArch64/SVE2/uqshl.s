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

uqshl z0.b, p0/m, z0.b, z1.b
// CHECK-INST: uqshl z0.b, p0/m, z0.b, z1.b
// CHECK-ENCODING: [0x20,0x80,0x09,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 20 80 09 44 <unknown>

uqshl z0.h, p0/m, z0.h, z1.h
// CHECK-INST: uqshl z0.h, p0/m, z0.h, z1.h
// CHECK-ENCODING: [0x20,0x80,0x49,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 20 80 49 44 <unknown>

uqshl z29.s, p7/m, z29.s, z30.s
// CHECK-INST: uqshl z29.s, p7/m, z29.s, z30.s
// CHECK-ENCODING: [0xdd,0x9f,0x89,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: dd 9f 89 44 <unknown>

uqshl z31.d, p7/m, z31.d, z30.d
// CHECK-INST: uqshl z31.d, p7/m, z31.d, z30.d
// CHECK-ENCODING: [0xdf,0x9f,0xc9,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: df 9f c9 44 <unknown>

uqshl z0.b, p0/m, z0.b, #0
// CHECK-INST: uqshl z0.b, p0/m, z0.b, #0
// CHECK-ENCODING: [0x00,0x81,0x07,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 00 81 07 04 <unknown>

uqshl z31.b, p0/m, z31.b, #7
// CHECK-INST: uqshl z31.b, p0/m, z31.b, #7
// CHECK-ENCODING: [0xff,0x81,0x07,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff 81 07 04 <unknown>

uqshl z0.h, p0/m, z0.h, #0
// CHECK-INST: uqshl z0.h, p0/m, z0.h, #0
// CHECK-ENCODING: [0x00,0x82,0x07,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 00 82 07 04 <unknown>

uqshl z31.h, p0/m, z31.h, #15
// CHECK-INST: uqshl z31.h, p0/m, z31.h, #15
// CHECK-ENCODING: [0xff,0x83,0x07,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff 83 07 04 <unknown>

uqshl z0.s, p0/m, z0.s, #0
// CHECK-INST: uqshl z0.s, p0/m, z0.s, #0
// CHECK-ENCODING: [0x00,0x80,0x47,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 00 80 47 04 <unknown>

uqshl z31.s, p0/m, z31.s, #31
// CHECK-INST: uqshl z31.s, p0/m, z31.s, #31
// CHECK-ENCODING: [0xff,0x83,0x47,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff 83 47 04 <unknown>

uqshl z0.d, p0/m, z0.d, #0
// CHECK-INST: uqshl z0.d, p0/m, z0.d, #0
// CHECK-ENCODING: [0x00,0x80,0x87,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 00 80 87 04 <unknown>

uqshl z31.d, p0/m, z31.d, #63
// CHECK-INST: uqshl z31.d, p0/m, z31.d, #63
// CHECK-ENCODING: [0xff,0x83,0xc7,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff 83 c7 04 <unknown>

// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z31.d, p0/z, z6.d
// CHECK-INST: movprfx z31.d, p0/z, z6.d
// CHECK-ENCODING: [0xdf,0x20,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: df 20 d0 04 <unknown>

uqshl z31.d, p0/m, z31.d, z30.d
// CHECK-INST: uqshl z31.d, p0/m, z31.d, z30.d
// CHECK-ENCODING: [0xdf,0x83,0xc9,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: df 83 c9 44 <unknown>

movprfx z31, z6
// CHECK-INST: movprfx z31, z6
// CHECK-ENCODING: [0xdf,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: df bc 20 04 <unknown>

uqshl z31.d, p7/m, z31.d, z30.d
// CHECK-INST: uqshl z31.d, p7/m, z31.d, z30.d
// CHECK-ENCODING: [0xdf,0x9f,0xc9,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: df 9f c9 44 <unknown>

movprfx z31.d, p0/z, z6.d
// CHECK-INST: movprfx z31.d, p0/z, z6.d
// CHECK-ENCODING: [0xdf,0x20,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: df 20 d0 04 <unknown>

uqshl z31.d, p0/m, z31.d, #63
// CHECK-INST: uqshl z31.d, p0/m, z31.d, #63
// CHECK-ENCODING: [0xff,0x83,0xc7,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff 83 c7 04 <unknown>

movprfx z31, z6
// CHECK-INST: movprfx z31, z6
// CHECK-ENCODING: [0xdf,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: df bc 20 04 <unknown>

uqshl z31.d, p0/m, z31.d, #63
// CHECK-INST: uqshl z31.d, p0/m, z31.d, #63
// CHECK-ENCODING: [0xff,0x83,0xc7,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff 83 c7 04 <unknown>
