// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:   | llvm-objdump -d --mattr=-sve2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN

sqshl z0.b, p0/m, z0.b, z1.b
// CHECK-INST: sqshl z0.b, p0/m, z0.b, z1.b
// CHECK-ENCODING: [0x20,0x80,0x08,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44088020 <unknown>

sqshl z0.h, p0/m, z0.h, z1.h
// CHECK-INST: sqshl z0.h, p0/m, z0.h, z1.h
// CHECK-ENCODING: [0x20,0x80,0x48,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44488020 <unknown>

sqshl z29.s, p7/m, z29.s, z30.s
// CHECK-INST: sqshl z29.s, p7/m, z29.s, z30.s
// CHECK-ENCODING: [0xdd,0x9f,0x88,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44889fdd <unknown>

sqshl z31.d, p7/m, z31.d, z30.d
// CHECK-INST: sqshl z31.d, p7/m, z31.d, z30.d
// CHECK-ENCODING: [0xdf,0x9f,0xc8,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44c89fdf <unknown>

sqshl z0.b, p0/m, z0.b, #0
// CHECK-INST: sqshl z0.b, p0/m, z0.b, #0
// CHECK-ENCODING: [0x00,0x81,0x06,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 04068100 <unknown>

sqshl z31.b, p0/m, z31.b, #7
// CHECK-INST: sqshl z31.b, p0/m, z31.b, #7
// CHECK-ENCODING: [0xff,0x81,0x06,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 040681ff <unknown>

sqshl z0.h, p0/m, z0.h, #0
// CHECK-INST: sqshl z0.h, p0/m, z0.h, #0
// CHECK-ENCODING: [0x00,0x82,0x06,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 04068200 <unknown>

sqshl z31.h, p0/m, z31.h, #15
// CHECK-INST: sqshl z31.h, p0/m, z31.h, #15
// CHECK-ENCODING: [0xff,0x83,0x06,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 040683ff <unknown>

sqshl z0.s, p0/m, z0.s, #0
// CHECK-INST: sqshl z0.s, p0/m, z0.s, #0
// CHECK-ENCODING: [0x00,0x80,0x46,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 04468000 <unknown>

sqshl z31.s, p0/m, z31.s, #31
// CHECK-INST: sqshl z31.s, p0/m, z31.s, #31
// CHECK-ENCODING: [0xff,0x83,0x46,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 044683ff <unknown>

sqshl z0.d, p0/m, z0.d, #0
// CHECK-INST: sqshl z0.d, p0/m, z0.d, #0
// CHECK-ENCODING: [0x00,0x80,0x86,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 04868000 <unknown>

sqshl z31.d, p0/m, z31.d, #63
// CHECK-INST: sqshl z31.d, p0/m, z31.d, #63
// CHECK-ENCODING: [0xff,0x83,0xc6,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 04c683ff <unknown>

// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z31.d, p0/z, z6.d
// CHECK-INST: movprfx z31.d, p0/z, z6.d
// CHECK-ENCODING: [0xdf,0x20,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04d020df <unknown>

sqshl z31.d, p0/m, z31.d, z30.d
// CHECK-INST: sqshl z31.d, p0/m, z31.d, z30.d
// CHECK-ENCODING: [0xdf,0x83,0xc8,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44c883df <unknown>

movprfx z31, z6
// CHECK-INST: movprfx z31, z6
// CHECK-ENCODING: [0xdf,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bcdf <unknown>

sqshl z31.d, p7/m, z31.d, z30.d
// CHECK-INST: sqshl z31.d, p7/m, z31.d, z30.d
// CHECK-ENCODING: [0xdf,0x9f,0xc8,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44c89fdf <unknown>

movprfx z31.d, p0/z, z6.d
// CHECK-INST: movprfx z31.d, p0/z, z6.d
// CHECK-ENCODING: [0xdf,0x20,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04d020df <unknown>

sqshl z31.d, p0/m, z31.d, #63
// CHECK-INST: sqshl z31.d, p0/m, z31.d, #63
// CHECK-ENCODING: [0xff,0x83,0xc6,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 04c683ff <unknown>

movprfx z31, z6
// CHECK-INST: movprfx z31, z6
// CHECK-ENCODING: [0xdf,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bcdf <unknown>

sqshl z31.d, p0/m, z31.d, #63
// CHECK-INST: sqshl z31.d, p0/m, z31.d, #63
// CHECK-ENCODING: [0xff,0x83,0xc6,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 04c683ff <unknown>
