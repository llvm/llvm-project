// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

fmaxnm  z0.h, p0/m, z0.h, #0.000000000000000
// CHECK-INST: fmaxnm	z0.h, p0/m, z0.h, #0.0
// CHECK-ENCODING: [0x00,0x80,0x5c,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 80 5c 65 <unknown>

fmaxnm  z0.h, p0/m, z0.h, #0.0
// CHECK-INST: fmaxnm	z0.h, p0/m, z0.h, #0.0
// CHECK-ENCODING: [0x00,0x80,0x5c,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 80 5c 65 <unknown>

fmaxnm  z0.s, p0/m, z0.s, #0.0
// CHECK-INST: fmaxnm	z0.s, p0/m, z0.s, #0.0
// CHECK-ENCODING: [0x00,0x80,0x9c,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 80 9c 65 <unknown>

fmaxnm  z0.d, p0/m, z0.d, #0.0
// CHECK-INST: fmaxnm	z0.d, p0/m, z0.d, #0.0
// CHECK-ENCODING: [0x00,0x80,0xdc,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 80 dc 65 <unknown>

fmaxnm  z31.h, p7/m, z31.h, #1.000000000000000
// CHECK-INST: fmaxnm	z31.h, p7/m, z31.h, #1.0
// CHECK-ENCODING: [0x3f,0x9c,0x5c,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 3f 9c 5c 65 <unknown>

fmaxnm  z31.h, p7/m, z31.h, #1.0
// CHECK-INST: fmaxnm	z31.h, p7/m, z31.h, #1.0
// CHECK-ENCODING: [0x3f,0x9c,0x5c,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 3f 9c 5c 65 <unknown>

fmaxnm  z31.s, p7/m, z31.s, #1.0
// CHECK-INST: fmaxnm	z31.s, p7/m, z31.s, #1.0
// CHECK-ENCODING: [0x3f,0x9c,0x9c,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 3f 9c 9c 65 <unknown>

fmaxnm  z31.d, p7/m, z31.d, #1.0
// CHECK-INST: fmaxnm	z31.d, p7/m, z31.d, #1.0
// CHECK-ENCODING: [0x3f,0x9c,0xdc,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 3f 9c dc 65 <unknown>

fmaxnm  z0.h, p7/m, z0.h, z31.h
// CHECK-INST: fmaxnm	z0.h, p7/m, z0.h, z31.h
// CHECK-ENCODING: [0xe0,0x9f,0x44,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 9f 44 65 <unknown>

fmaxnm  z0.s, p7/m, z0.s, z31.s
// CHECK-INST: fmaxnm	z0.s, p7/m, z0.s, z31.s
// CHECK-ENCODING: [0xe0,0x9f,0x84,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 9f 84 65 <unknown>

fmaxnm  z0.d, p7/m, z0.d, z31.d
// CHECK-INST: fmaxnm	z0.d, p7/m, z0.d, z31.d
// CHECK-ENCODING: [0xe0,0x9f,0xc4,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 9f c4 65 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z31.d, p7/z, z6.d
// CHECK-INST: movprfx	z31.d, p7/z, z6.d
// CHECK-ENCODING: [0xdf,0x3c,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: df 3c d0 04 <unknown>

fmaxnm  z31.d, p7/m, z31.d, #1.0
// CHECK-INST: fmaxnm	z31.d, p7/m, z31.d, #1.0
// CHECK-ENCODING: [0x3f,0x9c,0xdc,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 3f 9c dc 65 <unknown>

movprfx z31, z6
// CHECK-INST: movprfx	z31, z6
// CHECK-ENCODING: [0xdf,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: df bc 20 04 <unknown>

fmaxnm  z31.d, p7/m, z31.d, #1.0
// CHECK-INST: fmaxnm	z31.d, p7/m, z31.d, #1.0
// CHECK-ENCODING: [0x3f,0x9c,0xdc,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 3f 9c dc 65 <unknown>

movprfx z0.d, p7/z, z7.d
// CHECK-INST: movprfx	z0.d, p7/z, z7.d
// CHECK-ENCODING: [0xe0,0x3c,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 3c d0 04 <unknown>

fmaxnm  z0.d, p7/m, z0.d, z31.d
// CHECK-INST: fmaxnm	z0.d, p7/m, z0.d, z31.d
// CHECK-ENCODING: [0xe0,0x9f,0xc4,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 9f c4 65 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 bc 20 04 <unknown>

fmaxnm  z0.d, p7/m, z0.d, z31.d
// CHECK-INST: fmaxnm	z0.d, p7/m, z0.d, z31.d
// CHECK-ENCODING: [0xe0,0x9f,0xc4,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 9f c4 65 <unknown>
