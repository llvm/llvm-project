// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

fsubr   z0.h, p0/m, z0.h, #0.500000000000000
// CHECK-INST: fsubr	z0.h, p0/m, z0.h, #0.5
// CHECK-ENCODING: [0x00,0x80,0x5b,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 5b 65 <unknown>

fsubr   z0.h, p0/m, z0.h, #0.5
// CHECK-INST: fsubr	z0.h, p0/m, z0.h, #0.5
// CHECK-ENCODING: [0x00,0x80,0x5b,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 5b 65 <unknown>

fsubr   z0.s, p0/m, z0.s, #0.5
// CHECK-INST: fsubr	z0.s, p0/m, z0.s, #0.5
// CHECK-ENCODING: [0x00,0x80,0x9b,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 9b 65 <unknown>

fsubr   z0.d, p0/m, z0.d, #0.5
// CHECK-INST: fsubr	z0.d, p0/m, z0.d, #0.5
// CHECK-ENCODING: [0x00,0x80,0xdb,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 db 65 <unknown>

fsubr   z31.h, p7/m, z31.h, #1.000000000000000
// CHECK-INST: fsubr	z31.h, p7/m, z31.h, #1.0
// CHECK-ENCODING: [0x3f,0x9c,0x5b,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 3f 9c 5b 65 <unknown>

fsubr   z31.h, p7/m, z31.h, #1.0
// CHECK-INST: fsubr	z31.h, p7/m, z31.h, #1.0
// CHECK-ENCODING: [0x3f,0x9c,0x5b,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 3f 9c 5b 65 <unknown>

fsubr   z31.s, p7/m, z31.s, #1.0
// CHECK-INST: fsubr	z31.s, p7/m, z31.s, #1.0
// CHECK-ENCODING: [0x3f,0x9c,0x9b,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 3f 9c 9b 65 <unknown>

fsubr   z31.d, p7/m, z31.d, #1.0
// CHECK-INST: fsubr	z31.d, p7/m, z31.d, #1.0
// CHECK-ENCODING: [0x3f,0x9c,0xdb,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 3f 9c db 65 <unknown>

fsubr   z0.h, p7/m, z0.h, z31.h
// CHECK-INST: fsubr	z0.h, p7/m, z0.h, z31.h
// CHECK-ENCODING: [0xe0,0x9f,0x43,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 9f 43 65 <unknown>

fsubr   z0.s, p7/m, z0.s, z31.s
// CHECK-INST: fsubr	z0.s, p7/m, z0.s, z31.s
// CHECK-ENCODING: [0xe0,0x9f,0x83,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 9f 83 65 <unknown>

fsubr   z0.d, p7/m, z0.d, z31.d
// CHECK-INST: fsubr	z0.d, p7/m, z0.d, z31.d
// CHECK-ENCODING: [0xe0,0x9f,0xc3,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 9f c3 65 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z31.d, p7/z, z6.d
// CHECK-INST: movprfx	z31.d, p7/z, z6.d
// CHECK-ENCODING: [0xdf,0x3c,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: df 3c d0 04 <unknown>

fsubr   z31.d, p7/m, z31.d, #1.0
// CHECK-INST: fsubr	z31.d, p7/m, z31.d, #1.0
// CHECK-ENCODING: [0x3f,0x9c,0xdb,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 3f 9c db 65 <unknown>

movprfx z31, z6
// CHECK-INST: movprfx	z31, z6
// CHECK-ENCODING: [0xdf,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: df bc 20 04 <unknown>

fsubr   z31.d, p7/m, z31.d, #1.0
// CHECK-INST: fsubr	z31.d, p7/m, z31.d, #1.0
// CHECK-ENCODING: [0x3f,0x9c,0xdb,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 3f 9c db 65 <unknown>

movprfx z0.d, p7/z, z7.d
// CHECK-INST: movprfx	z0.d, p7/z, z7.d
// CHECK-ENCODING: [0xe0,0x3c,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 3c d0 04 <unknown>

fsubr   z0.d, p7/m, z0.d, z31.d
// CHECK-INST: fsubr	z0.d, p7/m, z0.d, z31.d
// CHECK-ENCODING: [0xe0,0x9f,0xc3,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 9f c3 65 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 bc 20 04 <unknown>

fsubr   z0.d, p7/m, z0.d, z31.d
// CHECK-INST: fsubr	z0.d, p7/m, z0.d, z31.d
// CHECK-ENCODING: [0xe0,0x9f,0xc3,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 9f c3 65 <unknown>
