// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:   | llvm-objdump -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN

fsubr   z0.h, p0/m, z0.h, #0.500000000000000
// CHECK-INST: fsubr	z0.h, p0/m, z0.h, #0.5
// CHECK-ENCODING: [0x00,0x80,0x5b,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 655b8000 <unknown>

fsubr   z0.h, p0/m, z0.h, #0.5
// CHECK-INST: fsubr	z0.h, p0/m, z0.h, #0.5
// CHECK-ENCODING: [0x00,0x80,0x5b,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 655b8000 <unknown>

fsubr   z0.s, p0/m, z0.s, #0.5
// CHECK-INST: fsubr	z0.s, p0/m, z0.s, #0.5
// CHECK-ENCODING: [0x00,0x80,0x9b,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 659b8000 <unknown>

fsubr   z0.d, p0/m, z0.d, #0.5
// CHECK-INST: fsubr	z0.d, p0/m, z0.d, #0.5
// CHECK-ENCODING: [0x00,0x80,0xdb,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65db8000 <unknown>

fsubr   z31.h, p7/m, z31.h, #1.000000000000000
// CHECK-INST: fsubr	z31.h, p7/m, z31.h, #1.0
// CHECK-ENCODING: [0x3f,0x9c,0x5b,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 655b9c3f <unknown>

fsubr   z31.h, p7/m, z31.h, #1.0
// CHECK-INST: fsubr	z31.h, p7/m, z31.h, #1.0
// CHECK-ENCODING: [0x3f,0x9c,0x5b,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 655b9c3f <unknown>

fsubr   z31.s, p7/m, z31.s, #1.0
// CHECK-INST: fsubr	z31.s, p7/m, z31.s, #1.0
// CHECK-ENCODING: [0x3f,0x9c,0x9b,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 659b9c3f <unknown>

fsubr   z31.d, p7/m, z31.d, #1.0
// CHECK-INST: fsubr	z31.d, p7/m, z31.d, #1.0
// CHECK-ENCODING: [0x3f,0x9c,0xdb,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65db9c3f <unknown>

fsubr   z0.h, p7/m, z0.h, z31.h
// CHECK-INST: fsubr	z0.h, p7/m, z0.h, z31.h
// CHECK-ENCODING: [0xe0,0x9f,0x43,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65439fe0 <unknown>

fsubr   z0.s, p7/m, z0.s, z31.s
// CHECK-INST: fsubr	z0.s, p7/m, z0.s, z31.s
// CHECK-ENCODING: [0xe0,0x9f,0x83,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65839fe0 <unknown>

fsubr   z0.d, p7/m, z0.d, z31.d
// CHECK-INST: fsubr	z0.d, p7/m, z0.d, z31.d
// CHECK-ENCODING: [0xe0,0x9f,0xc3,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65c39fe0 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z31.d, p7/z, z6.d
// CHECK-INST: movprfx	z31.d, p7/z, z6.d
// CHECK-ENCODING: [0xdf,0x3c,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04d03cdf <unknown>

fsubr   z31.d, p7/m, z31.d, #1.0
// CHECK-INST: fsubr	z31.d, p7/m, z31.d, #1.0
// CHECK-ENCODING: [0x3f,0x9c,0xdb,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65db9c3f <unknown>

movprfx z31, z6
// CHECK-INST: movprfx	z31, z6
// CHECK-ENCODING: [0xdf,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bcdf <unknown>

fsubr   z31.d, p7/m, z31.d, #1.0
// CHECK-INST: fsubr	z31.d, p7/m, z31.d, #1.0
// CHECK-ENCODING: [0x3f,0x9c,0xdb,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65db9c3f <unknown>

movprfx z0.d, p7/z, z7.d
// CHECK-INST: movprfx	z0.d, p7/z, z7.d
// CHECK-ENCODING: [0xe0,0x3c,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04d03ce0 <unknown>

fsubr   z0.d, p7/m, z0.d, z31.d
// CHECK-INST: fsubr	z0.d, p7/m, z0.d, z31.d
// CHECK-ENCODING: [0xe0,0x9f,0xc3,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65c39fe0 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bce0 <unknown>

fsubr   z0.d, p7/m, z0.d, z31.d
// CHECK-INST: fsubr	z0.d, p7/m, z0.d, z31.d
// CHECK-ENCODING: [0xe0,0x9f,0xc3,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65c39fe0 <unknown>
