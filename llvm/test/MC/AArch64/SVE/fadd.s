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

fadd    z0.h, p0/m, z0.h, #0.500000000000000
// CHECK-INST: fadd    z0.h, p0/m, z0.h, #0.5
// CHECK-ENCODING: [0x00,0x80,0x58,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65588000 <unknown>

fadd    z0.h, p0/m, z0.h, #0.5
// CHECK-INST: fadd    z0.h, p0/m, z0.h, #0.5
// CHECK-ENCODING: [0x00,0x80,0x58,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65588000 <unknown>

fadd    z0.s, p0/m, z0.s, #0.5
// CHECK-INST: fadd    z0.s, p0/m, z0.s, #0.5
// CHECK-ENCODING: [0x00,0x80,0x98,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65988000 <unknown>

fadd    z0.d, p0/m, z0.d, #0.5
// CHECK-INST: fadd    z0.d, p0/m, z0.d, #0.5
// CHECK-ENCODING: [0x00,0x80,0xd8,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65d88000 <unknown>

fadd    z31.h, p7/m, z31.h, #1.000000000000000
// CHECK-INST: fadd    z31.h, p7/m, z31.h, #1.0
// CHECK-ENCODING: [0x3f,0x9c,0x58,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65589c3f <unknown>

fadd    z31.h, p7/m, z31.h, #1.0
// CHECK-INST: fadd    z31.h, p7/m, z31.h, #1.0
// CHECK-ENCODING: [0x3f,0x9c,0x58,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65589c3f <unknown>

fadd    z31.s, p7/m, z31.s, #1.0
// CHECK-INST: fadd    z31.s, p7/m, z31.s, #1.0
// CHECK-ENCODING: [0x3f,0x9c,0x98,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65989c3f <unknown>

fadd    z31.d, p7/m, z31.d, #1.0
// CHECK-INST: fadd    z31.d, p7/m, z31.d, #1.0
// CHECK-ENCODING: [0x3f,0x9c,0xd8,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65d89c3f <unknown>

fadd    z0.h, p7/m, z0.h, z31.h
// CHECK-INST: fadd	z0.h, p7/m, z0.h, z31.h
// CHECK-ENCODING: [0xe0,0x9f,0x40,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65409fe0 <unknown>

fadd    z0.s, p7/m, z0.s, z31.s
// CHECK-INST: fadd	z0.s, p7/m, z0.s, z31.s
// CHECK-ENCODING: [0xe0,0x9f,0x80,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65809fe0 <unknown>

fadd    z0.d, p7/m, z0.d, z31.d
// CHECK-INST: fadd	z0.d, p7/m, z0.d, z31.d
// CHECK-ENCODING: [0xe0,0x9f,0xc0,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65c09fe0 <unknown>

fadd z0.h, z1.h, z31.h
// CHECK-INST: fadd	z0.h, z1.h, z31.h
// CHECK-ENCODING: [0x20,0x00,0x5f,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 655f0020 <unknown>

fadd z0.s, z1.s, z31.s
// CHECK-INST: fadd	z0.s, z1.s, z31.s
// CHECK-ENCODING: [0x20,0x00,0x9f,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 659f0020 <unknown>

fadd z0.d, z1.d, z31.d
// CHECK-INST: fadd	z0.d, z1.d, z31.d
// CHECK-ENCODING: [0x20,0x00,0xdf,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65df0020 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z31.d, p7/z, z6.d
// CHECK-INST: movprfx	z31.d, p7/z, z6.d
// CHECK-ENCODING: [0xdf,0x3c,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04d03cdf <unknown>

fadd    z31.d, p7/m, z31.d, #1.0
// CHECK-INST: fadd	z31.d, p7/m, z31.d, #1.0
// CHECK-ENCODING: [0x3f,0x9c,0xd8,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65d89c3f <unknown>

movprfx z31, z6
// CHECK-INST: movprfx	z31, z6
// CHECK-ENCODING: [0xdf,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bcdf <unknown>

fadd    z31.d, p7/m, z31.d, #1.0
// CHECK-INST: fadd	z31.d, p7/m, z31.d, #1.0
// CHECK-ENCODING: [0x3f,0x9c,0xd8,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65d89c3f <unknown>

movprfx z0.d, p7/z, z7.d
// CHECK-INST: movprfx	z0.d, p7/z, z7.d
// CHECK-ENCODING: [0xe0,0x3c,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04d03ce0 <unknown>

fadd    z0.d, p7/m, z0.d, z31.d
// CHECK-INST: fadd	z0.d, p7/m, z0.d, z31.d
// CHECK-ENCODING: [0xe0,0x9f,0xc0,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65c09fe0 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bce0 <unknown>

fadd    z0.d, p7/m, z0.d, z31.d
// CHECK-INST: fadd	z0.d, p7/m, z0.d, z31.d
// CHECK-ENCODING: [0xe0,0x9f,0xc0,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65c09fe0 <unknown>
