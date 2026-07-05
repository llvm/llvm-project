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

fmax    z0.h, p0/m, z0.h, #0.000000000000000
// CHECK-INST: fmax    z0.h, p0/m, z0.h, #0.0
// CHECK-ENCODING: [0x00,0x80,0x5e,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 655e8000 <unknown>

fmax    z0.h, p0/m, z0.h, #0.0
// CHECK-INST: fmax    z0.h, p0/m, z0.h, #0.0
// CHECK-ENCODING: [0x00,0x80,0x5e,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 655e8000 <unknown>

fmax    z0.s, p0/m, z0.s, #0.0
// CHECK-INST: fmax    z0.s, p0/m, z0.s, #0.0
// CHECK-ENCODING: [0x00,0x80,0x9e,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 659e8000 <unknown>

fmax    z31.d, p7/m, z31.d, #1.0
// CHECK-INST: fmax    z31.d, p7/m, z31.d, #1.0
// CHECK-ENCODING: [0x3f,0x9c,0xde,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65de9c3f <unknown>

fmax    z31.h, p7/m, z31.h, #1.0
// CHECK-INST: fmax    z31.h, p7/m, z31.h, #1.0
// CHECK-ENCODING: [0x3f,0x9c,0x5e,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 655e9c3f <unknown>

fmax    z31.s, p7/m, z31.s, #1.0
// CHECK-INST: fmax    z31.s, p7/m, z31.s, #1.0
// CHECK-ENCODING: [0x3f,0x9c,0x9e,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 659e9c3f <unknown>

fmax    z0.d, p0/m, z0.d, #0.0
// CHECK-INST: fmax    z0.d, p0/m, z0.d, #0.0
// CHECK-ENCODING: [0x00,0x80,0xde,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65de8000 <unknown>

fmax    z0.h, p7/m, z0.h, z31.h
// CHECK-INST: fmax	z0.h, p7/m, z0.h, z31.h
// CHECK-ENCODING: [0xe0,0x9f,0x46,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65469fe0 <unknown>

fmax    z0.s, p7/m, z0.s, z31.s
// CHECK-INST: fmax	z0.s, p7/m, z0.s, z31.s
// CHECK-ENCODING: [0xe0,0x9f,0x86,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65869fe0 <unknown>

fmax    z0.d, p7/m, z0.d, z31.d
// CHECK-INST: fmax	z0.d, p7/m, z0.d, z31.d
// CHECK-ENCODING: [0xe0,0x9f,0xc6,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65c69fe0 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z0.d, p0/z, z7.d
// CHECK-INST: movprfx	z0.d, p0/z, z7.d
// CHECK-ENCODING: [0xe0,0x20,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04d020e0 <unknown>

fmax    z0.d, p0/m, z0.d, #0.0
// CHECK-INST: fmax	z0.d, p0/m, z0.d, #0.0
// CHECK-ENCODING: [0x00,0x80,0xde,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65de8000 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bce0 <unknown>

fmax    z0.d, p0/m, z0.d, #0.0
// CHECK-INST: fmax	z0.d, p0/m, z0.d, #0.0
// CHECK-ENCODING: [0x00,0x80,0xde,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65de8000 <unknown>

movprfx z0.d, p7/z, z7.d
// CHECK-INST: movprfx	z0.d, p7/z, z7.d
// CHECK-ENCODING: [0xe0,0x3c,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04d03ce0 <unknown>

fmax    z0.d, p7/m, z0.d, z31.d
// CHECK-INST: fmax	z0.d, p7/m, z0.d, z31.d
// CHECK-ENCODING: [0xe0,0x9f,0xc6,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65c69fe0 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bce0 <unknown>

fmax    z0.d, p7/m, z0.d, z31.d
// CHECK-INST: fmax	z0.d, p7/m, z0.d, z31.d
// CHECK-ENCODING: [0xe0,0x9f,0xc6,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65c69fe0 <unknown>
