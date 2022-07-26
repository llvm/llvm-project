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

urshr    z0.b, p0/m, z0.b, #1
// CHECK-INST: urshr	z0.b, p0/m, z0.b, #1
// CHECK-ENCODING: [0xe0,0x81,0x0d,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 040d81e0 <unknown>

urshr    z31.b, p0/m, z31.b, #8
// CHECK-INST: urshr	z31.b, p0/m, z31.b, #8
// CHECK-ENCODING: [0x1f,0x81,0x0d,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 040d811f <unknown>

urshr    z0.h, p0/m, z0.h, #1
// CHECK-INST: urshr	z0.h, p0/m, z0.h, #1
// CHECK-ENCODING: [0xe0,0x83,0x0d,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 040d83e0 <unknown>

urshr    z31.h, p0/m, z31.h, #16
// CHECK-INST: urshr	z31.h, p0/m, z31.h, #16
// CHECK-ENCODING: [0x1f,0x82,0x0d,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 040d821f <unknown>

urshr    z0.s, p0/m, z0.s, #1
// CHECK-INST: urshr	z0.s, p0/m, z0.s, #1
// CHECK-ENCODING: [0xe0,0x83,0x4d,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 044d83e0 <unknown>

urshr    z31.s, p0/m, z31.s, #32
// CHECK-INST: urshr	z31.s, p0/m, z31.s, #32
// CHECK-ENCODING: [0x1f,0x80,0x4d,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 044d801f <unknown>

urshr    z0.d, p0/m, z0.d, #1
// CHECK-INST: urshr	z0.d, p0/m, z0.d, #1
// CHECK-ENCODING: [0xe0,0x83,0xcd,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 04cd83e0 <unknown>

urshr    z31.d, p0/m, z31.d, #64
// CHECK-INST: urshr	z31.d, p0/m, z31.d, #64
// CHECK-ENCODING: [0x1f,0x80,0x8d,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 048d801f <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z31.d, p0/z, z6.d
// CHECK-INST: movprfx	z31.d, p0/z, z6.d
// CHECK-ENCODING: [0xdf,0x20,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04d020df <unknown>

urshr    z31.d, p0/m, z31.d, #64
// CHECK-INST: urshr	z31.d, p0/m, z31.d, #64
// CHECK-ENCODING: [0x1f,0x80,0x8d,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 048d801f <unknown>

movprfx z31, z6
// CHECK-INST: movprfx	z31, z6
// CHECK-ENCODING: [0xdf,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bcdf <unknown>

urshr    z31.d, p0/m, z31.d, #64
// CHECK-INST: urshr	z31.d, p0/m, z31.d, #64
// CHECK-ENCODING: [0x1f,0x80,0x8d,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 048d801f <unknown>
