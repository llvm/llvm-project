// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

urshr    z0.b, p0/m, z0.b, #1
// CHECK-INST: urshr	z0.b, p0/m, z0.b, #1
// CHECK-ENCODING: [0xe0,0x81,0x0d,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: e0 81 0d 04 <unknown>

urshr    z31.b, p0/m, z31.b, #8
// CHECK-INST: urshr	z31.b, p0/m, z31.b, #8
// CHECK-ENCODING: [0x1f,0x81,0x0d,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 1f 81 0d 04 <unknown>

urshr    z0.h, p0/m, z0.h, #1
// CHECK-INST: urshr	z0.h, p0/m, z0.h, #1
// CHECK-ENCODING: [0xe0,0x83,0x0d,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: e0 83 0d 04 <unknown>

urshr    z31.h, p0/m, z31.h, #16
// CHECK-INST: urshr	z31.h, p0/m, z31.h, #16
// CHECK-ENCODING: [0x1f,0x82,0x0d,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 1f 82 0d 04 <unknown>

urshr    z0.s, p0/m, z0.s, #1
// CHECK-INST: urshr	z0.s, p0/m, z0.s, #1
// CHECK-ENCODING: [0xe0,0x83,0x4d,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: e0 83 4d 04 <unknown>

urshr    z31.s, p0/m, z31.s, #32
// CHECK-INST: urshr	z31.s, p0/m, z31.s, #32
// CHECK-ENCODING: [0x1f,0x80,0x4d,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 1f 80 4d 04 <unknown>

urshr    z0.d, p0/m, z0.d, #1
// CHECK-INST: urshr	z0.d, p0/m, z0.d, #1
// CHECK-ENCODING: [0xe0,0x83,0xcd,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: e0 83 cd 04 <unknown>

urshr    z31.d, p0/m, z31.d, #64
// CHECK-INST: urshr	z31.d, p0/m, z31.d, #64
// CHECK-ENCODING: [0x1f,0x80,0x8d,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 1f 80 8d 04 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z31.d, p0/z, z6.d
// CHECK-INST: movprfx	z31.d, p0/z, z6.d
// CHECK-ENCODING: [0xdf,0x20,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: df 20 d0 04 <unknown>

urshr    z31.d, p0/m, z31.d, #64
// CHECK-INST: urshr	z31.d, p0/m, z31.d, #64
// CHECK-ENCODING: [0x1f,0x80,0x8d,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 1f 80 8d 04 <unknown>

movprfx z31, z6
// CHECK-INST: movprfx	z31, z6
// CHECK-ENCODING: [0xdf,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: df bc 20 04 <unknown>

urshr    z31.d, p0/m, z31.d, #64
// CHECK-INST: urshr	z31.d, p0/m, z31.d, #64
// CHECK-ENCODING: [0x1f,0x80,0x8d,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 1f 80 8d 04 <unknown>
