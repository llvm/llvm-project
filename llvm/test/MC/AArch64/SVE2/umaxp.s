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

umaxp z0.b, p0/m, z0.b, z1.b
// CHECK-INST: umaxp z0.b, p0/m, z0.b, z1.b
// CHECK-ENCODING: [0x20,0xa0,0x15,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 4415a020 <unknown>

umaxp z0.h, p0/m, z0.h, z1.h
// CHECK-INST: umaxp z0.h, p0/m, z0.h, z1.h
// CHECK-ENCODING: [0x20,0xa0,0x55,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 4455a020 <unknown>

umaxp z29.s, p7/m, z29.s, z30.s
// CHECK-INST: umaxp z29.s, p7/m, z29.s, z30.s
// CHECK-ENCODING: [0xdd,0xbf,0x95,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 4495bfdd <unknown>

umaxp z31.d, p7/m, z31.d, z30.d
// CHECK-INST: umaxp z31.d, p7/m, z31.d, z30.d
// CHECK-ENCODING: [0xdf,0xbf,0xd5,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44d5bfdf <unknown>

// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z31.d, p0/z, z6.d
// CHECK-INST: movprfx z31.d, p0/z, z6.d
// CHECK-ENCODING: [0xdf,0x20,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04d020df <unknown>

umaxp z31.d, p0/m, z31.d, z30.d
// CHECK-INST: umaxp z31.d, p0/m, z31.d, z30.d
// CHECK-ENCODING: [0xdf,0xa3,0xd5,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44d5a3df <unknown>

movprfx z31, z6
// CHECK-INST: movprfx z31, z6
// CHECK-ENCODING: [0xdf,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bcdf <unknown>

umaxp z31.d, p7/m, z31.d, z30.d
// CHECK-INST: umaxp z31.d, p7/m, z31.d, z30.d
// CHECK-ENCODING: [0xdf,0xbf,0xd5,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44d5bfdf <unknown>
