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

sqabs z31.b, p7/m, z31.b
// CHECK-INST: sqabs z31.b, p7/m, z31.b
// CHECK-ENCODING: [0xff,0xbf,0x08,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 4408bfff <unknown>

sqabs z31.h, p7/m, z31.h
// CHECK-INST: sqabs z31.h, p7/m, z31.h
// CHECK-ENCODING: [0xff,0xbf,0x48,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 4448bfff <unknown>

sqabs z31.s, p7/m, z31.s
// CHECK-INST: sqabs z31.s, p7/m, z31.s
// CHECK-ENCODING: [0xff,0xbf,0x88,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 4488bfff <unknown>

sqabs z31.d, p7/m, z31.d
// CHECK-INST: sqabs z31.d, p7/m, z31.d
// CHECK-ENCODING: [0xff,0xbf,0xc8,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44c8bfff <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z4.s, p7/z, z6.s
// CHECK-INST: movprfx	z4.s, p7/z, z6.s
// CHECK-ENCODING: [0xc4,0x3c,0x90,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04903cc4 <unknown>

sqabs z4.s, p7/m, z31.s
// CHECK-INST: sqabs z4.s, p7/m, z31.s
// CHECK-ENCODING: [0xe4,0xbf,0x88,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 4488bfe4 <unknown>

movprfx z4, z6
// CHECK-INST: movprfx	z4, z6
// CHECK-ENCODING: [0xc4,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bcc4 <unknown>

sqabs z4.s, p7/m, z31.s
// CHECK-INST: sqabs z4.s, p7/m, z31.s
// CHECK-ENCODING: [0xe4,0xbf,0x88,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 4488bfe4 <unknown>
