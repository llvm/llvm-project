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

sqneg z31.b, p7/m, z31.b
// CHECK-INST: sqneg z31.b, p7/m, z31.b
// CHECK-ENCODING: [0xff,0xbf,0x09,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 4409bfff <unknown>

sqneg z31.h, p7/m, z31.h
// CHECK-INST: sqneg z31.h, p7/m, z31.h
// CHECK-ENCODING: [0xff,0xbf,0x49,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 4449bfff <unknown>

sqneg z31.s, p7/m, z31.s
// CHECK-INST: sqneg z31.s, p7/m, z31.s
// CHECK-ENCODING: [0xff,0xbf,0x89,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 4489bfff <unknown>

sqneg z31.d, p7/m, z31.d
// CHECK-INST: sqneg z31.d, p7/m, z31.d
// CHECK-ENCODING: [0xff,0xbf,0xc9,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44c9bfff <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z4.s, p7/z, z6.s
// CHECK-INST: movprfx	z4.s, p7/z, z6.s
// CHECK-ENCODING: [0xc4,0x3c,0x90,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04903cc4 <unknown>

sqneg z4.s, p7/m, z31.s
// CHECK-INST: sqneg z4.s, p7/m, z31.s
// CHECK-ENCODING: [0xe4,0xbf,0x89,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 4489bfe4 <unknown>

movprfx z4, z6
// CHECK-INST: movprfx	z4, z6
// CHECK-ENCODING: [0xc4,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bcc4 <unknown>

sqneg z4.s, p7/m, z31.s
// CHECK-INST: sqneg z4.s, p7/m, z31.s
// CHECK-ENCODING: [0xe4,0xbf,0x89,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 4489bfe4 <unknown>
