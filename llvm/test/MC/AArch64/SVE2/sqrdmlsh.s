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

sqrdmlsh z0.b, z1.b, z31.b
// CHECK-INST: sqrdmlsh z0.b, z1.b, z31.b
// CHECK-ENCODING: [0x20,0x74,0x1f,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 441f7420 <unknown>

sqrdmlsh z0.h, z1.h, z31.h
// CHECK-INST: sqrdmlsh z0.h, z1.h, z31.h
// CHECK-ENCODING: [0x20,0x74,0x5f,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 445f7420 <unknown>

sqrdmlsh z0.s, z1.s, z31.s
// CHECK-INST: sqrdmlsh z0.s, z1.s, z31.s
// CHECK-ENCODING: [0x20,0x74,0x9f,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 449f7420 <unknown>

sqrdmlsh z0.d, z1.d, z31.d
// CHECK-INST: sqrdmlsh z0.d, z1.d, z31.d
// CHECK-ENCODING: [0x20,0x74,0xdf,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44df7420 <unknown>

sqrdmlsh z0.h, z1.h, z7.h[7]
// CHECK-INST: sqrdmlsh	z0.h, z1.h, z7.h[7]
// CHECK-ENCODING: [0x20,0x14,0x7f,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 447f1420 <unknown>

sqrdmlsh z0.s, z1.s, z7.s[3]
// CHECK-INST: sqrdmlsh	z0.s, z1.s, z7.s[3]
// CHECK-ENCODING: [0x20,0x14,0xbf,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44bf1420 <unknown>

sqrdmlsh z0.d, z1.d, z15.d[1]
// CHECK-INST: sqrdmlsh	z0.d, z1.d, z15.d[1]
// CHECK-ENCODING: [0x20,0x14,0xff,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44ff1420 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bce0 <unknown>

sqrdmlsh z0.d, z1.d, z31.d
// CHECK-INST: sqrdmlsh z0.d, z1.d, z31.d
// CHECK-ENCODING: [0x20,0x74,0xdf,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44df7420 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bce0 <unknown>

sqrdmlsh z0.d, z1.d, z15.d[1]
// CHECK-INST: sqrdmlsh	z0.d, z1.d, z15.d[1]
// CHECK-ENCODING: [0x20,0x14,0xff,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44ff1420 <unknown>
