// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d -mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN


sqrdmlah z0.b, z1.b, z31.b
// CHECK-INST: sqrdmlah z0.b, z1.b, z31.b
// CHECK-ENCODING: [0x20,0x70,0x1f,0x44]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 70 1f 44 <unknown>

sqrdmlah z0.h, z1.h, z31.h
// CHECK-INST: sqrdmlah z0.h, z1.h, z31.h
// CHECK-ENCODING: [0x20,0x70,0x5f,0x44]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 70 5f 44 <unknown>

sqrdmlah z0.s, z1.s, z31.s
// CHECK-INST: sqrdmlah z0.s, z1.s, z31.s
// CHECK-ENCODING: [0x20,0x70,0x9f,0x44]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 70 9f 44 <unknown>

sqrdmlah z0.d, z1.d, z31.d
// CHECK-INST: sqrdmlah z0.d, z1.d, z31.d
// CHECK-ENCODING: [0x20,0x70,0xdf,0x44]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 70 df 44 <unknown>

sqrdmlah z0.h, z1.h, z7.h[7]
// CHECK-INST: sqrdmlah	z0.h, z1.h, z7.h[7]
// CHECK-ENCODING: [0x20,0x10,0x7f,0x44]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 10 7f 44 <unknown>

sqrdmlah z0.s, z1.s, z7.s[3]
// CHECK-INST: sqrdmlah	z0.s, z1.s, z7.s[3]
// CHECK-ENCODING: [0x20,0x10,0xbf,0x44]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 10 bf 44 <unknown>

sqrdmlah z0.d, z1.d, z15.d[1]
// CHECK-INST: sqrdmlah	z0.d, z1.d, z15.d[1]
// CHECK-ENCODING: [0x20,0x10,0xff,0x44]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 10 ff 44 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 bc 20 04 <unknown>

sqrdmlah z0.d, z1.d, z31.d
// CHECK-INST: sqrdmlah z0.d, z1.d, z31.d
// CHECK-ENCODING: [0x20,0x70,0xdf,0x44]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 70 df 44 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 bc 20 04 <unknown>

sqrdmlah z0.d, z1.d, z15.d[1]
// CHECK-INST: sqrdmlah	z0.d, z1.d, z15.d[1]
// CHECK-ENCODING: [0x20,0x10,0xff,0x44]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 10 ff 44 <unknown>
