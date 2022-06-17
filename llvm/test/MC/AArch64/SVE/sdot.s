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

sdot  z0.s, z1.b, z31.b
// CHECK-INST: sdot	z0.s, z1.b, z31.b
// CHECK-ENCODING: [0x20,0x00,0x9f,0x44]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 00 9f 44 <unknown>

sdot  z0.d, z1.h, z31.h
// CHECK-INST: sdot	z0.d, z1.h, z31.h
// CHECK-ENCODING: [0x20,0x00,0xdf,0x44]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 00 df 44 <unknown>

sdot  z0.s, z1.b, z7.b[3]
// CHECK-INST: sdot	z0.s, z1.b, z7.b[3]
// CHECK-ENCODING: [0x20,0x00,0xbf,0x44]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 00 bf 44 <unknown>

sdot  z0.d, z1.h, z15.h[1]
// CHECK-INST: sdot	z0.d, z1.h, z15.h[1]
// CHECK-ENCODING: [0x20,0x00,0xff,0x44]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 00 ff 44 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 bc 20 04 <unknown>

sdot  z0.d, z1.h, z31.h
// CHECK-INST: sdot	z0.d, z1.h, z31.h
// CHECK-ENCODING: [0x20,0x00,0xdf,0x44]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 00 df 44 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 bc 20 04 <unknown>

sdot  z0.d, z1.h, z15.h[1]
// CHECK-INST: sdot	z0.d, z1.h, z15.h[1]
// CHECK-ENCODING: [0x20,0x00,0xff,0x44]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 00 ff 44 <unknown>
