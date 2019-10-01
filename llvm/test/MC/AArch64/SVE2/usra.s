// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d -mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

usra     z0.b, z0.b, #1
// CHECK-INST: usra	z0.b, z0.b, #1
// CHECK-ENCODING: [0x00,0xe4,0x0f,0x45]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 00 e4 0f 45 <unknown>

usra     z31.b, z31.b, #8
// CHECK-INST: usra	z31.b, z31.b, #8
// CHECK-ENCODING: [0xff,0xe7,0x08,0x45]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: ff e7 08 45 <unknown>

usra     z0.h, z0.h, #1
// CHECK-INST: usra	z0.h, z0.h, #1
// CHECK-ENCODING: [0x00,0xe4,0x1f,0x45]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 00 e4 1f 45 <unknown>

usra     z31.h, z31.h, #16
// CHECK-INST: usra	z31.h, z31.h, #16
// CHECK-ENCODING: [0xff,0xe7,0x10,0x45]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: ff e7 10 45 <unknown>

usra     z0.s, z0.s, #1
// CHECK-INST: usra	z0.s, z0.s, #1
// CHECK-ENCODING: [0x00,0xe4,0x5f,0x45]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 00 e4 5f 45 <unknown>

usra     z31.s, z31.s, #32
// CHECK-INST: usra	z31.s, z31.s, #32
// CHECK-ENCODING: [0xff,0xe7,0x40,0x45]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: ff e7 40 45 <unknown>

usra     z0.d, z0.d, #1
// CHECK-INST: usra	z0.d, z0.d, #1
// CHECK-ENCODING: [0x00,0xe4,0xdf,0x45]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 00 e4 df 45 <unknown>

usra     z31.d, z31.d, #64
// CHECK-INST: usra	z31.d, z31.d, #64
// CHECK-ENCODING: [0xff,0xe7,0x80,0x45]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: ff e7 80 45 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 bc 20 04 <unknown>

usra     z0.d, z1.d, #1
// CHECK-INST: usra	z0.d, z1.d, #1
// CHECK-ENCODING: [0x20,0xe4,0xdf,0x45]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 e4 df 45 <unknown>
