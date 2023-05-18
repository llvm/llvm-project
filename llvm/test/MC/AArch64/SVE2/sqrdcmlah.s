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

sqrdcmlah   z0.b, z1.b, z2.b, #0
// CHECK-INST: sqrdcmlah   z0.b, z1.b, z2.b, #0
// CHECK-ENCODING: [0x20,0x30,0x02,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44023020 <unknown>

sqrdcmlah   z0.h, z1.h, z2.h, #0
// CHECK-INST: sqrdcmlah   z0.h, z1.h, z2.h, #0
// CHECK-ENCODING: [0x20,0x30,0x42,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44423020 <unknown>

sqrdcmlah   z0.s, z1.s, z2.s, #0
// CHECK-INST: sqrdcmlah   z0.s, z1.s, z2.s, #0
// CHECK-ENCODING: [0x20,0x30,0x82,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44823020 <unknown>

sqrdcmlah   z0.d, z1.d, z2.d, #0
// CHECK-INST: sqrdcmlah   z0.d, z1.d, z2.d, #0
// CHECK-ENCODING: [0x20,0x30,0xc2,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44c23020 <unknown>

sqrdcmlah   z29.b, z30.b, z31.b, #90
// CHECK-INST: sqrdcmlah   z29.b, z30.b, z31.b, #90
// CHECK-ENCODING: [0xdd,0x37,0x1f,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 441f37dd <unknown>

sqrdcmlah   z29.h, z30.h, z31.h, #90
// CHECK-INST: sqrdcmlah   z29.h, z30.h, z31.h, #90
// CHECK-ENCODING: [0xdd,0x37,0x5f,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 445f37dd <unknown>

sqrdcmlah   z29.s, z30.s, z31.s, #90
// CHECK-INST: sqrdcmlah   z29.s, z30.s, z31.s, #90
// CHECK-ENCODING: [0xdd,0x37,0x9f,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 449f37dd <unknown>

sqrdcmlah   z29.d, z30.d, z31.d, #90
// CHECK-INST: sqrdcmlah   z29.d, z30.d, z31.d, #90
// CHECK-ENCODING: [0xdd,0x37,0xdf,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44df37dd <unknown>

sqrdcmlah   z31.b, z31.b, z31.b, #180
// CHECK-INST: sqrdcmlah   z31.b, z31.b, z31.b, #180
// CHECK-ENCODING: [0xff,0x3b,0x1f,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 441f3bff <unknown>

sqrdcmlah   z31.h, z31.h, z31.h, #180
// CHECK-INST: sqrdcmlah   z31.h, z31.h, z31.h, #180
// CHECK-ENCODING: [0xff,0x3b,0x5f,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 445f3bff <unknown>

sqrdcmlah   z31.s, z31.s, z31.s, #180
// CHECK-INST: sqrdcmlah   z31.s, z31.s, z31.s, #180
// CHECK-ENCODING: [0xff,0x3b,0x9f,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 449f3bff <unknown>

sqrdcmlah   z31.d, z31.d, z31.d, #180
// CHECK-INST: sqrdcmlah   z31.d, z31.d, z31.d, #180
// CHECK-ENCODING: [0xff,0x3b,0xdf,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44df3bff <unknown>

sqrdcmlah   z15.b, z16.b, z17.b, #270
// CHECK-INST: sqrdcmlah   z15.b, z16.b, z17.b, #270
// CHECK-ENCODING: [0x0f,0x3e,0x11,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44113e0f <unknown>

sqrdcmlah   z15.h, z16.h, z17.h, #270
// CHECK-INST: sqrdcmlah   z15.h, z16.h, z17.h, #270
// CHECK-ENCODING: [0x0f,0x3e,0x51,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44513e0f <unknown>

sqrdcmlah   z15.s, z16.s, z17.s, #270
// CHECK-INST: sqrdcmlah   z15.s, z16.s, z17.s, #270
// CHECK-ENCODING: [0x0f,0x3e,0x91,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44913e0f <unknown>

sqrdcmlah   z15.d, z16.d, z17.d, #270
// CHECK-INST: sqrdcmlah   z15.d, z16.d, z17.d, #270
// CHECK-ENCODING: [0x0f,0x3e,0xd1,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44d13e0f <unknown>

sqrdcmlah   z0.h, z1.h, z2.h[0], #0
// CHECK-INST: sqrdcmlah   z0.h, z1.h, z2.h[0], #0
// CHECK-ENCODING: [0x20,0x70,0xa2,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44a27020 <unknown>

sqrdcmlah   z0.s, z1.s, z2.s[0], #0
// CHECK-INST: sqrdcmlah   z0.s, z1.s, z2.s[0], #0
// CHECK-ENCODING: [0x20,0x70,0xe2,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44e27020 <unknown>

sqrdcmlah   z31.h, z30.h, z7.h[0], #180
// CHECK-INST: sqrdcmlah   z31.h, z30.h, z7.h[0], #180
// CHECK-ENCODING: [0xdf,0x7b,0xa7,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44a77bdf <unknown>

sqrdcmlah   z31.s, z30.s, z7.s[0], #180
// CHECK-INST: sqrdcmlah   z31.s, z30.s, z7.s[0], #180
// CHECK-ENCODING: [0xdf,0x7b,0xe7,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44e77bdf <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z4, z6
// CHECK-INST: movprfx	z4, z6
// CHECK-ENCODING: [0xc4,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bcc4 <unknown>

sqrdcmlah   z4.d, z31.d, z31.d, #270
// CHECK-INST: sqrdcmlah   z4.d, z31.d, z31.d, #270
// CHECK-ENCODING: [0xe4,0x3f,0xdf,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44df3fe4 <unknown>

movprfx z21, z28
// CHECK-INST: movprfx	z21, z28
// CHECK-ENCODING: [0x95,0xbf,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bf95 <unknown>

sqrdcmlah   z21.s, z10.s, z5.s[1], #90
// CHECK-INST: sqrdcmlah	z21.s, z10.s, z5.s[1], #90
// CHECK-ENCODING: [0x55,0x75,0xf5,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44f57555 <unknown>
