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

cmla   z0.b, z1.b, z2.b, #0
// CHECK-INST: cmla   z0.b, z1.b, z2.b, #0
// CHECK-ENCODING: [0x20,0x20,0x02,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 20 20 02 44 <unknown>

cmla   z0.h, z1.h, z2.h, #0
// CHECK-INST: cmla   z0.h, z1.h, z2.h, #0
// CHECK-ENCODING: [0x20,0x20,0x42,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 20 20 42 44 <unknown>

cmla   z0.s, z1.s, z2.s, #0
// CHECK-INST: cmla   z0.s, z1.s, z2.s, #0
// CHECK-ENCODING: [0x20,0x20,0x82,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 20 20 82 44 <unknown>

cmla   z0.d, z1.d, z2.d, #0
// CHECK-INST: cmla   z0.d, z1.d, z2.d, #0
// CHECK-ENCODING: [0x20,0x20,0xc2,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 20 20 c2 44 <unknown>

cmla   z29.b, z30.b, z31.b, #90
// CHECK-INST: cmla   z29.b, z30.b, z31.b, #90
// CHECK-ENCODING: [0xdd,0x27,0x1f,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: dd 27 1f 44 <unknown>

cmla   z29.h, z30.h, z31.h, #90
// CHECK-INST: cmla   z29.h, z30.h, z31.h, #90
// CHECK-ENCODING: [0xdd,0x27,0x5f,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: dd 27 5f 44 <unknown>

cmla   z29.s, z30.s, z31.s, #90
// CHECK-INST: cmla   z29.s, z30.s, z31.s, #90
// CHECK-ENCODING: [0xdd,0x27,0x9f,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: dd 27 9f 44 <unknown>

cmla   z29.d, z30.d, z31.d, #90
// CHECK-INST: cmla   z29.d, z30.d, z31.d, #90
// CHECK-ENCODING: [0xdd,0x27,0xdf,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: dd 27 df 44 <unknown>

cmla   z31.b, z31.b, z31.b, #180
// CHECK-INST: cmla   z31.b, z31.b, z31.b, #180
// CHECK-ENCODING: [0xff,0x2b,0x1f,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff 2b 1f 44 <unknown>

cmla   z31.h, z31.h, z31.h, #180
// CHECK-INST: cmla   z31.h, z31.h, z31.h, #180
// CHECK-ENCODING: [0xff,0x2b,0x5f,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff 2b 5f 44 <unknown>

cmla   z31.s, z31.s, z31.s, #180
// CHECK-INST: cmla   z31.s, z31.s, z31.s, #180
// CHECK-ENCODING: [0xff,0x2b,0x9f,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff 2b 9f 44 <unknown>

cmla   z31.d, z31.d, z31.d, #180
// CHECK-INST: cmla   z31.d, z31.d, z31.d, #180
// CHECK-ENCODING: [0xff,0x2b,0xdf,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff 2b df 44 <unknown>

cmla   z15.b, z16.b, z17.b, #270
// CHECK-INST: cmla   z15.b, z16.b, z17.b, #270
// CHECK-ENCODING: [0x0f,0x2e,0x11,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 0f 2e 11 44 <unknown>

cmla   z15.h, z16.h, z17.h, #270
// CHECK-INST: cmla   z15.h, z16.h, z17.h, #270
// CHECK-ENCODING: [0x0f,0x2e,0x51,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 0f 2e 51 44 <unknown>

cmla   z15.s, z16.s, z17.s, #270
// CHECK-INST: cmla   z15.s, z16.s, z17.s, #270
// CHECK-ENCODING: [0x0f,0x2e,0x91,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 0f 2e 91 44 <unknown>

cmla   z15.d, z16.d, z17.d, #270
// CHECK-INST: cmla   z15.d, z16.d, z17.d, #270
// CHECK-ENCODING: [0x0f,0x2e,0xd1,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 0f 2e d1 44 <unknown>

cmla   z0.h, z1.h, z2.h[0], #0
// CHECK-INST: cmla   z0.h, z1.h, z2.h[0], #0
// CHECK-ENCODING: [0x20,0x60,0xa2,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 20 60 a2 44 <unknown>

cmla   z0.s, z1.s, z2.s[0], #0
// CHECK-INST: cmla   z0.s, z1.s, z2.s[0], #0
// CHECK-ENCODING: [0x20,0x60,0xe2,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 20 60 e2 44 <unknown>

cmla   z31.h, z30.h, z7.h[0], #180
// CHECK-INST: cmla   z31.h, z30.h, z7.h[0], #180
// CHECK-ENCODING: [0xdf,0x6b,0xa7,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: df 6b a7 44 <unknown>

cmla   z31.s, z30.s, z7.s[0], #180
// CHECK-INST: cmla   z31.s, z30.s, z7.s[0], #180
// CHECK-ENCODING: [0xdf,0x6b,0xe7,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: df 6b e7 44 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z4, z6
// CHECK-INST: movprfx	z4, z6
// CHECK-ENCODING: [0xc4,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c4 bc 20 04 <unknown>

cmla   z4.d, z31.d, z31.d, #270
// CHECK-INST: cmla   z4.d, z31.d, z31.d, #270
// CHECK-ENCODING: [0xe4,0x2f,0xdf,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: e4 2f df 44 <unknown>

movprfx z21, z28
// CHECK-INST: movprfx	z21, z28
// CHECK-ENCODING: [0x95,0xbf,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 95 bf 20 04 <unknown>

cmla   z21.s, z10.s, z5.s[1], #90
// CHECK-INST: cmla	z21.s, z10.s, z5.s[1], #90
// CHECK-ENCODING: [0x55,0x65,0xf5,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 55 65 f5 44 <unknown>
