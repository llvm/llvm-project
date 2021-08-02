// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+streaming-sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

fcmla   z0.h, p0/m, z0.h, z0.h, #0
// CHECK-INST: fcmla z0.h, p0/m, z0.h, z0.h, #0
// CHECK-ENCODING: [0x00,0x00,0x40,0x64]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 00 40 64 <unknown>

fcmla   z0.s, p0/m, z0.s, z0.s, #0
// CHECK-INST: fcmla z0.s, p0/m, z0.s, z0.s, #0
// CHECK-ENCODING: [0x00,0x00,0x80,0x64]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 00 80 64 <unknown>

fcmla   z0.d, p0/m, z0.d, z0.d, #0
// CHECK-INST: fcmla z0.d, p0/m, z0.d, z0.d, #0
// CHECK-ENCODING: [0x00,0x00,0xc0,0x64]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 00 c0 64 <unknown>

fcmla   z0.h, p0/m, z1.h, z2.h, #90
// CHECK-INST: fcmla z0.h, p0/m, z1.h, z2.h, #90
// CHECK-ENCODING: [0x20,0x20,0x42,0x64]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 20 20 42 64 <unknown>

fcmla   z0.s, p0/m, z1.s, z2.s, #90
// CHECK-INST: fcmla z0.s, p0/m, z1.s, z2.s, #90
// CHECK-ENCODING: [0x20,0x20,0x82,0x64]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 20 20 82 64 <unknown>

fcmla   z0.d, p0/m, z1.d, z2.d, #90
// CHECK-INST: fcmla z0.d, p0/m, z1.d, z2.d, #90
// CHECK-ENCODING: [0x20,0x20,0xc2,0x64]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 20 20 c2 64 <unknown>

fcmla   z29.h, p7/m, z30.h, z31.h, #180
// CHECK-INST: fcmla z29.h, p7/m, z30.h, z31.h, #180
// CHECK-ENCODING: [0xdd,0x5f,0x5f,0x64]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: dd 5f 5f 64 <unknown>

fcmla   z29.s, p7/m, z30.s, z31.s, #180
// CHECK-INST: fcmla z29.s, p7/m, z30.s, z31.s, #180
// CHECK-ENCODING: [0xdd,0x5f,0x9f,0x64]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: dd 5f 9f 64 <unknown>

fcmla   z29.d, p7/m, z30.d, z31.d, #180
// CHECK-INST: fcmla z29.d, p7/m, z30.d, z31.d, #180
// CHECK-ENCODING: [0xdd,0x5f,0xdf,0x64]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: dd 5f df 64 <unknown>

fcmla   z31.h, p7/m, z31.h, z31.h, #270
// CHECK-INST: fcmla z31.h, p7/m, z31.h, z31.h, #270
// CHECK-ENCODING: [0xff,0x7f,0x5f,0x64]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: ff 7f 5f 64 <unknown>

fcmla   z31.s, p7/m, z31.s, z31.s, #270
// CHECK-INST: fcmla z31.s, p7/m, z31.s, z31.s, #270
// CHECK-ENCODING: [0xff,0x7f,0x9f,0x64]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: ff 7f 9f 64 <unknown>

fcmla   z31.d, p7/m, z31.d, z31.d, #270
// CHECK-INST: fcmla z31.d, p7/m, z31.d, z31.d, #270
// CHECK-ENCODING: [0xff,0x7f,0xdf,0x64]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: ff 7f df 64 <unknown>

fcmla   z0.h, z0.h, z0.h[0], #0
// CHECK-INST: fcmla   z0.h, z0.h, z0.h[0], #0
// CHECK-ENCODING: [0x00,0x10,0xa0,0x64]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 10 a0 64 <unknown>

fcmla   z23.s, z13.s, z8.s[0], #270
// CHECK-INST: fcmla   z23.s, z13.s, z8.s[0], #270
// CHECK-ENCODING: [0xb7,0x1d,0xe8,0x64]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: b7 1d e8 64 <unknown>

fcmla   z31.h, z31.h, z7.h[3], #270
// CHECK-INST: fcmla   z31.h, z31.h, z7.h[3], #270
// CHECK-ENCODING: [0xff,0x1f,0xbf,0x64]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: ff 1f bf 64 <unknown>

fcmla   z21.s, z10.s, z5.s[1], #90
// CHECK-INST: fcmla   z21.s, z10.s, z5.s[1], #90
// CHECK-ENCODING: [0x55,0x15,0xf5,0x64]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 55 15 f5 64 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z4.d, p7/z, z6.d
// CHECK-INST: movprfx	z4.d, p7/z, z6.d
// CHECK-ENCODING: [0xc4,0x3c,0xd0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: c4 3c d0 04 <unknown>

fcmla   z4.d, p7/m, z31.d, z31.d, #270
// CHECK-INST: fcmla	z4.d, p7/m, z31.d, z31.d, #270
// CHECK-ENCODING: [0xe4,0x7f,0xdf,0x64]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e4 7f df 64 <unknown>

movprfx z4, z6
// CHECK-INST: movprfx	z4, z6
// CHECK-ENCODING: [0xc4,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: c4 bc 20 04 <unknown>

fcmla   z4.d, p7/m, z31.d, z31.d, #270
// CHECK-INST: fcmla	z4.d, p7/m, z31.d, z31.d, #270
// CHECK-ENCODING: [0xe4,0x7f,0xdf,0x64]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e4 7f df 64 <unknown>

movprfx z21, z28
// CHECK-INST: movprfx	z21, z28
// CHECK-ENCODING: [0x95,0xbf,0x20,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 95 bf 20 04 <unknown>

fcmla   z21.s, z10.s, z5.s[1], #90
// CHECK-INST: fcmla	z21.s, z10.s, z5.s[1], #90
// CHECK-ENCODING: [0x55,0x15,0xf5,0x64]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 55 15 f5 64 <unknown>
