// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

eor     z5.b, z5.b, #0xf9
// CHECK-INST: eor     z5.b, z5.b, #0xf9
// CHECK-ENCODING: [0xa5,0x2e,0x40,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a5 2e 40 05 <unknown>

eor     z23.h, z23.h, #0xfff9
// CHECK-INST: eor     z23.h, z23.h, #0xfff9
// CHECK-ENCODING: [0xb7,0x6d,0x40,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: b7 6d 40 05 <unknown>

eor     z0.s, z0.s, #0xfffffff9
// CHECK-INST: eor     z0.s, z0.s, #0xfffffff9
// CHECK-ENCODING: [0xa0,0xeb,0x40,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a0 eb 40 05 <unknown>

eor     z0.d, z0.d, #0xfffffffffffffff9
// CHECK-INST: eor     z0.d, z0.d, #0xfffffffffffffff9
// CHECK-ENCODING: [0xa0,0xef,0x43,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a0 ef 43 05 <unknown>

eor     z5.b, z5.b, #0x6
// CHECK-INST: eor     z5.b, z5.b, #0x6
// CHECK-ENCODING: [0x25,0x3e,0x40,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 25 3e 40 05 <unknown>

eor     z23.h, z23.h, #0x6
// CHECK-INST: eor     z23.h, z23.h, #0x6
// CHECK-ENCODING: [0x37,0x7c,0x40,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 37 7c 40 05 <unknown>

eor     z0.s, z0.s, #0x6
// CHECK-INST: eor     z0.s, z0.s, #0x6
// CHECK-ENCODING: [0x20,0xf8,0x40,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 20 f8 40 05 <unknown>

eor     z0.d, z0.d, #0x6
// CHECK-INST: eor     z0.d, z0.d, #0x6
// CHECK-ENCODING: [0x20,0xf8,0x43,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 20 f8 43 05 <unknown>

eor     z23.d, z13.d, z8.d
// CHECK-INST: eor     z23.d, z13.d, z8.d
// CHECK-ENCODING: [0xb7,0x31,0xa8,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: b7 31 a8 04 <unknown>

eor     z0.d, z0.d, z0.d
// CHECK-INST: eor     z0.d, z0.d, z0.d
// CHECK-ENCODING: [0x00,0x30,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 30 a0 04 <unknown>

eor     z31.s, p7/m, z31.s, z31.s
// CHECK-INST: eor     z31.s, p7/m, z31.s, z31.s
// CHECK-ENCODING: [0xff,0x1f,0x99,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 1f 99 04 <unknown>

eor     z31.h, p7/m, z31.h, z31.h
// CHECK-INST: eor     z31.h, p7/m, z31.h, z31.h
// CHECK-ENCODING: [0xff,0x1f,0x59,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 1f 59 04 <unknown>

eor     z31.d, p7/m, z31.d, z31.d
// CHECK-INST: eor     z31.d, p7/m, z31.d, z31.d
// CHECK-ENCODING: [0xff,0x1f,0xd9,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 1f d9 04 <unknown>

eor     z31.b, p7/m, z31.b, z31.b
// CHECK-INST: eor     z31.b, p7/m, z31.b, z31.b
// CHECK-ENCODING: [0xff,0x1f,0x19,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 1f 19 04 <unknown>

eor     p0.b, p0/z, p0.b, p1.b
// CHECK-INST: eor     p0.b, p0/z, p0.b, p1.b
// CHECK-ENCODING: [0x00,0x42,0x01,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 42 01 25 <unknown>

eor     p0.b, p0/z, p0.b, p0.b
// CHECK-INST: not     p0.b, p0/z, p0.b
// CHECK-ENCODING: [0x00,0x42,0x00,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 42 00 25 <unknown>

eor     p15.b, p15/z, p15.b, p15.b
// CHECK-INST: not     p15.b, p15/z, p15.b
// CHECK-ENCODING: [0xef,0x7f,0x0f,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ef 7f 0f 25 <unknown>


// --------------------------------------------------------------------------//
// Test aliases.

eor     z0.s, z0.s, z0.s
// CHECK-INST: eor     z0.d, z0.d, z0.d
// CHECK-ENCODING: [0x00,0x30,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 30 a0 04 <unknown>

eor     z0.h, z0.h, z0.h
// CHECK-INST: eor     z0.d, z0.d, z0.d
// CHECK-ENCODING: [0x00,0x30,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 30 a0 04 <unknown>

eor     z0.b, z0.b, z0.b
// CHECK-INST: eor     z0.d, z0.d, z0.d
// CHECK-ENCODING: [0x00,0x30,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 30 a0 04 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z4.b, p7/z, z6.b
// CHECK-INST: movprfx	z4.b, p7/z, z6.b
// CHECK-ENCODING: [0xc4,0x3c,0x10,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4 3c 10 04 <unknown>

eor     z4.b, p7/m, z4.b, z31.b
// CHECK-INST: eor	z4.b, p7/m, z4.b, z31.b
// CHECK-ENCODING: [0xe4,0x1f,0x19,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e4 1f 19 04 <unknown>

movprfx z4, z6
// CHECK-INST: movprfx	z4, z6
// CHECK-ENCODING: [0xc4,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4 bc 20 04 <unknown>

eor     z4.b, p7/m, z4.b, z31.b
// CHECK-INST: eor	z4.b, p7/m, z4.b, z31.b
// CHECK-ENCODING: [0xe4,0x1f,0x19,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e4 1f 19 04 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 bc 20 04 <unknown>

eor     z0.d, z0.d, #0x6
// CHECK-INST: eor	z0.d, z0.d, #0x6
// CHECK-ENCODING: [0x20,0xf8,0x43,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 20 f8 43 05 <unknown>
