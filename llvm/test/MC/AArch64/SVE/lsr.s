// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

lsr     z0.b, z0.b, #1
// CHECK-INST: lsr	z0.b, z0.b, #1
// CHECK-ENCODING: [0x00,0x94,0x2f,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 94 2f 04 <unknown>

lsr     z31.b, z31.b, #8
// CHECK-INST: lsr	z31.b, z31.b, #8
// CHECK-ENCODING: [0xff,0x97,0x28,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 97 28 04 <unknown>

lsr     z0.h, z0.h, #1
// CHECK-INST: lsr	z0.h, z0.h, #1
// CHECK-ENCODING: [0x00,0x94,0x3f,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 94 3f 04 <unknown>

lsr     z31.h, z31.h, #16
// CHECK-INST: lsr	z31.h, z31.h, #16
// CHECK-ENCODING: [0xff,0x97,0x30,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 97 30 04 <unknown>

lsr     z0.s, z0.s, #1
// CHECK-INST: lsr	z0.s, z0.s, #1
// CHECK-ENCODING: [0x00,0x94,0x7f,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 94 7f 04 <unknown>

lsr     z31.s, z31.s, #32
// CHECK-INST: lsr	z31.s, z31.s, #32
// CHECK-ENCODING: [0xff,0x97,0x60,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 97 60 04 <unknown>

lsr     z0.d, z0.d, #1
// CHECK-INST: lsr	z0.d, z0.d, #1
// CHECK-ENCODING: [0x00,0x94,0xff,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 94 ff 04 <unknown>

lsr     z31.d, z31.d, #64
// CHECK-INST: lsr	z31.d, z31.d, #64
// CHECK-ENCODING: [0xff,0x97,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 97 a0 04 <unknown>

lsr     z0.b, p0/m, z0.b, #1
// CHECK-INST: lsr	z0.b, p0/m, z0.b, #1
// CHECK-ENCODING: [0xe0,0x81,0x01,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 81 01 04 <unknown>

lsr     z31.b, p0/m, z31.b, #8
// CHECK-INST: lsr	z31.b, p0/m, z31.b, #8
// CHECK-ENCODING: [0x1f,0x81,0x01,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 1f 81 01 04 <unknown>

lsr     z0.h, p0/m, z0.h, #1
// CHECK-INST: lsr	z0.h, p0/m, z0.h, #1
// CHECK-ENCODING: [0xe0,0x83,0x01,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 83 01 04 <unknown>

lsr     z31.h, p0/m, z31.h, #16
// CHECK-INST: lsr	z31.h, p0/m, z31.h, #16
// CHECK-ENCODING: [0x1f,0x82,0x01,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 1f 82 01 04 <unknown>

lsr     z0.s, p0/m, z0.s, #1
// CHECK-INST: lsr	z0.s, p0/m, z0.s, #1
// CHECK-ENCODING: [0xe0,0x83,0x41,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 83 41 04 <unknown>

lsr     z31.s, p0/m, z31.s, #32
// CHECK-INST: lsr	z31.s, p0/m, z31.s, #32
// CHECK-ENCODING: [0x1f,0x80,0x41,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 1f 80 41 04 <unknown>

lsr     z0.d, p0/m, z0.d, #1
// CHECK-INST: lsr	z0.d, p0/m, z0.d, #1
// CHECK-ENCODING: [0xe0,0x83,0xc1,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 83 c1 04 <unknown>

lsr     z31.d, p0/m, z31.d, #64
// CHECK-INST: lsr	z31.d, p0/m, z31.d, #64
// CHECK-ENCODING: [0x1f,0x80,0x81,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 1f 80 81 04 <unknown>

lsr     z0.b, p0/m, z0.b, z0.b
// CHECK-INST: lsr	z0.b, p0/m, z0.b, z0.b
// CHECK-ENCODING: [0x00,0x80,0x11,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 11 04 <unknown>

lsr     z0.h, p0/m, z0.h, z0.h
// CHECK-INST: lsr	z0.h, p0/m, z0.h, z0.h
// CHECK-ENCODING: [0x00,0x80,0x51,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 51 04 <unknown>

lsr     z0.s, p0/m, z0.s, z0.s
// CHECK-INST: lsr	z0.s, p0/m, z0.s, z0.s
// CHECK-ENCODING: [0x00,0x80,0x91,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 91 04 <unknown>

lsr     z0.d, p0/m, z0.d, z0.d
// CHECK-INST: lsr	z0.d, p0/m, z0.d, z0.d
// CHECK-ENCODING: [0x00,0x80,0xd1,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 d1 04 <unknown>

lsr     z0.b, p0/m, z0.b, z1.d
// CHECK-INST: lsr	z0.b, p0/m, z0.b, z1.d
// CHECK-ENCODING: [0x20,0x80,0x19,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 20 80 19 04 <unknown>

lsr     z0.h, p0/m, z0.h, z1.d
// CHECK-INST: lsr	z0.h, p0/m, z0.h, z1.d
// CHECK-ENCODING: [0x20,0x80,0x59,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 20 80 59 04 <unknown>

lsr     z0.s, p0/m, z0.s, z1.d
// CHECK-INST: lsr	z0.s, p0/m, z0.s, z1.d
// CHECK-ENCODING: [0x20,0x80,0x99,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 20 80 99 04 <unknown>

lsr     z0.b, z1.b, z2.d
// CHECK-INST: lsr	z0.b, z1.b, z2.d
// CHECK-ENCODING: [0x20,0x84,0x22,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 20 84 22 04 <unknown>

lsr     z0.h, z1.h, z2.d
// CHECK-INST: lsr	z0.h, z1.h, z2.d
// CHECK-ENCODING: [0x20,0x84,0x62,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 20 84 62 04 <unknown>

lsr     z0.s, z1.s, z2.d
// CHECK-INST: lsr	z0.s, z1.s, z2.d
// CHECK-ENCODING: [0x20,0x84,0xa2,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 20 84 a2 04 <unknown>
