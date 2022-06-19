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

lsl     z0.b, z0.b, #0
// CHECK-INST: lsl	z0.b, z0.b, #0
// CHECK-ENCODING: [0x00,0x9c,0x28,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 9c 28 04 <unknown>

lsl     z31.b, z31.b, #7
// CHECK-INST: lsl	z31.b, z31.b, #7
// CHECK-ENCODING: [0xff,0x9f,0x2f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 9f 2f 04 <unknown>

lsl     z0.h, z0.h, #0
// CHECK-INST: lsl	z0.h, z0.h, #0
// CHECK-ENCODING: [0x00,0x9c,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 9c 30 04 <unknown>

lsl     z31.h, z31.h, #15
// CHECK-INST: lsl	z31.h, z31.h, #15
// CHECK-ENCODING: [0xff,0x9f,0x3f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 9f 3f 04 <unknown>

lsl     z0.s, z0.s, #0
// CHECK-INST: lsl	z0.s, z0.s, #0
// CHECK-ENCODING: [0x00,0x9c,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 9c 60 04 <unknown>

lsl     z31.s, z31.s, #31
// CHECK-INST: lsl	z31.s, z31.s, #31
// CHECK-ENCODING: [0xff,0x9f,0x7f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 9f 7f 04 <unknown>

lsl     z0.d, z0.d, #0
// CHECK-INST: lsl	z0.d, z0.d, #0
// CHECK-ENCODING: [0x00,0x9c,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 9c a0 04 <unknown>

lsl     z31.d, z31.d, #63
// CHECK-INST: lsl	z31.d, z31.d, #63
// CHECK-ENCODING: [0xff,0x9f,0xff,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 9f ff 04 <unknown>

lsl     z0.b, p0/m, z0.b, #0
// CHECK-INST: lsl	z0.b, p0/m, z0.b, #0
// CHECK-ENCODING: [0x00,0x81,0x03,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 81 03 04 <unknown>

lsl     z31.b, p0/m, z31.b, #7
// CHECK-INST: lsl	z31.b, p0/m, z31.b, #7
// CHECK-ENCODING: [0xff,0x81,0x03,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 81 03 04 <unknown>

lsl     z0.h, p0/m, z0.h, #0
// CHECK-INST: lsl	z0.h, p0/m, z0.h, #0
// CHECK-ENCODING: [0x00,0x82,0x03,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 82 03 04 <unknown>

lsl     z31.h, p0/m, z31.h, #15
// CHECK-INST: lsl	z31.h, p0/m, z31.h, #15
// CHECK-ENCODING: [0xff,0x83,0x03,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 83 03 04 <unknown>

lsl     z0.s, p0/m, z0.s, #0
// CHECK-INST: lsl	z0.s, p0/m, z0.s, #0
// CHECK-ENCODING: [0x00,0x80,0x43,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 80 43 04 <unknown>

lsl     z31.s, p0/m, z31.s, #31
// CHECK-INST: lsl	z31.s, p0/m, z31.s, #31
// CHECK-ENCODING: [0xff,0x83,0x43,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 83 43 04 <unknown>

lsl     z0.d, p0/m, z0.d, #0
// CHECK-INST: lsl	z0.d, p0/m, z0.d, #0
// CHECK-ENCODING: [0x00,0x80,0x83,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 80 83 04 <unknown>

lsl     z31.d, p0/m, z31.d, #63
// CHECK-INST: lsl	z31.d, p0/m, z31.d, #63
// CHECK-ENCODING: [0xff,0x83,0xc3,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 83 c3 04 <unknown>

lsl     z0.b, p0/m, z0.b, z0.b
// CHECK-INST: lsl	z0.b, p0/m, z0.b, z0.b
// CHECK-ENCODING: [0x00,0x80,0x13,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 80 13 04 <unknown>

lsl     z0.h, p0/m, z0.h, z0.h
// CHECK-INST: lsl	z0.h, p0/m, z0.h, z0.h
// CHECK-ENCODING: [0x00,0x80,0x53,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 80 53 04 <unknown>

lsl     z0.s, p0/m, z0.s, z0.s
// CHECK-INST: lsl	z0.s, p0/m, z0.s, z0.s
// CHECK-ENCODING: [0x00,0x80,0x93,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 80 93 04 <unknown>

lsl     z0.d, p0/m, z0.d, z0.d
// CHECK-INST: lsl	z0.d, p0/m, z0.d, z0.d
// CHECK-ENCODING: [0x00,0x80,0xd3,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 80 d3 04 <unknown>

lsl     z0.b, p0/m, z0.b, z1.d
// CHECK-INST: lsl	z0.b, p0/m, z0.b, z1.d
// CHECK-ENCODING: [0x20,0x80,0x1b,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 80 1b 04 <unknown>

lsl     z0.h, p0/m, z0.h, z1.d
// CHECK-INST: lsl	z0.h, p0/m, z0.h, z1.d
// CHECK-ENCODING: [0x20,0x80,0x5b,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 80 5b 04 <unknown>

lsl     z0.s, p0/m, z0.s, z1.d
// CHECK-INST: lsl	z0.s, p0/m, z0.s, z1.d
// CHECK-ENCODING: [0x20,0x80,0x9b,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 80 9b 04 <unknown>

lsl     z0.b, z1.b, z2.d
// CHECK-INST: lsl	z0.b, z1.b, z2.d
// CHECK-ENCODING: [0x20,0x8c,0x22,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 8c 22 04 <unknown>

lsl     z0.h, z1.h, z2.d
// CHECK-INST: lsl	z0.h, z1.h, z2.d
// CHECK-ENCODING: [0x20,0x8c,0x62,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 8c 62 04 <unknown>

lsl     z0.s, z1.s, z2.d
// CHECK-INST: lsl	z0.s, z1.s, z2.d
// CHECK-ENCODING: [0x20,0x8c,0xa2,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 8c a2 04 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z31.d, p0/z, z6.d
// CHECK-INST: movprfx	z31.d, p0/z, z6.d
// CHECK-ENCODING: [0xdf,0x20,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: df 20 d0 04 <unknown>

lsl     z31.d, p0/m, z31.d, #63
// CHECK-INST: lsl	z31.d, p0/m, z31.d, #63
// CHECK-ENCODING: [0xff,0x83,0xc3,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 83 c3 04 <unknown>

movprfx z31, z6
// CHECK-INST: movprfx	z31, z6
// CHECK-ENCODING: [0xdf,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: df bc 20 04 <unknown>

lsl     z31.d, p0/m, z31.d, #63
// CHECK-INST: lsl	z31.d, p0/m, z31.d, #63
// CHECK-ENCODING: [0xff,0x83,0xc3,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 83 c3 04 <unknown>

movprfx z0.s, p0/z, z7.s
// CHECK-INST: movprfx	z0.s, p0/z, z7.s
// CHECK-ENCODING: [0xe0,0x20,0x90,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 20 90 04 <unknown>

lsl     z0.s, p0/m, z0.s, z1.d
// CHECK-INST: lsl	z0.s, p0/m, z0.s, z1.d
// CHECK-ENCODING: [0x20,0x80,0x9b,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 80 9b 04 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 bc 20 04 <unknown>

lsl     z0.s, p0/m, z0.s, z1.d
// CHECK-INST: lsl	z0.s, p0/m, z0.s, z1.d
// CHECK-ENCODING: [0x20,0x80,0x9b,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 80 9b 04 <unknown>
