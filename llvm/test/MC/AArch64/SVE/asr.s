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

asr     z0.b, z0.b, #1
// CHECK-INST: asr	z0.b, z0.b, #1
// CHECK-ENCODING: [0x00,0x90,0x2f,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 90 2f 04 <unknown>

asr     z31.b, z31.b, #8
// CHECK-INST: asr	z31.b, z31.b, #8
// CHECK-ENCODING: [0xff,0x93,0x28,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: ff 93 28 04 <unknown>

asr     z0.h, z0.h, #1
// CHECK-INST: asr	z0.h, z0.h, #1
// CHECK-ENCODING: [0x00,0x90,0x3f,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 90 3f 04 <unknown>

asr     z31.h, z31.h, #16
// CHECK-INST: asr	z31.h, z31.h, #16
// CHECK-ENCODING: [0xff,0x93,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: ff 93 30 04 <unknown>

asr     z0.s, z0.s, #1
// CHECK-INST: asr	z0.s, z0.s, #1
// CHECK-ENCODING: [0x00,0x90,0x7f,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 90 7f 04 <unknown>

asr     z31.s, z31.s, #32
// CHECK-INST: asr	z31.s, z31.s, #32
// CHECK-ENCODING: [0xff,0x93,0x60,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: ff 93 60 04 <unknown>

asr     z0.d, z0.d, #1
// CHECK-INST: asr	z0.d, z0.d, #1
// CHECK-ENCODING: [0x00,0x90,0xff,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 90 ff 04 <unknown>

asr     z31.d, z31.d, #64
// CHECK-INST: asr	z31.d, z31.d, #64
// CHECK-ENCODING: [0xff,0x93,0xa0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: ff 93 a0 04 <unknown>

asr     z0.b, p0/m, z0.b, #1
// CHECK-INST: asr	z0.b, p0/m, z0.b, #1
// CHECK-ENCODING: [0xe0,0x81,0x00,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 81 00 04 <unknown>

asr     z31.b, p0/m, z31.b, #8
// CHECK-INST: asr	z31.b, p0/m, z31.b, #8
// CHECK-ENCODING: [0x1f,0x81,0x00,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 1f 81 00 04 <unknown>

asr     z0.h, p0/m, z0.h, #1
// CHECK-INST: asr	z0.h, p0/m, z0.h, #1
// CHECK-ENCODING: [0xe0,0x83,0x00,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 83 00 04 <unknown>

asr     z31.h, p0/m, z31.h, #16
// CHECK-INST: asr	z31.h, p0/m, z31.h, #16
// CHECK-ENCODING: [0x1f,0x82,0x00,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 1f 82 00 04 <unknown>

asr     z0.s, p0/m, z0.s, #1
// CHECK-INST: asr	z0.s, p0/m, z0.s, #1
// CHECK-ENCODING: [0xe0,0x83,0x40,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 83 40 04 <unknown>

asr     z31.s, p0/m, z31.s, #32
// CHECK-INST: asr	z31.s, p0/m, z31.s, #32
// CHECK-ENCODING: [0x1f,0x80,0x40,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 1f 80 40 04 <unknown>

asr     z0.d, p0/m, z0.d, #1
// CHECK-INST: asr	z0.d, p0/m, z0.d, #1
// CHECK-ENCODING: [0xe0,0x83,0xc0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 83 c0 04 <unknown>

asr     z31.d, p0/m, z31.d, #64
// CHECK-INST: asr	z31.d, p0/m, z31.d, #64
// CHECK-ENCODING: [0x1f,0x80,0x80,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 1f 80 80 04 <unknown>

asr     z0.b, p0/m, z0.b, z0.b
// CHECK-INST: asr	z0.b, p0/m, z0.b, z0.b
// CHECK-ENCODING: [0x00,0x80,0x10,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 80 10 04 <unknown>

asr     z0.h, p0/m, z0.h, z0.h
// CHECK-INST: asr	z0.h, p0/m, z0.h, z0.h
// CHECK-ENCODING: [0x00,0x80,0x50,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 80 50 04 <unknown>

asr     z0.s, p0/m, z0.s, z0.s
// CHECK-INST: asr	z0.s, p0/m, z0.s, z0.s
// CHECK-ENCODING: [0x00,0x80,0x90,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 80 90 04 <unknown>

asr     z0.d, p0/m, z0.d, z0.d
// CHECK-INST: asr	z0.d, p0/m, z0.d, z0.d
// CHECK-ENCODING: [0x00,0x80,0xd0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 80 d0 04 <unknown>

asr     z0.b, p0/m, z0.b, z1.d
// CHECK-INST: asr	z0.b, p0/m, z0.b, z1.d
// CHECK-ENCODING: [0x20,0x80,0x18,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 20 80 18 04 <unknown>

asr     z0.h, p0/m, z0.h, z1.d
// CHECK-INST: asr	z0.h, p0/m, z0.h, z1.d
// CHECK-ENCODING: [0x20,0x80,0x58,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 20 80 58 04 <unknown>

asr     z0.s, p0/m, z0.s, z1.d
// CHECK-INST: asr	z0.s, p0/m, z0.s, z1.d
// CHECK-ENCODING: [0x20,0x80,0x98,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 20 80 98 04 <unknown>

asr     z0.b, z1.b, z2.d
// CHECK-INST: asr	z0.b, z1.b, z2.d
// CHECK-ENCODING: [0x20,0x80,0x22,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 20 80 22 04 <unknown>

asr     z0.h, z1.h, z2.d
// CHECK-INST: asr	z0.h, z1.h, z2.d
// CHECK-ENCODING: [0x20,0x80,0x62,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 20 80 62 04 <unknown>

asr     z0.s, z1.s, z2.d
// CHECK-INST: asr	z0.s, z1.s, z2.d
// CHECK-ENCODING: [0x20,0x80,0xa2,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 20 80 a2 04 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z31.d, p0/z, z6.d
// CHECK-INST: movprfx	z31.d, p0/z, z6.d
// CHECK-ENCODING: [0xdf,0x20,0xd0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: df 20 d0 04 <unknown>

asr     z31.d, p0/m, z31.d, #64
// CHECK-INST: asr	z31.d, p0/m, z31.d, #64
// CHECK-ENCODING: [0x1f,0x80,0x80,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 1f 80 80 04 <unknown>

movprfx z31, z6
// CHECK-INST: movprfx	z31, z6
// CHECK-ENCODING: [0xdf,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: df bc 20 04 <unknown>

asr     z31.d, p0/m, z31.d, #64
// CHECK-INST: asr	z31.d, p0/m, z31.d, #64
// CHECK-ENCODING: [0x1f,0x80,0x80,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 1f 80 80 04 <unknown>

movprfx z0.s, p0/z, z7.s
// CHECK-INST: movprfx	z0.s, p0/z, z7.s
// CHECK-ENCODING: [0xe0,0x20,0x90,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 20 90 04 <unknown>

asr     z0.s, p0/m, z0.s, z1.d
// CHECK-INST: asr	z0.s, p0/m, z0.s, z1.d
// CHECK-ENCODING: [0x20,0x80,0x98,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 20 80 98 04 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 bc 20 04 <unknown>

asr     z0.s, p0/m, z0.s, z1.d
// CHECK-INST: asr	z0.s, p0/m, z0.s, z1.d
// CHECK-ENCODING: [0x20,0x80,0x98,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 20 80 98 04 <unknown>
