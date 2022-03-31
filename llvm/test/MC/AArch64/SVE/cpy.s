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

cpy     z0.b, p0/m, w0
// CHECK-INST: mov     z0.b, p0/m, w0
// CHECK-ENCODING: [0x00,0xa0,0x28,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 a0 28 05 <unknown>

cpy     z0.h, p0/m, w0
// CHECK-INST: mov     z0.h, p0/m, w0
// CHECK-ENCODING: [0x00,0xa0,0x68,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 a0 68 05 <unknown>

cpy     z0.s, p0/m, w0
// CHECK-INST: mov     z0.s, p0/m, w0
// CHECK-ENCODING: [0x00,0xa0,0xa8,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 a0 a8 05 <unknown>

cpy     z0.d, p0/m, x0
// CHECK-INST: mov     z0.d, p0/m, x0
// CHECK-ENCODING: [0x00,0xa0,0xe8,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 a0 e8 05 <unknown>

cpy     z31.b, p7/m, wsp
// CHECK-INST: mov     z31.b, p7/m, wsp
// CHECK-ENCODING: [0xff,0xbf,0x28,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff bf 28 05 <unknown>

cpy     z31.h, p7/m, wsp
// CHECK-INST: mov     z31.h, p7/m, wsp
// CHECK-ENCODING: [0xff,0xbf,0x68,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff bf 68 05 <unknown>

cpy     z31.s, p7/m, wsp
// CHECK-INST: mov     z31.s, p7/m, wsp
// CHECK-ENCODING: [0xff,0xbf,0xa8,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff bf a8 05 <unknown>

cpy     z31.d, p7/m, sp
// CHECK-INST: mov     z31.d, p7/m, sp
// CHECK-ENCODING: [0xff,0xbf,0xe8,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff bf e8 05 <unknown>

cpy     z0.b, p0/m, b0
// CHECK-INST: mov     z0.b, p0/m, b0
// CHECK-ENCODING: [0x00,0x80,0x20,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 80 20 05 <unknown>

cpy     z31.b, p7/m, b31
// CHECK-INST: mov     z31.b, p7/m, b31
// CHECK-ENCODING: [0xff,0x9f,0x20,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 9f 20 05 <unknown>

cpy     z0.h, p0/m, h0
// CHECK-INST: mov     z0.h, p0/m, h0
// CHECK-ENCODING: [0x00,0x80,0x60,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 80 60 05 <unknown>

cpy     z31.h, p7/m, h31
// CHECK-INST: mov     z31.h, p7/m, h31
// CHECK-ENCODING: [0xff,0x9f,0x60,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 9f 60 05 <unknown>

cpy     z0.s, p0/m, s0
// CHECK-INST: mov     z0.s, p0/m, s0
// CHECK-ENCODING: [0x00,0x80,0xa0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 80 a0 05 <unknown>

cpy     z31.s, p7/m, s31
// CHECK-INST: mov     z31.s, p7/m, s31
// CHECK-ENCODING: [0xff,0x9f,0xa0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 9f a0 05 <unknown>

cpy     z0.d, p0/m, d0
// CHECK-INST: mov     z0.d, p0/m, d0
// CHECK-ENCODING: [0x00,0x80,0xe0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 80 e0 05 <unknown>

cpy     z31.d, p7/m, d31
// CHECK-INST: mov     z31.d, p7/m, d31
// CHECK-ENCODING: [0xff,0x9f,0xe0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 9f e0 05 <unknown>

cpy     z5.b, p0/z, #-128
// CHECK-INST: mov     z5.b, p0/z, #-128
// CHECK-ENCODING: [0x05,0x10,0x10,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05 10 10 05  <unknown>

cpy     z5.b, p0/z, #127
// CHECK-INST: mov     z5.b, p0/z, #127
// CHECK-ENCODING: [0xe5,0x0f,0x10,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e5 0f 10 05  <unknown>

cpy     z5.b, p0/z, #255
// CHECK-INST: mov     z5.b, p0/z, #-1
// CHECK-ENCODING: [0xe5,0x1f,0x10,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e5 1f 10 05  <unknown>

cpy     z21.h, p0/z, #-128
// CHECK-INST: mov     z21.h, p0/z, #-128
// CHECK-ENCODING: [0x15,0x10,0x50,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 15 10 50 05  <unknown>

cpy     z21.h, p0/z, #-128, lsl #8
// CHECK-INST: mov     z21.h, p0/z, #-32768
// CHECK-ENCODING: [0x15,0x30,0x50,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 15 30 50 05  <unknown>

cpy     z21.h, p0/z, #-32768
// CHECK-INST: mov     z21.h, p0/z, #-32768
// CHECK-ENCODING: [0x15,0x30,0x50,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 15 30 50 05  <unknown>

cpy     z21.h, p0/z, #127
// CHECK-INST: mov     z21.h, p0/z, #127
// CHECK-ENCODING: [0xf5,0x0f,0x50,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: f5 0f 50 05  <unknown>

cpy     z21.h, p0/z, #127, lsl #8
// CHECK-INST: mov     z21.h, p0/z, #32512
// CHECK-ENCODING: [0xf5,0x2f,0x50,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: f5 2f 50 05  <unknown>

cpy     z21.h, p0/z, #32512
// CHECK-INST: mov     z21.h, p0/z, #32512
// CHECK-ENCODING: [0xf5,0x2f,0x50,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: f5 2f 50 05  <unknown>

cpy     z21.s, p0/z, #-128
// CHECK-INST: mov     z21.s, p0/z, #-128
// CHECK-ENCODING: [0x15,0x10,0x90,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 15 10 90 05  <unknown>

cpy     z21.s, p0/z, #-128, lsl #8
// CHECK-INST: mov     z21.s, p0/z, #-32768
// CHECK-ENCODING: [0x15,0x30,0x90,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 15 30 90 05  <unknown>

cpy     z21.s, p0/z, #-32768
// CHECK-INST: mov     z21.s, p0/z, #-32768
// CHECK-ENCODING: [0x15,0x30,0x90,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 15 30 90 05  <unknown>

cpy     z21.s, p0/z, #127
// CHECK-INST: mov     z21.s, p0/z, #127
// CHECK-ENCODING: [0xf5,0x0f,0x90,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: f5 0f 90 05  <unknown>

cpy     z21.s, p0/z, #127, lsl #8
// CHECK-INST: mov     z21.s, p0/z, #32512
// CHECK-ENCODING: [0xf5,0x2f,0x90,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: f5 2f 90 05  <unknown>

cpy     z21.s, p0/z, #32512
// CHECK-INST: mov     z21.s, p0/z, #32512
// CHECK-ENCODING: [0xf5,0x2f,0x90,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: f5 2f 90 05  <unknown>

cpy     z21.d, p0/z, #-128
// CHECK-INST: mov     z21.d, p0/z, #-128
// CHECK-ENCODING: [0x15,0x10,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 15 10 d0 05  <unknown>

cpy     z21.d, p0/z, #-128, lsl #8
// CHECK-INST: mov     z21.d, p0/z, #-32768
// CHECK-ENCODING: [0x15,0x30,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 15 30 d0 05  <unknown>

cpy     z21.d, p0/z, #-32768
// CHECK-INST: mov     z21.d, p0/z, #-32768
// CHECK-ENCODING: [0x15,0x30,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 15 30 d0 05  <unknown>

cpy     z21.d, p0/z, #127
// CHECK-INST: mov     z21.d, p0/z, #127
// CHECK-ENCODING: [0xf5,0x0f,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: f5 0f d0 05  <unknown>

cpy     z21.d, p0/z, #127, lsl #8
// CHECK-INST: mov     z21.d, p0/z, #32512
// CHECK-ENCODING: [0xf5,0x2f,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: f5 2f d0 05  <unknown>

cpy     z21.d, p0/z, #32512
// CHECK-INST: mov     z21.d, p0/z, #32512
// CHECK-ENCODING: [0xf5,0x2f,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: f5 2f d0 05  <unknown>

// --------------------------------------------------------------------------//
// Tests where the negative immediate is in bounds when interpreted
// as the element type.

cpy z0.b, p0/z, #-129
// CHECK-INST: mov     z0.b, p0/z, #127
// CHECK-ENCODING: [0xe0,0x0f,0x10,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 0f 10 05  <unknown>

cpy z0.h, p0/z, #-33024
// CHECK-INST: mov     z0.h, p0/z, #32512
// CHECK-ENCODING: [0xe0,0x2f,0x50,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 2f 50 05  <unknown>

cpy z0.h, p0/z, #-129, lsl #8
// CHECK-INST: mov     z0.h, p0/z, #32512
// CHECK-ENCODING: [0xe0,0x2f,0x50,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 2f 50 05  <unknown>


// --------------------------------------------------------------------------//
// Tests for merging variant (/m) and testing the range of predicate (> 7)
// is allowed.

cpy     z5.b, p15/m, #-128
// CHECK-INST: mov     z5.b, p15/m, #-128
// CHECK-ENCODING: [0x05,0x50,0x1f,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05 50 1f 05  <unknown>

cpy     z21.h, p15/m, #-128
// CHECK-INST: mov     z21.h, p15/m, #-128
// CHECK-ENCODING: [0x15,0x50,0x5f,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 15 50 5f 05  <unknown>

cpy     z21.h, p15/m, #-128, lsl #8
// CHECK-INST: mov     z21.h, p15/m, #-32768
// CHECK-ENCODING: [0x15,0x70,0x5f,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 15 70 5f 05  <unknown>

cpy     z21.s, p15/m, #-128
// CHECK-INST: mov     z21.s, p15/m, #-128
// CHECK-ENCODING: [0x15,0x50,0x9f,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 15 50 9f 05  <unknown>

cpy     z21.s, p15/m, #-128, lsl #8
// CHECK-INST: mov     z21.s, p15/m, #-32768
// CHECK-ENCODING: [0x15,0x70,0x9f,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 15 70 9f 05  <unknown>

cpy     z21.d, p15/m, #-128
// CHECK-INST: mov     z21.d, p15/m, #-128
// CHECK-ENCODING: [0x15,0x50,0xdf,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 15 50 df 05  <unknown>

cpy     z21.d, p15/m, #-128, lsl #8
// CHECK-INST: mov     z21.d, p15/m, #-32768
// CHECK-ENCODING: [0x15,0x70,0xdf,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 15 70 df 05  <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z31.d, p7/z, z6.d
// CHECK-INST: movprfx	z31.d, p7/z, z6.d
// CHECK-ENCODING: [0xdf,0x3c,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: df 3c d0 04 <unknown>

cpy     z31.d, p7/m, sp
// CHECK-INST: mov	z31.d, p7/m, sp
// CHECK-ENCODING: [0xff,0xbf,0xe8,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff bf e8 05 <unknown>

movprfx z31, z6
// CHECK-INST: movprfx	z31, z6
// CHECK-ENCODING: [0xdf,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: df bc 20 04 <unknown>

cpy     z31.d, p7/m, sp
// CHECK-INST: mov	z31.d, p7/m, sp
// CHECK-ENCODING: [0xff,0xbf,0xe8,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff bf e8 05 <unknown>

movprfx z21.d, p7/z, z28.d
// CHECK-INST: movprfx	z21.d, p7/z, z28.d
// CHECK-ENCODING: [0x95,0x3f,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 95 3f d0 04 <unknown>

cpy     z21.d, p7/m, #-128, lsl #8
// CHECK-INST: mov	z21.d, p7/m, #-32768
// CHECK-ENCODING: [0x15,0x70,0xd7,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 15 70 d7 05 <unknown>

movprfx z21, z28
// CHECK-INST: movprfx	z21, z28
// CHECK-ENCODING: [0x95,0xbf,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 95 bf 20 04 <unknown>

cpy     z21.d, p15/m, #-128, lsl #8
// CHECK-INST: mov	z21.d, p15/m, #-32768
// CHECK-ENCODING: [0x15,0x70,0xdf,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 15 70 df 05 <unknown>

movprfx z4.d, p7/z, z6.d
// CHECK-INST: movprfx	z4.d, p7/z, z6.d
// CHECK-ENCODING: [0xc4,0x3c,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c4 3c d0 04 <unknown>

cpy     z4.d, p7/m, d31
// CHECK-INST: mov	z4.d, p7/m, d31
// CHECK-ENCODING: [0xe4,0x9f,0xe0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e4 9f e0 05 <unknown>

movprfx z4, z6
// CHECK-INST: movprfx	z4, z6
// CHECK-ENCODING: [0xc4,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c4 bc 20 04 <unknown>

cpy     z4.d, p7/m, d31
// CHECK-INST: mov	z4.d, p7/m, d31
// CHECK-ENCODING: [0xe4,0x9f,0xe0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e4 9f e0 05 <unknown>
