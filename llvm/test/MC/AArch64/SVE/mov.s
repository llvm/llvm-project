// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

mov     z0.b, w0
// CHECK-INST: mov     z0.b, w0
// CHECK-ENCODING: [0x00,0x38,0x20,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 38 20 05 <unknown>

mov     z0.h, w0
// CHECK-INST: mov     z0.h, w0
// CHECK-ENCODING: [0x00,0x38,0x60,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 38 60 05 <unknown>

mov     z0.s, w0
// CHECK-INST: mov     z0.s, w0
// CHECK-ENCODING: [0x00,0x38,0xa0,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 38 a0 05 <unknown>

mov     z0.d, x0
// CHECK-INST: mov     z0.d, x0
// CHECK-ENCODING: [0x00,0x38,0xe0,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 38 e0 05 <unknown>

mov     z31.h, wsp
// CHECK-INST: mov     z31.h, wsp
// CHECK-ENCODING: [0xff,0x3b,0x60,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 3b 60 05 <unknown>

mov     z31.s, wsp
// CHECK-INST: mov     z31.s, wsp
// CHECK-ENCODING: [0xff,0x3b,0xa0,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 3b a0 05 <unknown>

mov     z31.d, sp
// CHECK-INST: mov     z31.d, sp
// CHECK-ENCODING: [0xff,0x3b,0xe0,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 3b e0 05 <unknown>

mov     z31.b, wsp
// CHECK-INST: mov     z31.b, wsp
// CHECK-ENCODING: [0xff,0x3b,0x20,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 3b 20 05 <unknown>

mov     z0.d, z0.d
// CHECK-INST: mov     z0.d, z0.d
// CHECK-ENCODING: [0x00,0x30,0x60,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 30 60 04 <unknown>

mov     z31.d, z0.d
// CHECK-INST: mov     z31.d, z0.d
// CHECK-ENCODING: [0x1f,0x30,0x60,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 1f 30 60 04 <unknown>

mov     z5.b, #-128
// CHECK-INST: mov     z5.b, #-128
// CHECK-ENCODING: [0x05,0xd0,0x38,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 05 d0 38 25 <unknown>

mov     z5.b, #127
// CHECK-INST: mov     z5.b, #127
// CHECK-ENCODING: [0xe5,0xcf,0x38,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e5 cf 38 25 <unknown>

mov     z5.b, #255
// CHECK-INST: mov     z5.b, #-1
// CHECK-ENCODING: [0xe5,0xdf,0x38,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e5 df 38 25 <unknown>

mov     z21.h, #-128
// CHECK-INST: mov     z21.h, #-128
// CHECK-ENCODING: [0x15,0xd0,0x78,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 d0 78 25 <unknown>

mov     z21.h, #-128, lsl #8
// CHECK-INST: mov     z21.h, #-32768
// CHECK-ENCODING: [0x15,0xf0,0x78,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 f0 78 25 <unknown>

mov     z21.h, #-32768
// CHECK-INST: mov     z21.h, #-32768
// CHECK-ENCODING: [0x15,0xf0,0x78,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 f0 78 25 <unknown>

mov     z21.h, #127
// CHECK-INST: mov     z21.h, #127
// CHECK-ENCODING: [0xf5,0xcf,0x78,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: f5 cf 78 25 <unknown>

mov     z21.h, #127, lsl #8
// CHECK-INST: mov     z21.h, #32512
// CHECK-ENCODING: [0xf5,0xef,0x78,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: f5 ef 78 25 <unknown>

mov     z21.h, #32512
// CHECK-INST: mov     z21.h, #32512
// CHECK-ENCODING: [0xf5,0xef,0x78,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: f5 ef 78 25 <unknown>

mov     z21.s, #-128
// CHECK-INST: mov     z21.s, #-128
// CHECK-ENCODING: [0x15,0xd0,0xb8,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 d0 b8 25 <unknown>

mov     z21.s, #-128, lsl #8
// CHECK-INST: mov     z21.s, #-32768
// CHECK-ENCODING: [0x15,0xf0,0xb8,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 f0 b8 25 <unknown>

mov     z21.s, #-32768
// CHECK-INST: mov     z21.s, #-32768
// CHECK-ENCODING: [0x15,0xf0,0xb8,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 f0 b8 25 <unknown>

mov     z21.s, #127
// CHECK-INST: mov     z21.s, #127
// CHECK-ENCODING: [0xf5,0xcf,0xb8,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: f5 cf b8 25 <unknown>

mov     z21.s, #127, lsl #8
// CHECK-INST: mov     z21.s, #32512
// CHECK-ENCODING: [0xf5,0xef,0xb8,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: f5 ef b8 25 <unknown>

mov     z21.s, #32512
// CHECK-INST: mov     z21.s, #32512
// CHECK-ENCODING: [0xf5,0xef,0xb8,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: f5 ef b8 25 <unknown>

mov     z21.d, #-128
// CHECK-INST: mov     z21.d, #-128
// CHECK-ENCODING: [0x15,0xd0,0xf8,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 d0 f8 25 <unknown>

mov     z21.d, #-128, lsl #8
// CHECK-INST: mov     z21.d, #-32768
// CHECK-ENCODING: [0x15,0xf0,0xf8,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 f0 f8 25 <unknown>

mov     z21.d, #-32768
// CHECK-INST: mov     z21.d, #-32768
// CHECK-ENCODING: [0x15,0xf0,0xf8,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 f0 f8 25 <unknown>

mov     z21.d, #127
// CHECK-INST: mov     z21.d, #127
// CHECK-ENCODING: [0xf5,0xcf,0xf8,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: f5 cf f8 25 <unknown>

mov     z21.d, #127, lsl #8
// CHECK-INST: mov     z21.d, #32512
// CHECK-ENCODING: [0xf5,0xef,0xf8,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: f5 ef f8 25 <unknown>

mov     z21.d, #32512
// CHECK-INST: mov     z21.d, #32512
// CHECK-ENCODING: [0xf5,0xef,0xf8,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: f5 ef f8 25 <unknown>

mov     z0.h, #32768
// CHECK-INST: mov    z0.h, #-32768
// CHECK-ENCODING: [0x00,0xf0,0x78,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 f0 78 25 <unknown>

mov     z0.h, #65280
// CHECK-INST: mov    z0.h, #-256
// CHECK-ENCODING: [0xe0,0xff,0x78,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 ff 78 25 <unknown>

mov     z0.h, #-33024
// CHECK-INST: mov z0.h, #32512
// CHECK-ENCODING: [0xe0,0xef,0x78,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 ef 78 25 <unknown>

mov     z0.h, #-32769
// CHECK-INST: mov z0.h, #32767
// CHECK-ENCODING: [0xc0,0x05,0xc0,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c0 05 c0 05 <unknown>

mov     z0.s, #-32769
// CHECK-INST: mov     z0.s, #0xffff7fff
// CHECK-ENCODING: [0xc0,0x83,0xc0,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c0 83 c0 05 <unknown>

mov     z0.s, #32768
// CHECK-INST: mov     z0.s, #32768
// CHECK-ENCODING: [0x00,0x88,0xc0,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 88 c0 05 <unknown>

mov     z0.d, #-32769
// CHECK-INST: mov     z0.d, #0xffffffffffff7fff
// CHECK-ENCODING: [0xc0,0x87,0xc3,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c0 87 c3 05 <unknown>

mov     z0.d, #32768
// CHECK-INST: mov     z0.d, #32768
// CHECK-ENCODING: [0x00,0x88,0xc3,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 88 c3 05 <unknown>

mov     z0.d, #0xe0000000000003ff
// CHECK-INST: mov     z0.d, #0xe0000000000003ff
// CHECK-ENCODING: [0x80,0x19,0xc2,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 80 19 c2 05 <unknown>

mov     z5.b, p0/z, #-128
// CHECK-INST: mov     z5.b, p0/z, #-128
// CHECK-ENCODING: [0x05,0x10,0x10,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 05 10 10 05  <unknown>

mov     z5.b, p0/z, #127
// CHECK-INST: mov     z5.b, p0/z, #127
// CHECK-ENCODING: [0xe5,0x0f,0x10,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e5 0f 10 05  <unknown>

mov     z5.b, p0/z, #255
// CHECK-INST: mov     z5.b, p0/z, #-1
// CHECK-ENCODING: [0xe5,0x1f,0x10,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e5 1f 10 05  <unknown>

mov     z21.h, p0/z, #-128
// CHECK-INST: mov     z21.h, p0/z, #-128
// CHECK-ENCODING: [0x15,0x10,0x50,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 10 50 05  <unknown>

mov     z21.h, p0/z, #-128, lsl #8
// CHECK-INST: mov     z21.h, p0/z, #-32768
// CHECK-ENCODING: [0x15,0x30,0x50,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 30 50 05  <unknown>

mov     z21.h, p0/z, #-32768
// CHECK-INST: mov     z21.h, p0/z, #-32768
// CHECK-ENCODING: [0x15,0x30,0x50,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 30 50 05  <unknown>

mov     z21.h, p0/z, #127
// CHECK-INST: mov     z21.h, p0/z, #127
// CHECK-ENCODING: [0xf5,0x0f,0x50,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: f5 0f 50 05  <unknown>

mov     z21.h, p0/z, #127, lsl #8
// CHECK-INST: mov     z21.h, p0/z, #32512
// CHECK-ENCODING: [0xf5,0x2f,0x50,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: f5 2f 50 05  <unknown>

mov     z21.h, p0/z, #32512
// CHECK-INST: mov     z21.h, p0/z, #32512
// CHECK-ENCODING: [0xf5,0x2f,0x50,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: f5 2f 50 05  <unknown>

mov     z21.s, p0/z, #-128
// CHECK-INST: mov     z21.s, p0/z, #-128
// CHECK-ENCODING: [0x15,0x10,0x90,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 10 90 05  <unknown>

mov     z21.s, p0/z, #-128, lsl #8
// CHECK-INST: mov     z21.s, p0/z, #-32768
// CHECK-ENCODING: [0x15,0x30,0x90,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 30 90 05  <unknown>

mov     z21.s, p0/z, #-32768
// CHECK-INST: mov     z21.s, p0/z, #-32768
// CHECK-ENCODING: [0x15,0x30,0x90,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 30 90 05  <unknown>

mov     z21.s, p0/z, #127
// CHECK-INST: mov     z21.s, p0/z, #127
// CHECK-ENCODING: [0xf5,0x0f,0x90,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: f5 0f 90 05  <unknown>

mov     z21.s, p0/z, #127, lsl #8
// CHECK-INST: mov     z21.s, p0/z, #32512
// CHECK-ENCODING: [0xf5,0x2f,0x90,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: f5 2f 90 05  <unknown>

mov     z21.s, p0/z, #32512
// CHECK-INST: mov     z21.s, p0/z, #32512
// CHECK-ENCODING: [0xf5,0x2f,0x90,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: f5 2f 90 05  <unknown>

mov     z21.d, p0/z, #-128
// CHECK-INST: mov     z21.d, p0/z, #-128
// CHECK-ENCODING: [0x15,0x10,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 10 d0 05  <unknown>

mov     z21.d, p0/z, #-128, lsl #8
// CHECK-INST: mov     z21.d, p0/z, #-32768
// CHECK-ENCODING: [0x15,0x30,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 30 d0 05  <unknown>

mov     z21.d, p0/z, #-32768
// CHECK-INST: mov     z21.d, p0/z, #-32768
// CHECK-ENCODING: [0x15,0x30,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 30 d0 05  <unknown>

mov     z21.d, p0/z, #127
// CHECK-INST: mov     z21.d, p0/z, #127
// CHECK-ENCODING: [0xf5,0x0f,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: f5 0f d0 05  <unknown>

mov     z21.d, p0/z, #127, lsl #8
// CHECK-INST: mov     z21.d, p0/z, #32512
// CHECK-ENCODING: [0xf5,0x2f,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: f5 2f d0 05  <unknown>

mov     z21.d, p0/z, #32512
// CHECK-INST: mov     z21.d, p0/z, #32512
// CHECK-ENCODING: [0xf5,0x2f,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: f5 2f d0 05  <unknown>


// --------------------------------------------------------------------------//
// Tests where the negative immediate is in bounds when interpreted
// as the element type.

mov     z0.b, #-129
// CHECK-INST: mov     z0.b, #127
// CHECK-ENCODING: [0xe0,0xcf,0x38,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 cf 38 25 <unknown>

mov     z0.h, #-129, lsl #8
// CHECK-INST: mov     z0.h, #32512
// CHECK-ENCODING: [0xe0,0xef,0x78,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 ef 78 25 <unknown>

mov     z5.h, #0xfffa
// CHECK-INST: mov     z5.h, #-6
// CHECK-ENCODING: [0x45,0xdf,0x78,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 45 df 78 25 <unknown>

mov     z5.s, #0xfffffffa
// CHECK-INST: mov     z5.s, #-6
// CHECK-ENCODING: [0x45,0xdf,0xb8,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 45 df b8 25 <unknown>

mov     z5.d, #0xfffffffffffffffa
// CHECK-INST: mov     z5.d, #-6
// CHECK-ENCODING: [0x45,0xdf,0xf8,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 45 df f8 25 <unknown>

mov     z0.b, p0/z, #-129
// CHECK-INST: mov     z0.b, p0/z, #127
// CHECK-ENCODING: [0xe0,0x0f,0x10,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 0f 10 05 <unknown>

mov     z0.h, p0/z, #-33024
// CHECK-INST: mov     z0.h, p0/z, #32512
// CHECK-ENCODING: [0xe0,0x2f,0x50,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 2f 50 05 <unknown>

mov     z0.h, p0/z, #-129, lsl #8
// CHECK-INST: mov     z0.h, p0/z, #32512
// CHECK-ENCODING: [0xe0,0x2f,0x50,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 2f 50 05 <unknown>

// --------------------------------------------------------------------------//
// Tests for merging variant (/m) and testing the range of predicate (> 7)
// is allowed.

mov     z5.b, p15/m, #-128
// CHECK-INST: mov     z5.b, p15/m, #-128
// CHECK-ENCODING: [0x05,0x50,0x1f,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 05 50 1f 05  <unknown>

mov     z21.h, p15/m, #-128
// CHECK-INST: mov     z21.h, p15/m, #-128
// CHECK-ENCODING: [0x15,0x50,0x5f,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 50 5f 05  <unknown>

mov     z21.h, p15/m, #-128, lsl #8
// CHECK-INST: mov     z21.h, p15/m, #-32768
// CHECK-ENCODING: [0x15,0x70,0x5f,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 70 5f 05  <unknown>

mov     z21.s, p15/m, #-128
// CHECK-INST: mov     z21.s, p15/m, #-128
// CHECK-ENCODING: [0x15,0x50,0x9f,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 50 9f 05  <unknown>

mov     z21.s, p15/m, #-128, lsl #8
// CHECK-INST: mov     z21.s, p15/m, #-32768
// CHECK-ENCODING: [0x15,0x70,0x9f,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 70 9f 05  <unknown>

mov     z21.d, p15/m, #-128
// CHECK-INST: mov     z21.d, p15/m, #-128
// CHECK-ENCODING: [0x15,0x50,0xdf,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 50 df 05  <unknown>

mov     z21.d, p15/m, #-128, lsl #8
// CHECK-INST: mov     z21.d, p15/m, #-32768
// CHECK-ENCODING: [0x15,0x70,0xdf,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 70 df 05  <unknown>

// --------------------------------------------------------------------------//
// Tests for indexed variant

mov     z0.b, z0.b[0]
// CHECK-INST: mov     z0.b, b0
// CHECK-ENCODING: [0x00,0x20,0x21,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 20 21 05 <unknown>

mov     z0.h, z0.h[0]
// CHECK-INST: mov     z0.h, h0
// CHECK-ENCODING: [0x00,0x20,0x22,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 20 22 05 <unknown>

mov     z0.s, z0.s[0]
// CHECK-INST: mov     z0.s, s0
// CHECK-ENCODING: [0x00,0x20,0x24,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 20 24 05 <unknown>

mov     z0.d, z0.d[0]
// CHECK-INST: mov     z0.d, d0
// CHECK-ENCODING: [0x00,0x20,0x28,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 20 28 05 <unknown>

mov     z0.q, z0.q[0]
// CHECK-INST: mov     z0.q, q0
// CHECK-ENCODING: [0x00,0x20,0x30,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 20 30 05 <unknown>

mov     z0.b, b0
// CHECK-INST: mov     z0.b, b0
// CHECK-ENCODING: [0x00,0x20,0x21,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 20 21 05 <unknown>

mov     z0.h, h0
// CHECK-INST: mov     z0.h, h0
// CHECK-ENCODING: [0x00,0x20,0x22,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 20 22 05 <unknown>

mov     z0.s, s0
// CHECK-INST: mov     z0.s, s0
// CHECK-ENCODING: [0x00,0x20,0x24,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 20 24 05 <unknown>

mov     z0.d, d0
// CHECK-INST: mov     z0.d, d0
// CHECK-ENCODING: [0x00,0x20,0x28,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 20 28 05 <unknown>

mov     z0.q, q0
// CHECK-INST: mov     z0.q, q0
// CHECK-ENCODING: [0x00,0x20,0x30,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 20 30 05 <unknown>

mov     z31.b, z31.b[63]
// CHECK-INST: mov     z31.b, z31.b[63]
// CHECK-ENCODING: [0xff,0x23,0xff,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 23 ff 05 <unknown>

mov     z31.h, z31.h[31]
// CHECK-INST: mov     z31.h, z31.h[31]
// CHECK-ENCODING: [0xff,0x23,0xfe,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 23 fe 05 <unknown>

mov     z31.s, z31.s[15]
// CHECK-INST: mov     z31.s, z31.s[15]
// CHECK-ENCODING: [0xff,0x23,0xfc,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 23 fc 05 <unknown>

mov     z31.d, z31.d[7]
// CHECK-INST: mov     z31.d, z31.d[7]
// CHECK-ENCODING: [0xff,0x23,0xf8,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 23 f8 05 <unknown>

mov     z5.q, z17.q[3]
// CHECK-INST: mov     z5.q, z17.q[3]
// CHECK-ENCODING: [0x25,0x22,0xf0,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 25 22 f0 05 <unknown>


// --------------------------------------------------------------------------//
// Tests for predicated copy of SIMD/FP registers.

mov     z0.b, p0/m, w0
// CHECK-INST: mov     z0.b, p0/m, w0
// CHECK-ENCODING: [0x00,0xa0,0x28,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 28 05 <unknown>

mov     z0.h, p0/m, w0
// CHECK-INST: mov     z0.h, p0/m, w0
// CHECK-ENCODING: [0x00,0xa0,0x68,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 68 05 <unknown>

mov     z0.s, p0/m, w0
// CHECK-INST: mov     z0.s, p0/m, w0
// CHECK-ENCODING: [0x00,0xa0,0xa8,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 a8 05 <unknown>

mov     z0.d, p0/m, x0
// CHECK-INST: mov     z0.d, p0/m, x0
// CHECK-ENCODING: [0x00,0xa0,0xe8,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 e8 05 <unknown>

mov     z31.b, p7/m, wsp
// CHECK-INST: mov     z31.b, p7/m, wsp
// CHECK-ENCODING: [0xff,0xbf,0x28,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf 28 05 <unknown>

mov     z31.h, p7/m, wsp
// CHECK-INST: mov     z31.h, p7/m, wsp
// CHECK-ENCODING: [0xff,0xbf,0x68,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf 68 05 <unknown>

mov     z31.s, p7/m, wsp
// CHECK-INST: mov     z31.s, p7/m, wsp
// CHECK-ENCODING: [0xff,0xbf,0xa8,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf a8 05 <unknown>

mov     z31.d, p7/m, sp
// CHECK-INST: mov     z31.d, p7/m, sp
// CHECK-ENCODING: [0xff,0xbf,0xe8,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf e8 05 <unknown>

mov     z0.b, p0/m, b0
// CHECK-INST: mov     z0.b, p0/m, b0
// CHECK-ENCODING: [0x00,0x80,0x20,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 20 05 <unknown>

mov     z31.b, p7/m, b31
// CHECK-INST: mov     z31.b, p7/m, b31
// CHECK-ENCODING: [0xff,0x9f,0x20,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 9f 20 05 <unknown>

mov     z0.h, p0/m, h0
// CHECK-INST: mov     z0.h, p0/m, h0
// CHECK-ENCODING: [0x00,0x80,0x60,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 60 05 <unknown>

mov     z31.h, p7/m, h31
// CHECK-INST: mov     z31.h, p7/m, h31
// CHECK-ENCODING: [0xff,0x9f,0x60,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 9f 60 05 <unknown>

mov     z0.s, p0/m, s0
// CHECK-INST: mov     z0.s, p0/m, s0
// CHECK-ENCODING: [0x00,0x80,0xa0,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 a0 05 <unknown>

mov     z31.s, p7/m, s31
// CHECK-INST: mov     z31.s, p7/m, s31
// CHECK-ENCODING: [0xff,0x9f,0xa0,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 9f a0 05 <unknown>

mov     z0.d, p0/m, d0
// CHECK-INST: mov     z0.d, p0/m, d0
// CHECK-ENCODING: [0x00,0x80,0xe0,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 e0 05 <unknown>

mov     z31.d, p7/m, d31
// CHECK-INST: mov     z31.d, p7/m, d31
// CHECK-ENCODING: [0xff,0x9f,0xe0,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 9f e0 05 <unknown>

mov     p0.b, p0/m, p0.b
// CHECK-INST: mov     p0.b, p0/m, p0.b
// CHECK-ENCODING: [0x10,0x42,0x00,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 10 42 00 25 <unknown>

mov     p15.b, p15/m, p15.b
// CHECK-INST: mov     p15.b, p15/m, p15.b
// CHECK-ENCODING: [0xff,0x7f,0x0f,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 7f 0f 25 <unknown>

mov     z31.b, p15/m, z31.b
// CHECK-INST: mov     z31.b, p15/m, z31.b
// CHECK-ENCODING: [0xff,0xff,0x3f,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff ff 3f 05 <unknown>

mov     z31.h, p15/m, z31.h
// CHECK-INST: mov     z31.h, p15/m, z31.h
// CHECK-ENCODING: [0xff,0xff,0x7f,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff ff 7f 05 <unknown>

mov     z31.s, p15/m, z31.s
// CHECK-INST: mov     z31.s, p15/m, z31.s
// CHECK-ENCODING: [0xff,0xff,0xbf,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff ff bf 05 <unknown>

mov     z31.d, p15/m, z31.d
// CHECK-INST: mov     z31.d, p15/m, z31.d
// CHECK-ENCODING: [0xff,0xff,0xff,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff ff ff 05 <unknown>

mov     p0.b, p0.b
// CHECK-INST: mov     p0.b, p0.b
// CHECK-ENCODING: [0x00,0x40,0x80,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 40 80 25 <unknown>

mov     p15.b, p15.b
// CHECK-INST: mov     p15.b, p15.b
// CHECK-ENCODING: [0xef,0x7d,0x8f,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ef 7d 8f 25 <unknown>

mov     p0.b, p0/z, p0.b
// CHECK-INST: mov     p0.b, p0/z, p0.b
// CHECK-ENCODING: [0x00,0x40,0x00,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 40 00 25 <unknown>

mov     p15.b, p15/z, p15.b
// CHECK-INST: mov     p15.b, p15/z, p15.b
// CHECK-ENCODING: [0xef,0x7d,0x0f,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ef 7d 0f 25 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z31.d, p7/z, z6.d
// CHECK-INST: movprfx	z31.d, p7/z, z6.d
// CHECK-ENCODING: [0xdf,0x3c,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: df 3c d0 04 <unknown>

mov     z31.d, p7/m, sp
// CHECK-INST: mov	z31.d, p7/m, sp
// CHECK-ENCODING: [0xff,0xbf,0xe8,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf e8 05 <unknown>

movprfx z31, z6
// CHECK-INST: movprfx	z31, z6
// CHECK-ENCODING: [0xdf,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: df bc 20 04 <unknown>

mov     z31.d, p7/m, sp
// CHECK-INST: mov	z31.d, p7/m, sp
// CHECK-ENCODING: [0xff,0xbf,0xe8,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf e8 05 <unknown>

movprfx z21.d, p7/z, z28.d
// CHECK-INST: movprfx	z21.d, p7/z, z28.d
// CHECK-ENCODING: [0x95,0x3f,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 95 3f d0 04 <unknown>

mov     z21.d, p7/m, #-128, lsl #8
// CHECK-INST: mov	z21.d, p7/m, #-32768
// CHECK-ENCODING: [0x15,0x70,0xd7,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 70 d7 05 <unknown>

movprfx z21, z28
// CHECK-INST: movprfx	z21, z28
// CHECK-ENCODING: [0x95,0xbf,0x20,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 95 bf 20 04 <unknown>

mov     z21.d, p15/m, #-128, lsl #8
// CHECK-INST: mov	z21.d, p15/m, #-32768
// CHECK-ENCODING: [0x15,0x70,0xdf,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 70 df 05 <unknown>

movprfx z4.d, p7/z, z6.d
// CHECK-INST: movprfx	z4.d, p7/z, z6.d
// CHECK-ENCODING: [0xc4,0x3c,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4 3c d0 04 <unknown>

mov     z4.d, p7/m, d31
// CHECK-INST: mov	z4.d, p7/m, d31
// CHECK-ENCODING: [0xe4,0x9f,0xe0,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e4 9f e0 05 <unknown>

movprfx z4, z6
// CHECK-INST: movprfx	z4, z6
// CHECK-ENCODING: [0xc4,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4 bc 20 04 <unknown>

mov     z4.d, p7/m, d31
// CHECK-INST: mov	z4.d, p7/m, d31
// CHECK-ENCODING: [0xe4,0x9f,0xe0,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e4 9f e0 05 <unknown>
