// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme < %s \
// RUN:        | llvm-objdump -d --mattr=+sme - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

// --------------------------------------------------------------------------//
// Horizontal

st1w    {za0h.s[w12, 0]}, p0, [x0, x0, lsl #2]
// CHECK-INST: st1w    {za0h.s[w12, 0]}, p0, [x0, x0, lsl #2]
// CHECK-ENCODING: [0x00,0x00,0xa0,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 00 a0 e0 <unknown>

st1w    {za1h.s[w14, 1]}, p5, [x10, x21, lsl #2]
// CHECK-INST: st1w    {za1h.s[w14, 1]}, p5, [x10, x21, lsl #2]
// CHECK-ENCODING: [0x45,0x55,0xb5,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 55 b5 e0 <unknown>

st1w    {za1h.s[w15, 3]}, p3, [x13, x8, lsl #2]
// CHECK-INST: st1w    {za1h.s[w15, 3]}, p3, [x13, x8, lsl #2]
// CHECK-ENCODING: [0xa7,0x6d,0xa8,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 6d a8 e0 <unknown>

st1w    {za3h.s[w15, 3]}, p7, [sp]
// CHECK-INST: st1w    {za3h.s[w15, 3]}, p7, [sp]
// CHECK-ENCODING: [0xef,0x7f,0xbf,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef 7f bf e0 <unknown>

st1w    {za1h.s[w12, 1]}, p3, [x17, x16, lsl #2]
// CHECK-INST: st1w    {za1h.s[w12, 1]}, p3, [x17, x16, lsl #2]
// CHECK-ENCODING: [0x25,0x0e,0xb0,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 0e b0 e0 <unknown>

st1w    {za0h.s[w12, 1]}, p1, [x1, x30, lsl #2]
// CHECK-INST: st1w    {za0h.s[w12, 1]}, p1, [x1, x30, lsl #2]
// CHECK-ENCODING: [0x21,0x04,0xbe,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 04 be e0 <unknown>

st1w    {za2h.s[w14, 0]}, p5, [x19, x20, lsl #2]
// CHECK-INST: st1w    {za2h.s[w14, 0]}, p5, [x19, x20, lsl #2]
// CHECK-ENCODING: [0x68,0x56,0xb4,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 56 b4 e0 <unknown>

st1w    {za0h.s[w12, 0]}, p6, [x12, x2, lsl #2]
// CHECK-INST: st1w    {za0h.s[w12, 0]}, p6, [x12, x2, lsl #2]
// CHECK-ENCODING: [0x80,0x19,0xa2,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 19 a2 e0 <unknown>

st1w    {za0h.s[w14, 1]}, p2, [x1, x26, lsl #2]
// CHECK-INST: st1w    {za0h.s[w14, 1]}, p2, [x1, x26, lsl #2]
// CHECK-ENCODING: [0x21,0x48,0xba,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 48 ba e0 <unknown>

st1w    {za3h.s[w12, 1]}, p2, [x22, x30, lsl #2]
// CHECK-INST: st1w    {za3h.s[w12, 1]}, p2, [x22, x30, lsl #2]
// CHECK-ENCODING: [0xcd,0x0a,0xbe,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 0a be e0 <unknown>

st1w    {za0h.s[w15, 2]}, p5, [x9, x1, lsl #2]
// CHECK-INST: st1w    {za0h.s[w15, 2]}, p5, [x9, x1, lsl #2]
// CHECK-ENCODING: [0x22,0x75,0xa1,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 75 a1 e0 <unknown>

st1w    {za1h.s[w13, 3]}, p2, [x12, x11, lsl #2]
// CHECK-INST: st1w    {za1h.s[w13, 3]}, p2, [x12, x11, lsl #2]
// CHECK-ENCODING: [0x87,0x29,0xab,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 29 ab e0 <unknown>

st1w    za0h.s[w12, 0], p0, [x0, x0, lsl #2]
// CHECK-INST: st1w    {za0h.s[w12, 0]}, p0, [x0, x0, lsl #2]
// CHECK-ENCODING: [0x00,0x00,0xa0,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 00 a0 e0 <unknown>

st1w    za1h.s[w14, 1], p5, [x10, x21, lsl #2]
// CHECK-INST: st1w    {za1h.s[w14, 1]}, p5, [x10, x21, lsl #2]
// CHECK-ENCODING: [0x45,0x55,0xb5,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 55 b5 e0 <unknown>

st1w    za1h.s[w15, 3], p3, [x13, x8, lsl #2]
// CHECK-INST: st1w    {za1h.s[w15, 3]}, p3, [x13, x8, lsl #2]
// CHECK-ENCODING: [0xa7,0x6d,0xa8,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 6d a8 e0 <unknown>

st1w    za3h.s[w15, 3], p7, [sp]
// CHECK-INST: st1w    {za3h.s[w15, 3]}, p7, [sp]
// CHECK-ENCODING: [0xef,0x7f,0xbf,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef 7f bf e0 <unknown>

st1w    za1h.s[w12, 1], p3, [x17, x16, lsl #2]
// CHECK-INST: st1w    {za1h.s[w12, 1]}, p3, [x17, x16, lsl #2]
// CHECK-ENCODING: [0x25,0x0e,0xb0,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 0e b0 e0 <unknown>

st1w    za0h.s[w12, 1], p1, [x1, x30, lsl #2]
// CHECK-INST: st1w    {za0h.s[w12, 1]}, p1, [x1, x30, lsl #2]
// CHECK-ENCODING: [0x21,0x04,0xbe,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 04 be e0 <unknown>

st1w    za2h.s[w14, 0], p5, [x19, x20, lsl #2]
// CHECK-INST: st1w    {za2h.s[w14, 0]}, p5, [x19, x20, lsl #2]
// CHECK-ENCODING: [0x68,0x56,0xb4,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 56 b4 e0 <unknown>

st1w    za0h.s[w12, 0], p6, [x12, x2, lsl #2]
// CHECK-INST: st1w    {za0h.s[w12, 0]}, p6, [x12, x2, lsl #2]
// CHECK-ENCODING: [0x80,0x19,0xa2,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 19 a2 e0 <unknown>

st1w    za0h.s[w14, 1], p2, [x1, x26, lsl #2]
// CHECK-INST: st1w    {za0h.s[w14, 1]}, p2, [x1, x26, lsl #2]
// CHECK-ENCODING: [0x21,0x48,0xba,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 48 ba e0 <unknown>

st1w    za3h.s[w12, 1], p2, [x22, x30, lsl #2]
// CHECK-INST: st1w    {za3h.s[w12, 1]}, p2, [x22, x30, lsl #2]
// CHECK-ENCODING: [0xcd,0x0a,0xbe,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 0a be e0 <unknown>

st1w    za0h.s[w15, 2], p5, [x9, x1, lsl #2]
// CHECK-INST: st1w    {za0h.s[w15, 2]}, p5, [x9, x1, lsl #2]
// CHECK-ENCODING: [0x22,0x75,0xa1,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 75 a1 e0 <unknown>

st1w    za1h.s[w13, 3], p2, [x12, x11, lsl #2]
// CHECK-INST: st1w    {za1h.s[w13, 3]}, p2, [x12, x11, lsl #2]
// CHECK-ENCODING: [0x87,0x29,0xab,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 29 ab e0 <unknown>

// --------------------------------------------------------------------------//
// Vertical

st1w    {za0v.s[w12, 0]}, p0, [x0, x0, lsl #2]
// CHECK-INST: st1w    {za0v.s[w12, 0]}, p0, [x0, x0, lsl #2]
// CHECK-ENCODING: [0x00,0x80,0xa0,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 80 a0 e0 <unknown>

st1w    {za1v.s[w14, 1]}, p5, [x10, x21, lsl #2]
// CHECK-INST: st1w    {za1v.s[w14, 1]}, p5, [x10, x21, lsl #2]
// CHECK-ENCODING: [0x45,0xd5,0xb5,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 d5 b5 e0 <unknown>

st1w    {za1v.s[w15, 3]}, p3, [x13, x8, lsl #2]
// CHECK-INST: st1w    {za1v.s[w15, 3]}, p3, [x13, x8, lsl #2]
// CHECK-ENCODING: [0xa7,0xed,0xa8,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 ed a8 e0 <unknown>

st1w    {za3v.s[w15, 3]}, p7, [sp]
// CHECK-INST: st1w    {za3v.s[w15, 3]}, p7, [sp]
// CHECK-ENCODING: [0xef,0xff,0xbf,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef ff bf e0 <unknown>

st1w    {za1v.s[w12, 1]}, p3, [x17, x16, lsl #2]
// CHECK-INST: st1w    {za1v.s[w12, 1]}, p3, [x17, x16, lsl #2]
// CHECK-ENCODING: [0x25,0x8e,0xb0,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 8e b0 e0 <unknown>

st1w    {za0v.s[w12, 1]}, p1, [x1, x30, lsl #2]
// CHECK-INST: st1w    {za0v.s[w12, 1]}, p1, [x1, x30, lsl #2]
// CHECK-ENCODING: [0x21,0x84,0xbe,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 84 be e0 <unknown>

st1w    {za2v.s[w14, 0]}, p5, [x19, x20, lsl #2]
// CHECK-INST: st1w    {za2v.s[w14, 0]}, p5, [x19, x20, lsl #2]
// CHECK-ENCODING: [0x68,0xd6,0xb4,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 d6 b4 e0 <unknown>

st1w    {za0v.s[w12, 0]}, p6, [x12, x2, lsl #2]
// CHECK-INST: st1w    {za0v.s[w12, 0]}, p6, [x12, x2, lsl #2]
// CHECK-ENCODING: [0x80,0x99,0xa2,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 99 a2 e0 <unknown>

st1w    {za0v.s[w14, 1]}, p2, [x1, x26, lsl #2]
// CHECK-INST: st1w    {za0v.s[w14, 1]}, p2, [x1, x26, lsl #2]
// CHECK-ENCODING: [0x21,0xc8,0xba,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 c8 ba e0 <unknown>

st1w    {za3v.s[w12, 1]}, p2, [x22, x30, lsl #2]
// CHECK-INST: st1w    {za3v.s[w12, 1]}, p2, [x22, x30, lsl #2]
// CHECK-ENCODING: [0xcd,0x8a,0xbe,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 8a be e0 <unknown>

st1w    {za0v.s[w15, 2]}, p5, [x9, x1, lsl #2]
// CHECK-INST: st1w    {za0v.s[w15, 2]}, p5, [x9, x1, lsl #2]
// CHECK-ENCODING: [0x22,0xf5,0xa1,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 f5 a1 e0 <unknown>

st1w    {za1v.s[w13, 3]}, p2, [x12, x11, lsl #2]
// CHECK-INST: st1w    {za1v.s[w13, 3]}, p2, [x12, x11, lsl #2]
// CHECK-ENCODING: [0x87,0xa9,0xab,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 a9 ab e0 <unknown>

st1w    za0v.s[w12, 0], p0, [x0, x0, lsl #2]
// CHECK-INST: st1w    {za0v.s[w12, 0]}, p0, [x0, x0, lsl #2]
// CHECK-ENCODING: [0x00,0x80,0xa0,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 80 a0 e0 <unknown>

st1w    za1v.s[w14, 1], p5, [x10, x21, lsl #2]
// CHECK-INST: st1w    {za1v.s[w14, 1]}, p5, [x10, x21, lsl #2]
// CHECK-ENCODING: [0x45,0xd5,0xb5,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 d5 b5 e0 <unknown>

st1w    za1v.s[w15, 3], p3, [x13, x8, lsl #2]
// CHECK-INST: st1w    {za1v.s[w15, 3]}, p3, [x13, x8, lsl #2]
// CHECK-ENCODING: [0xa7,0xed,0xa8,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 ed a8 e0 <unknown>

st1w    za3v.s[w15, 3], p7, [sp]
// CHECK-INST: st1w    {za3v.s[w15, 3]}, p7, [sp]
// CHECK-ENCODING: [0xef,0xff,0xbf,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef ff bf e0 <unknown>

st1w    za1v.s[w12, 1], p3, [x17, x16, lsl #2]
// CHECK-INST: st1w    {za1v.s[w12, 1]}, p3, [x17, x16, lsl #2]
// CHECK-ENCODING: [0x25,0x8e,0xb0,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 8e b0 e0 <unknown>

st1w    za0v.s[w12, 1], p1, [x1, x30, lsl #2]
// CHECK-INST: st1w    {za0v.s[w12, 1]}, p1, [x1, x30, lsl #2]
// CHECK-ENCODING: [0x21,0x84,0xbe,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 84 be e0 <unknown>

st1w    za2v.s[w14, 0], p5, [x19, x20, lsl #2]
// CHECK-INST: st1w    {za2v.s[w14, 0]}, p5, [x19, x20, lsl #2]
// CHECK-ENCODING: [0x68,0xd6,0xb4,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 d6 b4 e0 <unknown>

st1w    za0v.s[w12, 0], p6, [x12, x2, lsl #2]
// CHECK-INST: st1w    {za0v.s[w12, 0]}, p6, [x12, x2, lsl #2]
// CHECK-ENCODING: [0x80,0x99,0xa2,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 99 a2 e0 <unknown>

st1w    za0v.s[w14, 1], p2, [x1, x26, lsl #2]
// CHECK-INST: st1w    {za0v.s[w14, 1]}, p2, [x1, x26, lsl #2]
// CHECK-ENCODING: [0x21,0xc8,0xba,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 c8 ba e0 <unknown>

st1w    za3v.s[w12, 1], p2, [x22, x30, lsl #2]
// CHECK-INST: st1w    {za3v.s[w12, 1]}, p2, [x22, x30, lsl #2]
// CHECK-ENCODING: [0xcd,0x8a,0xbe,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 8a be e0 <unknown>

st1w    za0v.s[w15, 2], p5, [x9, x1, lsl #2]
// CHECK-INST: st1w    {za0v.s[w15, 2]}, p5, [x9, x1, lsl #2]
// CHECK-ENCODING: [0x22,0xf5,0xa1,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 f5 a1 e0 <unknown>

st1w    za1v.s[w13, 3], p2, [x12, x11, lsl #2]
// CHECK-INST: st1w    {za1v.s[w13, 3]}, p2, [x12, x11, lsl #2]
// CHECK-ENCODING: [0x87,0xa9,0xab,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 a9 ab e0 <unknown>
