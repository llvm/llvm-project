// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme < %s \
// RUN:        | llvm-objdump -d --mattr=+sme - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme < %s \
// RUN:   | llvm-objdump -d --mattr=-sme - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

// --------------------------------------------------------------------------//
// Horizontal

ld1w    {za0h.s[w12, 0]}, p0/z, [x0, x0, lsl #2]
// CHECK-INST: ld1w    {za0h.s[w12, 0]}, p0/z, [x0, x0, lsl #2]
// CHECK-ENCODING: [0x00,0x00,0x80,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 00 80 e0 <unknown>

ld1w    {za1h.s[w14, 1]}, p5/z, [x10, x21, lsl #2]
// CHECK-INST: ld1w    {za1h.s[w14, 1]}, p5/z, [x10, x21, lsl #2]
// CHECK-ENCODING: [0x45,0x55,0x95,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 55 95 e0 <unknown>

ld1w    {za1h.s[w15, 3]}, p3/z, [x13, x8, lsl #2]
// CHECK-INST: ld1w    {za1h.s[w15, 3]}, p3/z, [x13, x8, lsl #2]
// CHECK-ENCODING: [0xa7,0x6d,0x88,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 6d 88 e0 <unknown>

ld1w    {za3h.s[w15, 3]}, p7/z, [sp]
// CHECK-INST: ld1w    {za3h.s[w15, 3]}, p7/z, [sp]
// CHECK-ENCODING: [0xef,0x7f,0x9f,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef 7f 9f e0 <unknown>

ld1w    {za1h.s[w12, 1]}, p3/z, [x17, x16, lsl #2]
// CHECK-INST: ld1w    {za1h.s[w12, 1]}, p3/z, [x17, x16, lsl #2]
// CHECK-ENCODING: [0x25,0x0e,0x90,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 0e 90 e0 <unknown>

ld1w    {za0h.s[w12, 1]}, p1/z, [x1, x30, lsl #2]
// CHECK-INST: ld1w    {za0h.s[w12, 1]}, p1/z, [x1, x30, lsl #2]
// CHECK-ENCODING: [0x21,0x04,0x9e,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 04 9e e0 <unknown>

ld1w    {za2h.s[w14, 0]}, p5/z, [x19, x20, lsl #2]
// CHECK-INST: ld1w    {za2h.s[w14, 0]}, p5/z, [x19, x20, lsl #2]
// CHECK-ENCODING: [0x68,0x56,0x94,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 56 94 e0 <unknown>

ld1w    {za0h.s[w12, 0]}, p6/z, [x12, x2, lsl #2]
// CHECK-INST: ld1w    {za0h.s[w12, 0]}, p6/z, [x12, x2, lsl #2]
// CHECK-ENCODING: [0x80,0x19,0x82,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 19 82 e0 <unknown>

ld1w    {za0h.s[w14, 1]}, p2/z, [x1, x26, lsl #2]
// CHECK-INST: ld1w    {za0h.s[w14, 1]}, p2/z, [x1, x26, lsl #2]
// CHECK-ENCODING: [0x21,0x48,0x9a,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 48 9a e0 <unknown>

ld1w    {za3h.s[w12, 1]}, p2/z, [x22, x30, lsl #2]
// CHECK-INST: ld1w    {za3h.s[w12, 1]}, p2/z, [x22, x30, lsl #2]
// CHECK-ENCODING: [0xcd,0x0a,0x9e,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 0a 9e e0 <unknown>

ld1w    {za0h.s[w15, 2]}, p5/z, [x9, x1, lsl #2]
// CHECK-INST: ld1w    {za0h.s[w15, 2]}, p5/z, [x9, x1, lsl #2]
// CHECK-ENCODING: [0x22,0x75,0x81,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 75 81 e0 <unknown>

ld1w    {za1h.s[w13, 3]}, p2/z, [x12, x11, lsl #2]
// CHECK-INST: ld1w    {za1h.s[w13, 3]}, p2/z, [x12, x11, lsl #2]
// CHECK-ENCODING: [0x87,0x29,0x8b,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 29 8b e0 <unknown>

ld1w    za0h.s[w12, 0], p0/z, [x0, x0, lsl #2]
// CHECK-INST: ld1w    {za0h.s[w12, 0]}, p0/z, [x0, x0, lsl #2]
// CHECK-ENCODING: [0x00,0x00,0x80,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 00 80 e0 <unknown>

ld1w    za1h.s[w14, 1], p5/z, [x10, x21, lsl #2]
// CHECK-INST: ld1w    {za1h.s[w14, 1]}, p5/z, [x10, x21, lsl #2]
// CHECK-ENCODING: [0x45,0x55,0x95,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 55 95 e0 <unknown>

ld1w    za1h.s[w15, 3], p3/z, [x13, x8, lsl #2]
// CHECK-INST: ld1w    {za1h.s[w15, 3]}, p3/z, [x13, x8, lsl #2]
// CHECK-ENCODING: [0xa7,0x6d,0x88,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 6d 88 e0 <unknown>

ld1w    za3h.s[w15, 3], p7/z, [sp]
// CHECK-INST: ld1w    {za3h.s[w15, 3]}, p7/z, [sp]
// CHECK-ENCODING: [0xef,0x7f,0x9f,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef 7f 9f e0 <unknown>

ld1w    za1h.s[w12, 1], p3/z, [x17, x16, lsl #2]
// CHECK-INST: ld1w    {za1h.s[w12, 1]}, p3/z, [x17, x16, lsl #2]
// CHECK-ENCODING: [0x25,0x0e,0x90,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 0e 90 e0 <unknown>

ld1w    za0h.s[w12, 1], p1/z, [x1, x30, lsl #2]
// CHECK-INST: ld1w    {za0h.s[w12, 1]}, p1/z, [x1, x30, lsl #2]
// CHECK-ENCODING: [0x21,0x04,0x9e,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 04 9e e0 <unknown>

ld1w    za2h.s[w14, 0], p5/z, [x19, x20, lsl #2]
// CHECK-INST: ld1w    {za2h.s[w14, 0]}, p5/z, [x19, x20, lsl #2]
// CHECK-ENCODING: [0x68,0x56,0x94,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 56 94 e0 <unknown>

ld1w    za0h.s[w12, 0], p6/z, [x12, x2, lsl #2]
// CHECK-INST: ld1w    {za0h.s[w12, 0]}, p6/z, [x12, x2, lsl #2]
// CHECK-ENCODING: [0x80,0x19,0x82,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 19 82 e0 <unknown>

ld1w    za0h.s[w14, 1], p2/z, [x1, x26, lsl #2]
// CHECK-INST: ld1w    {za0h.s[w14, 1]}, p2/z, [x1, x26, lsl #2]
// CHECK-ENCODING: [0x21,0x48,0x9a,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 48 9a e0 <unknown>

ld1w    za3h.s[w12, 1], p2/z, [x22, x30, lsl #2]
// CHECK-INST: ld1w    {za3h.s[w12, 1]}, p2/z, [x22, x30, lsl #2]
// CHECK-ENCODING: [0xcd,0x0a,0x9e,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 0a 9e e0 <unknown>

ld1w    za0h.s[w15, 2], p5/z, [x9, x1, lsl #2]
// CHECK-INST: ld1w    {za0h.s[w15, 2]}, p5/z, [x9, x1, lsl #2]
// CHECK-ENCODING: [0x22,0x75,0x81,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 75 81 e0 <unknown>

ld1w    za1h.s[w13, 3], p2/z, [x12, x11, lsl #2]
// CHECK-INST: ld1w    {za1h.s[w13, 3]}, p2/z, [x12, x11, lsl #2]
// CHECK-ENCODING: [0x87,0x29,0x8b,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 29 8b e0 <unknown>

// --------------------------------------------------------------------------//
// Vertical

ld1w    {za0v.s[w12, 0]}, p0/z, [x0, x0, lsl #2]
// CHECK-INST: ld1w    {za0v.s[w12, 0]}, p0/z, [x0, x0, lsl #2]
// CHECK-ENCODING: [0x00,0x80,0x80,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 80 80 e0 <unknown>

ld1w    {za1v.s[w14, 1]}, p5/z, [x10, x21, lsl #2]
// CHECK-INST: ld1w    {za1v.s[w14, 1]}, p5/z, [x10, x21, lsl #2]
// CHECK-ENCODING: [0x45,0xd5,0x95,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 d5 95 e0 <unknown>

ld1w    {za1v.s[w15, 3]}, p3/z, [x13, x8, lsl #2]
// CHECK-INST: ld1w    {za1v.s[w15, 3]}, p3/z, [x13, x8, lsl #2]
// CHECK-ENCODING: [0xa7,0xed,0x88,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 ed 88 e0 <unknown>

ld1w    {za3v.s[w15, 3]}, p7/z, [sp]
// CHECK-INST: ld1w    {za3v.s[w15, 3]}, p7/z, [sp]
// CHECK-ENCODING: [0xef,0xff,0x9f,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef ff 9f e0 <unknown>

ld1w    {za1v.s[w12, 1]}, p3/z, [x17, x16, lsl #2]
// CHECK-INST: ld1w    {za1v.s[w12, 1]}, p3/z, [x17, x16, lsl #2]
// CHECK-ENCODING: [0x25,0x8e,0x90,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 8e 90 e0 <unknown>

ld1w    {za0v.s[w12, 1]}, p1/z, [x1, x30, lsl #2]
// CHECK-INST: ld1w    {za0v.s[w12, 1]}, p1/z, [x1, x30, lsl #2]
// CHECK-ENCODING: [0x21,0x84,0x9e,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 84 9e e0 <unknown>

ld1w    {za2v.s[w14, 0]}, p5/z, [x19, x20, lsl #2]
// CHECK-INST: ld1w    {za2v.s[w14, 0]}, p5/z, [x19, x20, lsl #2]
// CHECK-ENCODING: [0x68,0xd6,0x94,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 d6 94 e0 <unknown>

ld1w    {za0v.s[w12, 0]}, p6/z, [x12, x2, lsl #2]
// CHECK-INST: ld1w    {za0v.s[w12, 0]}, p6/z, [x12, x2, lsl #2]
// CHECK-ENCODING: [0x80,0x99,0x82,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 99 82 e0 <unknown>

ld1w    {za0v.s[w14, 1]}, p2/z, [x1, x26, lsl #2]
// CHECK-INST: ld1w    {za0v.s[w14, 1]}, p2/z, [x1, x26, lsl #2]
// CHECK-ENCODING: [0x21,0xc8,0x9a,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 c8 9a e0 <unknown>

ld1w    {za3v.s[w12, 1]}, p2/z, [x22, x30, lsl #2]
// CHECK-INST: ld1w    {za3v.s[w12, 1]}, p2/z, [x22, x30, lsl #2]
// CHECK-ENCODING: [0xcd,0x8a,0x9e,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 8a 9e e0 <unknown>

ld1w    {za0v.s[w15, 2]}, p5/z, [x9, x1, lsl #2]
// CHECK-INST: ld1w    {za0v.s[w15, 2]}, p5/z, [x9, x1, lsl #2]
// CHECK-ENCODING: [0x22,0xf5,0x81,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 f5 81 e0 <unknown>

ld1w    {za1v.s[w13, 3]}, p2/z, [x12, x11, lsl #2]
// CHECK-INST: ld1w    {za1v.s[w13, 3]}, p2/z, [x12, x11, lsl #2]
// CHECK-ENCODING: [0x87,0xa9,0x8b,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 a9 8b e0 <unknown>

ld1w    za0v.s[w12, 0], p0/z, [x0, x0, lsl #2]
// CHECK-INST: ld1w    {za0v.s[w12, 0]}, p0/z, [x0, x0, lsl #2]
// CHECK-ENCODING: [0x00,0x80,0x80,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 80 80 e0 <unknown>

ld1w    za1v.s[w14, 1], p5/z, [x10, x21, lsl #2]
// CHECK-INST: ld1w    {za1v.s[w14, 1]}, p5/z, [x10, x21, lsl #2]
// CHECK-ENCODING: [0x45,0xd5,0x95,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 d5 95 e0 <unknown>

ld1w    za1v.s[w15, 3], p3/z, [x13, x8, lsl #2]
// CHECK-INST: ld1w    {za1v.s[w15, 3]}, p3/z, [x13, x8, lsl #2]
// CHECK-ENCODING: [0xa7,0xed,0x88,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 ed 88 e0 <unknown>

ld1w    za3v.s[w15, 3], p7/z, [sp]
// CHECK-INST: ld1w    {za3v.s[w15, 3]}, p7/z, [sp]
// CHECK-ENCODING: [0xef,0xff,0x9f,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef ff 9f e0 <unknown>

ld1w    za1v.s[w12, 1], p3/z, [x17, x16, lsl #2]
// CHECK-INST: ld1w    {za1v.s[w12, 1]}, p3/z, [x17, x16, lsl #2]
// CHECK-ENCODING: [0x25,0x8e,0x90,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 8e 90 e0 <unknown>

ld1w    za0v.s[w12, 1], p1/z, [x1, x30, lsl #2]
// CHECK-INST: ld1w    {za0v.s[w12, 1]}, p1/z, [x1, x30, lsl #2]
// CHECK-ENCODING: [0x21,0x84,0x9e,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 84 9e e0 <unknown>

ld1w    za2v.s[w14, 0], p5/z, [x19, x20, lsl #2]
// CHECK-INST: ld1w    {za2v.s[w14, 0]}, p5/z, [x19, x20, lsl #2]
// CHECK-ENCODING: [0x68,0xd6,0x94,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 d6 94 e0 <unknown>

ld1w    za0v.s[w12, 0], p6/z, [x12, x2, lsl #2]
// CHECK-INST: ld1w    {za0v.s[w12, 0]}, p6/z, [x12, x2, lsl #2]
// CHECK-ENCODING: [0x80,0x99,0x82,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 99 82 e0 <unknown>

ld1w    za0v.s[w14, 1], p2/z, [x1, x26, lsl #2]
// CHECK-INST: ld1w    {za0v.s[w14, 1]}, p2/z, [x1, x26, lsl #2]
// CHECK-ENCODING: [0x21,0xc8,0x9a,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 c8 9a e0 <unknown>

ld1w    za3v.s[w12, 1], p2/z, [x22, x30, lsl #2]
// CHECK-INST: ld1w    {za3v.s[w12, 1]}, p2/z, [x22, x30, lsl #2]
// CHECK-ENCODING: [0xcd,0x8a,0x9e,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 8a 9e e0 <unknown>

ld1w    za0v.s[w15, 2], p5/z, [x9, x1, lsl #2]
// CHECK-INST: ld1w    {za0v.s[w15, 2]}, p5/z, [x9, x1, lsl #2]
// CHECK-ENCODING: [0x22,0xf5,0x81,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 f5 81 e0 <unknown>

ld1w    za1v.s[w13, 3], p2/z, [x12, x11, lsl #2]
// CHECK-INST: ld1w    {za1v.s[w13, 3]}, p2/z, [x12, x11, lsl #2]
// CHECK-ENCODING: [0x87,0xa9,0x8b,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 a9 8b e0 <unknown>
