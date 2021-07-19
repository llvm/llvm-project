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

ld1h    {za0h.h[w12, #0]}, p0/z, [x0, x0, lsl #1]
// CHECK-INST: ld1h    {za0h.h[w12, #0]}, p0/z, [x0, x0, lsl #1]
// CHECK-ENCODING: [0x00,0x00,0x40,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 00 40 e0 <unknown>

ld1h    {za0h.h[w14, #5]}, p5/z, [x10, x21, lsl #1]
// CHECK-INST: ld1h    {za0h.h[w14, #5]}, p5/z, [x10, x21, lsl #1]
// CHECK-ENCODING: [0x45,0x55,0x55,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 55 55 e0 <unknown>

ld1h    {za0h.h[w15, #7]}, p3/z, [x13, x8, lsl #1]
// CHECK-INST: ld1h    {za0h.h[w15, #7]}, p3/z, [x13, x8, lsl #1]
// CHECK-ENCODING: [0xa7,0x6d,0x48,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 6d 48 e0 <unknown>

ld1h    {za1h.h[w15, #7]}, p7/z, [sp]
// CHECK-INST: ld1h    {za1h.h[w15, #7]}, p7/z, [sp]
// CHECK-ENCODING: [0xef,0x7f,0x5f,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef 7f 5f e0 <unknown>

ld1h    {za0h.h[w12, #5]}, p3/z, [x17, x16, lsl #1]
// CHECK-INST: ld1h    {za0h.h[w12, #5]}, p3/z, [x17, x16, lsl #1]
// CHECK-ENCODING: [0x25,0x0e,0x50,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 0e 50 e0 <unknown>

ld1h    {za0h.h[w12, #1]}, p1/z, [x1, x30, lsl #1]
// CHECK-INST: ld1h    {za0h.h[w12, #1]}, p1/z, [x1, x30, lsl #1]
// CHECK-ENCODING: [0x21,0x04,0x5e,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 04 5e e0 <unknown>

ld1h    {za1h.h[w14, #0]}, p5/z, [x19, x20, lsl #1]
// CHECK-INST: ld1h    {za1h.h[w14, #0]}, p5/z, [x19, x20, lsl #1]
// CHECK-ENCODING: [0x68,0x56,0x54,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 56 54 e0 <unknown>

ld1h    {za0h.h[w12, #0]}, p6/z, [x12, x2, lsl #1]
// CHECK-INST: ld1h    {za0h.h[w12, #0]}, p6/z, [x12, x2, lsl #1]
// CHECK-ENCODING: [0x80,0x19,0x42,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 19 42 e0 <unknown>

ld1h    {za0h.h[w14, #1]}, p2/z, [x1, x26, lsl #1]
// CHECK-INST: ld1h    {za0h.h[w14, #1]}, p2/z, [x1, x26, lsl #1]
// CHECK-ENCODING: [0x21,0x48,0x5a,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 48 5a e0 <unknown>

ld1h    {za1h.h[w12, #5]}, p2/z, [x22, x30, lsl #1]
// CHECK-INST: ld1h    {za1h.h[w12, #5]}, p2/z, [x22, x30, lsl #1]
// CHECK-ENCODING: [0xcd,0x0a,0x5e,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 0a 5e e0 <unknown>

ld1h    {za0h.h[w15, #2]}, p5/z, [x9, x1, lsl #1]
// CHECK-INST: ld1h    {za0h.h[w15, #2]}, p5/z, [x9, x1, lsl #1]
// CHECK-ENCODING: [0x22,0x75,0x41,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 75 41 e0 <unknown>

ld1h    {za0h.h[w13, #7]}, p2/z, [x12, x11, lsl #1]
// CHECK-INST: ld1h    {za0h.h[w13, #7]}, p2/z, [x12, x11, lsl #1]
// CHECK-ENCODING: [0x87,0x29,0x4b,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 29 4b e0 <unknown>

ld1h    za0h.h[w12, #0], p0/z, [x0, x0, lsl #1]
// CHECK-INST: ld1h    {za0h.h[w12, #0]}, p0/z, [x0, x0, lsl #1]
// CHECK-ENCODING: [0x00,0x00,0x40,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 00 40 e0 <unknown>

ld1h    za0h.h[w14, #5], p5/z, [x10, x21, lsl #1]
// CHECK-INST: ld1h    {za0h.h[w14, #5]}, p5/z, [x10, x21, lsl #1]
// CHECK-ENCODING: [0x45,0x55,0x55,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 55 55 e0 <unknown>

ld1h    za0h.h[w15, #7], p3/z, [x13, x8, lsl #1]
// CHECK-INST: ld1h    {za0h.h[w15, #7]}, p3/z, [x13, x8, lsl #1]
// CHECK-ENCODING: [0xa7,0x6d,0x48,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 6d 48 e0 <unknown>

ld1h    za1h.h[w15, #7], p7/z, [sp]
// CHECK-INST: ld1h    {za1h.h[w15, #7]}, p7/z, [sp]
// CHECK-ENCODING: [0xef,0x7f,0x5f,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef 7f 5f e0 <unknown>

ld1h    za0h.h[w12, #5], p3/z, [x17, x16, lsl #1]
// CHECK-INST: ld1h    {za0h.h[w12, #5]}, p3/z, [x17, x16, lsl #1]
// CHECK-ENCODING: [0x25,0x0e,0x50,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 0e 50 e0 <unknown>

ld1h    za0h.h[w12, #1], p1/z, [x1, x30, lsl #1]
// CHECK-INST: ld1h    {za0h.h[w12, #1]}, p1/z, [x1, x30, lsl #1]
// CHECK-ENCODING: [0x21,0x04,0x5e,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 04 5e e0 <unknown>

ld1h    za1h.h[w14, #0], p5/z, [x19, x20, lsl #1]
// CHECK-INST: ld1h    {za1h.h[w14, #0]}, p5/z, [x19, x20, lsl #1]
// CHECK-ENCODING: [0x68,0x56,0x54,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 56 54 e0 <unknown>

ld1h    za0h.h[w12, #0], p6/z, [x12, x2, lsl #1]
// CHECK-INST: ld1h    {za0h.h[w12, #0]}, p6/z, [x12, x2, lsl #1]
// CHECK-ENCODING: [0x80,0x19,0x42,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 19 42 e0 <unknown>

ld1h    za0h.h[w14, #1], p2/z, [x1, x26, lsl #1]
// CHECK-INST: ld1h    {za0h.h[w14, #1]}, p2/z, [x1, x26, lsl #1]
// CHECK-ENCODING: [0x21,0x48,0x5a,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 48 5a e0 <unknown>

ld1h    za1h.h[w12, #5], p2/z, [x22, x30, lsl #1]
// CHECK-INST: ld1h    {za1h.h[w12, #5]}, p2/z, [x22, x30, lsl #1]
// CHECK-ENCODING: [0xcd,0x0a,0x5e,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 0a 5e e0 <unknown>

ld1h    za0h.h[w15, #2], p5/z, [x9, x1, lsl #1]
// CHECK-INST: ld1h    {za0h.h[w15, #2]}, p5/z, [x9, x1, lsl #1]
// CHECK-ENCODING: [0x22,0x75,0x41,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 75 41 e0 <unknown>

ld1h    za0h.h[w13, #7], p2/z, [x12, x11, lsl #1]
// CHECK-INST: ld1h    {za0h.h[w13, #7]}, p2/z, [x12, x11, lsl #1]
// CHECK-ENCODING: [0x87,0x29,0x4b,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 29 4b e0 <unknown>

// --------------------------------------------------------------------------//
// Vertical

ld1h    {za0v.h[w12, #0]}, p0/z, [x0, x0, lsl #1]
// CHECK-INST: ld1h    {za0v.h[w12, #0]}, p0/z, [x0, x0, lsl #1]
// CHECK-ENCODING: [0x00,0x80,0x40,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 80 40 e0 <unknown>

ld1h    {za0v.h[w14, #5]}, p5/z, [x10, x21, lsl #1]
// CHECK-INST: ld1h    {za0v.h[w14, #5]}, p5/z, [x10, x21, lsl #1]
// CHECK-ENCODING: [0x45,0xd5,0x55,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 d5 55 e0 <unknown>

ld1h    {za0v.h[w15, #7]}, p3/z, [x13, x8, lsl #1]
// CHECK-INST: ld1h    {za0v.h[w15, #7]}, p3/z, [x13, x8, lsl #1]
// CHECK-ENCODING: [0xa7,0xed,0x48,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 ed 48 e0 <unknown>

ld1h    {za1v.h[w15, #7]}, p7/z, [sp]
// CHECK-INST: ld1h    {za1v.h[w15, #7]}, p7/z, [sp]
// CHECK-ENCODING: [0xef,0xff,0x5f,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef ff 5f e0 <unknown>

ld1h    {za0v.h[w12, #5]}, p3/z, [x17, x16, lsl #1]
// CHECK-INST: ld1h    {za0v.h[w12, #5]}, p3/z, [x17, x16, lsl #1]
// CHECK-ENCODING: [0x25,0x8e,0x50,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 8e 50 e0 <unknown>

ld1h    {za0v.h[w12, #1]}, p1/z, [x1, x30, lsl #1]
// CHECK-INST: ld1h    {za0v.h[w12, #1]}, p1/z, [x1, x30, lsl #1]
// CHECK-ENCODING: [0x21,0x84,0x5e,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 84 5e e0 <unknown>

ld1h    {za1v.h[w14, #0]}, p5/z, [x19, x20, lsl #1]
// CHECK-INST: ld1h    {za1v.h[w14, #0]}, p5/z, [x19, x20, lsl #1]
// CHECK-ENCODING: [0x68,0xd6,0x54,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 d6 54 e0 <unknown>

ld1h    {za0v.h[w12, #0]}, p6/z, [x12, x2, lsl #1]
// CHECK-INST: ld1h    {za0v.h[w12, #0]}, p6/z, [x12, x2, lsl #1]
// CHECK-ENCODING: [0x80,0x99,0x42,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 99 42 e0 <unknown>

ld1h    {za0v.h[w14, #1]}, p2/z, [x1, x26, lsl #1]
// CHECK-INST: ld1h    {za0v.h[w14, #1]}, p2/z, [x1, x26, lsl #1]
// CHECK-ENCODING: [0x21,0xc8,0x5a,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 c8 5a e0 <unknown>

ld1h    {za1v.h[w12, #5]}, p2/z, [x22, x30, lsl #1]
// CHECK-INST: ld1h    {za1v.h[w12, #5]}, p2/z, [x22, x30, lsl #1]
// CHECK-ENCODING: [0xcd,0x8a,0x5e,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 8a 5e e0 <unknown>

ld1h    {za0v.h[w15, #2]}, p5/z, [x9, x1, lsl #1]
// CHECK-INST: ld1h    {za0v.h[w15, #2]}, p5/z, [x9, x1, lsl #1]
// CHECK-ENCODING: [0x22,0xf5,0x41,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 f5 41 e0 <unknown>

ld1h    {za0v.h[w13, #7]}, p2/z, [x12, x11, lsl #1]
// CHECK-INST: ld1h    {za0v.h[w13, #7]}, p2/z, [x12, x11, lsl #1]
// CHECK-ENCODING: [0x87,0xa9,0x4b,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 a9 4b e0 <unknown>

ld1h    za0v.h[w12, #0], p0/z, [x0, x0, lsl #1]
// CHECK-INST: ld1h    {za0v.h[w12, #0]}, p0/z, [x0, x0, lsl #1]
// CHECK-ENCODING: [0x00,0x80,0x40,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 80 40 e0 <unknown>

ld1h    za0v.h[w14, #5], p5/z, [x10, x21, lsl #1]
// CHECK-INST: ld1h    {za0v.h[w14, #5]}, p5/z, [x10, x21, lsl #1]
// CHECK-ENCODING: [0x45,0xd5,0x55,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 d5 55 e0 <unknown>

ld1h    za0v.h[w15, #7], p3/z, [x13, x8, lsl #1]
// CHECK-INST: ld1h    {za0v.h[w15, #7]}, p3/z, [x13, x8, lsl #1]
// CHECK-ENCODING: [0xa7,0xed,0x48,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 ed 48 e0 <unknown>

ld1h    za1v.h[w15, #7], p7/z, [sp]
// CHECK-INST: ld1h    {za1v.h[w15, #7]}, p7/z, [sp]
// CHECK-ENCODING: [0xef,0xff,0x5f,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef ff 5f e0 <unknown>

ld1h    za0v.h[w12, #5], p3/z, [x17, x16, lsl #1]
// CHECK-INST: ld1h    {za0v.h[w12, #5]}, p3/z, [x17, x16, lsl #1]
// CHECK-ENCODING: [0x25,0x8e,0x50,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 8e 50 e0 <unknown>

ld1h    za0v.h[w12, #1], p1/z, [x1, x30, lsl #1]
// CHECK-INST: ld1h    {za0v.h[w12, #1]}, p1/z, [x1, x30, lsl #1]
// CHECK-ENCODING: [0x21,0x84,0x5e,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 84 5e e0 <unknown>

ld1h    za1v.h[w14, #0], p5/z, [x19, x20, lsl #1]
// CHECK-INST: ld1h    {za1v.h[w14, #0]}, p5/z, [x19, x20, lsl #1]
// CHECK-ENCODING: [0x68,0xd6,0x54,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 d6 54 e0 <unknown>

ld1h    za0v.h[w12, #0], p6/z, [x12, x2, lsl #1]
// CHECK-INST: ld1h    {za0v.h[w12, #0]}, p6/z, [x12, x2, lsl #1]
// CHECK-ENCODING: [0x80,0x99,0x42,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 99 42 e0 <unknown>

ld1h    za0v.h[w14, #1], p2/z, [x1, x26, lsl #1]
// CHECK-INST: ld1h    {za0v.h[w14, #1]}, p2/z, [x1, x26, lsl #1]
// CHECK-ENCODING: [0x21,0xc8,0x5a,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 c8 5a e0 <unknown>

ld1h    za1v.h[w12, #5], p2/z, [x22, x30, lsl #1]
// CHECK-INST: ld1h    {za1v.h[w12, #5]}, p2/z, [x22, x30, lsl #1]
// CHECK-ENCODING: [0xcd,0x8a,0x5e,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 8a 5e e0 <unknown>

ld1h    za0v.h[w15, #2], p5/z, [x9, x1, lsl #1]
// CHECK-INST: ld1h    {za0v.h[w15, #2]}, p5/z, [x9, x1, lsl #1]
// CHECK-ENCODING: [0x22,0xf5,0x41,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 f5 41 e0 <unknown>

ld1h    za0v.h[w13, #7], p2/z, [x12, x11, lsl #1]
// CHECK-INST: ld1h    {za0v.h[w13, #7]}, p2/z, [x12, x11, lsl #1]
// CHECK-ENCODING: [0x87,0xa9,0x4b,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 a9 4b e0 <unknown>
