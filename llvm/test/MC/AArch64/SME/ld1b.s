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

ld1b    {za0h.b[w12, 0]}, p0/z, [x0, x0]
// CHECK-INST: ld1b    {za0h.b[w12, 0]}, p0/z, [x0, x0]
// CHECK-ENCODING: [0x00,0x00,0x00,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 00 00 e0 <unknown>

ld1b    {za0h.b[w14, 5]}, p5/z, [x10, x21]
// CHECK-INST: ld1b    {za0h.b[w14, 5]}, p5/z, [x10, x21]
// CHECK-ENCODING: [0x45,0x55,0x15,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 55 15 e0 <unknown>

ld1b    {za0h.b[w15, 7]}, p3/z, [x13, x8]
// CHECK-INST: ld1b    {za0h.b[w15, 7]}, p3/z, [x13, x8]
// CHECK-ENCODING: [0xa7,0x6d,0x08,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 6d 08 e0 <unknown>

ld1b    {za0h.b[w15, 15]}, p7/z, [sp]
// CHECK-INST: ld1b    {za0h.b[w15, 15]}, p7/z, [sp]
// CHECK-ENCODING: [0xef,0x7f,0x1f,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef 7f 1f e0 <unknown>

ld1b    {za0h.b[w12, 5]}, p3/z, [x17, x16]
// CHECK-INST: ld1b    {za0h.b[w12, 5]}, p3/z, [x17, x16]
// CHECK-ENCODING: [0x25,0x0e,0x10,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 0e 10 e0 <unknown>

ld1b    {za0h.b[w12, 1]}, p1/z, [x1, x30]
// CHECK-INST: ld1b    {za0h.b[w12, 1]}, p1/z, [x1, x30]
// CHECK-ENCODING: [0x21,0x04,0x1e,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 04 1e e0 <unknown>

ld1b    {za0h.b[w14, 8]}, p5/z, [x19, x20]
// CHECK-INST: ld1b    {za0h.b[w14, 8]}, p5/z, [x19, x20]
// CHECK-ENCODING: [0x68,0x56,0x14,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 56 14 e0 <unknown>

ld1b    {za0h.b[w12, 0]}, p6/z, [x12, x2]
// CHECK-INST: ld1b    {za0h.b[w12, 0]}, p6/z, [x12, x2]
// CHECK-ENCODING: [0x80,0x19,0x02,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 19 02 e0 <unknown>

ld1b    {za0h.b[w14, 1]}, p2/z, [x1, x26]
// CHECK-INST: ld1b    {za0h.b[w14, 1]}, p2/z, [x1, x26]
// CHECK-ENCODING: [0x21,0x48,0x1a,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 48 1a e0 <unknown>

ld1b    {za0h.b[w12, 13]}, p2/z, [x22, x30]
// CHECK-INST: ld1b    {za0h.b[w12, 13]}, p2/z, [x22, x30]
// CHECK-ENCODING: [0xcd,0x0a,0x1e,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 0a 1e e0 <unknown>

ld1b    {za0h.b[w15, 2]}, p5/z, [x9, x1]
// CHECK-INST: ld1b    {za0h.b[w15, 2]}, p5/z, [x9, x1]
// CHECK-ENCODING: [0x22,0x75,0x01,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 75 01 e0 <unknown>

ld1b    {za0h.b[w13, 7]}, p2/z, [x12, x11]
// CHECK-INST: ld1b    {za0h.b[w13, 7]}, p2/z, [x12, x11]
// CHECK-ENCODING: [0x87,0x29,0x0b,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 29 0b e0 <unknown>

ld1b    za0h.b[w12, 0], p0/z, [x0, x0]
// CHECK-INST: ld1b    {za0h.b[w12, 0]}, p0/z, [x0, x0]
// CHECK-ENCODING: [0x00,0x00,0x00,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 00 00 e0 <unknown>

ld1b    za0h.b[w14, 5], p5/z, [x10, x21]
// CHECK-INST: ld1b    {za0h.b[w14, 5]}, p5/z, [x10, x21]
// CHECK-ENCODING: [0x45,0x55,0x15,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 55 15 e0 <unknown>

ld1b    za0h.b[w15, 7], p3/z, [x13, x8]
// CHECK-INST: ld1b    {za0h.b[w15, 7]}, p3/z, [x13, x8]
// CHECK-ENCODING: [0xa7,0x6d,0x08,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 6d 08 e0 <unknown>

ld1b    za0h.b[w15, 15], p7/z, [sp]
// CHECK-INST: ld1b    {za0h.b[w15, 15]}, p7/z, [sp]
// CHECK-ENCODING: [0xef,0x7f,0x1f,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef 7f 1f e0 <unknown>

ld1b    za0h.b[w12, 5], p3/z, [x17, x16]
// CHECK-INST: ld1b    {za0h.b[w12, 5]}, p3/z, [x17, x16]
// CHECK-ENCODING: [0x25,0x0e,0x10,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 0e 10 e0 <unknown>

ld1b    za0h.b[w12, 1], p1/z, [x1, x30]
// CHECK-INST: ld1b    {za0h.b[w12, 1]}, p1/z, [x1, x30]
// CHECK-ENCODING: [0x21,0x04,0x1e,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 04 1e e0 <unknown>

ld1b    za0h.b[w14, 8], p5/z, [x19, x20]
// CHECK-INST: ld1b    {za0h.b[w14, 8]}, p5/z, [x19, x20]
// CHECK-ENCODING: [0x68,0x56,0x14,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 56 14 e0 <unknown>

ld1b    za0h.b[w12, 0], p6/z, [x12, x2]
// CHECK-INST: ld1b    {za0h.b[w12, 0]}, p6/z, [x12, x2]
// CHECK-ENCODING: [0x80,0x19,0x02,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 19 02 e0 <unknown>

ld1b    za0h.b[w14, 1], p2/z, [x1, x26]
// CHECK-INST: ld1b    {za0h.b[w14, 1]}, p2/z, [x1, x26]
// CHECK-ENCODING: [0x21,0x48,0x1a,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 48 1a e0 <unknown>

ld1b    za0h.b[w12, 13], p2/z, [x22, x30]
// CHECK-INST: ld1b    {za0h.b[w12, 13]}, p2/z, [x22, x30]
// CHECK-ENCODING: [0xcd,0x0a,0x1e,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 0a 1e e0 <unknown>

ld1b    za0h.b[w15, 2], p5/z, [x9, x1]
// CHECK-INST: ld1b    {za0h.b[w15, 2]}, p5/z, [x9, x1]
// CHECK-ENCODING: [0x22,0x75,0x01,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 75 01 e0 <unknown>

ld1b    za0h.b[w13, 7], p2/z, [x12, x11]
// CHECK-INST: ld1b    {za0h.b[w13, 7]}, p2/z, [x12, x11]
// CHECK-ENCODING: [0x87,0x29,0x0b,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 29 0b e0 <unknown>

// --------------------------------------------------------------------------//
// Vertical

ld1b    {za0v.b[w12, 0]}, p0/z, [x0, x0]
// CHECK-INST: ld1b    {za0v.b[w12, 0]}, p0/z, [x0, x0]
// CHECK-ENCODING: [0x00,0x80,0x00,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 80 00 e0 <unknown>

ld1b    {za0v.b[w14, 5]}, p5/z, [x10, x21]
// CHECK-INST: ld1b    {za0v.b[w14, 5]}, p5/z, [x10, x21]
// CHECK-ENCODING: [0x45,0xd5,0x15,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 d5 15 e0 <unknown>

ld1b    {za0v.b[w15, 7]}, p3/z, [x13, x8]
// CHECK-INST: ld1b    {za0v.b[w15, 7]}, p3/z, [x13, x8]
// CHECK-ENCODING: [0xa7,0xed,0x08,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 ed 08 e0 <unknown>

ld1b    {za0v.b[w15, 15]}, p7/z, [sp]
// CHECK-INST: ld1b    {za0v.b[w15, 15]}, p7/z, [sp]
// CHECK-ENCODING: [0xef,0xff,0x1f,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef ff 1f e0 <unknown>

ld1b    {za0v.b[w12, 5]}, p3/z, [x17, x16]
// CHECK-INST: ld1b    {za0v.b[w12, 5]}, p3/z, [x17, x16]
// CHECK-ENCODING: [0x25,0x8e,0x10,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 8e 10 e0 <unknown>

ld1b    {za0v.b[w12, 1]}, p1/z, [x1, x30]
// CHECK-INST: ld1b    {za0v.b[w12, 1]}, p1/z, [x1, x30]
// CHECK-ENCODING: [0x21,0x84,0x1e,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 84 1e e0 <unknown>

ld1b    {za0v.b[w14, 8]}, p5/z, [x19, x20]
// CHECK-INST: ld1b    {za0v.b[w14, 8]}, p5/z, [x19, x20]
// CHECK-ENCODING: [0x68,0xd6,0x14,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 d6 14 e0 <unknown>

ld1b    {za0v.b[w12, 0]}, p6/z, [x12, x2]
// CHECK-INST: ld1b    {za0v.b[w12, 0]}, p6/z, [x12, x2]
// CHECK-ENCODING: [0x80,0x99,0x02,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 99 02 e0 <unknown>

ld1b    {za0v.b[w14, 1]}, p2/z, [x1, x26]
// CHECK-INST: ld1b    {za0v.b[w14, 1]}, p2/z, [x1, x26]
// CHECK-ENCODING: [0x21,0xc8,0x1a,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 c8 1a e0 <unknown>

ld1b    {za0v.b[w12, 13]}, p2/z, [x22, x30]
// CHECK-INST: ld1b    {za0v.b[w12, 13]}, p2/z, [x22, x30]
// CHECK-ENCODING: [0xcd,0x8a,0x1e,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 8a 1e e0 <unknown>

ld1b    {za0v.b[w15, 2]}, p5/z, [x9, x1]
// CHECK-INST: ld1b    {za0v.b[w15, 2]}, p5/z, [x9, x1]
// CHECK-ENCODING: [0x22,0xf5,0x01,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 f5 01 e0 <unknown>

ld1b    {za0v.b[w13, 7]}, p2/z, [x12, x11]
// CHECK-INST: ld1b    {za0v.b[w13, 7]}, p2/z, [x12, x11]
// CHECK-ENCODING: [0x87,0xa9,0x0b,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 a9 0b e0 <unknown>

ld1b    za0v.b[w12, 0], p0/z, [x0, x0]
// CHECK-INST: ld1b    {za0v.b[w12, 0]}, p0/z, [x0, x0]
// CHECK-ENCODING: [0x00,0x80,0x00,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 80 00 e0 <unknown>

ld1b    za0v.b[w14, 5], p5/z, [x10, x21]
// CHECK-INST: ld1b    {za0v.b[w14, 5]}, p5/z, [x10, x21]
// CHECK-ENCODING: [0x45,0xd5,0x15,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 d5 15 e0 <unknown>

ld1b    za0v.b[w15, 7], p3/z, [x13, x8]
// CHECK-INST: ld1b    {za0v.b[w15, 7]}, p3/z, [x13, x8]
// CHECK-ENCODING: [0xa7,0xed,0x08,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 ed 08 e0 <unknown>

ld1b    za0v.b[w15, 15], p7/z, [sp]
// CHECK-INST: ld1b    {za0v.b[w15, 15]}, p7/z, [sp]
// CHECK-ENCODING: [0xef,0xff,0x1f,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef ff 1f e0 <unknown>

ld1b    za0v.b[w12, 5], p3/z, [x17, x16]
// CHECK-INST: ld1b    {za0v.b[w12, 5]}, p3/z, [x17, x16]
// CHECK-ENCODING: [0x25,0x8e,0x10,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 8e 10 e0 <unknown>

ld1b    za0v.b[w12, 1], p1/z, [x1, x30]
// CHECK-INST: ld1b    {za0v.b[w12, 1]}, p1/z, [x1, x30]
// CHECK-ENCODING: [0x21,0x84,0x1e,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 84 1e e0 <unknown>

ld1b    za0v.b[w14, 8], p5/z, [x19, x20]
// CHECK-INST: ld1b    {za0v.b[w14, 8]}, p5/z, [x19, x20]
// CHECK-ENCODING: [0x68,0xd6,0x14,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 d6 14 e0 <unknown>

ld1b    za0v.b[w12, 0], p6/z, [x12, x2]
// CHECK-INST: ld1b    {za0v.b[w12, 0]}, p6/z, [x12, x2]
// CHECK-ENCODING: [0x80,0x99,0x02,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 99 02 e0 <unknown>

ld1b    za0v.b[w14, 1], p2/z, [x1, x26]
// CHECK-INST: ld1b    {za0v.b[w14, 1]}, p2/z, [x1, x26]
// CHECK-ENCODING: [0x21,0xc8,0x1a,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 c8 1a e0 <unknown>

ld1b    za0v.b[w12, 13], p2/z, [x22, x30]
// CHECK-INST: ld1b    {za0v.b[w12, 13]}, p2/z, [x22, x30]
// CHECK-ENCODING: [0xcd,0x8a,0x1e,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 8a 1e e0 <unknown>

ld1b    za0v.b[w15, 2], p5/z, [x9, x1]
// CHECK-INST: ld1b    {za0v.b[w15, 2]}, p5/z, [x9, x1]
// CHECK-ENCODING: [0x22,0xf5,0x01,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 f5 01 e0 <unknown>

ld1b    za0v.b[w13, 7], p2/z, [x12, x11]
// CHECK-INST: ld1b    {za0v.b[w13, 7]}, p2/z, [x12, x11]
// CHECK-ENCODING: [0x87,0xa9,0x0b,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 a9 0b e0 <unknown>
