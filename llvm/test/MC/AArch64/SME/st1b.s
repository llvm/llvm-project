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

st1b    {za0h.b[w12, 0]}, p0, [x0, x0]
// CHECK-INST: st1b    {za0h.b[w12, 0]}, p0, [x0, x0]
// CHECK-ENCODING: [0x00,0x00,0x20,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 00 20 e0 <unknown>

st1b    {za0h.b[w14, 5]}, p5, [x10, x21]
// CHECK-INST: st1b    {za0h.b[w14, 5]}, p5, [x10, x21]
// CHECK-ENCODING: [0x45,0x55,0x35,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 55 35 e0 <unknown>

st1b    {za0h.b[w15, 7]}, p3, [x13, x8]
// CHECK-INST: st1b    {za0h.b[w15, 7]}, p3, [x13, x8]
// CHECK-ENCODING: [0xa7,0x6d,0x28,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 6d 28 e0 <unknown>

st1b    {za0h.b[w15, 15]}, p7, [sp]
// CHECK-INST: st1b    {za0h.b[w15, 15]}, p7, [sp]
// CHECK-ENCODING: [0xef,0x7f,0x3f,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef 7f 3f e0 <unknown>

st1b    {za0h.b[w12, 5]}, p3, [x17, x16]
// CHECK-INST: st1b    {za0h.b[w12, 5]}, p3, [x17, x16]
// CHECK-ENCODING: [0x25,0x0e,0x30,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 0e 30 e0 <unknown>

st1b    {za0h.b[w12, 1]}, p1, [x1, x30]
// CHECK-INST: st1b    {za0h.b[w12, 1]}, p1, [x1, x30]
// CHECK-ENCODING: [0x21,0x04,0x3e,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 04 3e e0 <unknown>

st1b    {za0h.b[w14, 8]}, p5, [x19, x20]
// CHECK-INST: st1b    {za0h.b[w14, 8]}, p5, [x19, x20]
// CHECK-ENCODING: [0x68,0x56,0x34,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 56 34 e0 <unknown>

st1b    {za0h.b[w12, 0]}, p6, [x12, x2]
// CHECK-INST: st1b    {za0h.b[w12, 0]}, p6, [x12, x2]
// CHECK-ENCODING: [0x80,0x19,0x22,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 19 22 e0 <unknown>

st1b    {za0h.b[w14, 1]}, p2, [x1, x26]
// CHECK-INST: st1b    {za0h.b[w14, 1]}, p2, [x1, x26]
// CHECK-ENCODING: [0x21,0x48,0x3a,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 48 3a e0 <unknown>

st1b    {za0h.b[w12, 13]}, p2, [x22, x30]
// CHECK-INST: st1b    {za0h.b[w12, 13]}, p2, [x22, x30]
// CHECK-ENCODING: [0xcd,0x0a,0x3e,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 0a 3e e0 <unknown>

st1b    {za0h.b[w15, 2]}, p5, [x9, x1]
// CHECK-INST: st1b    {za0h.b[w15, 2]}, p5, [x9, x1]
// CHECK-ENCODING: [0x22,0x75,0x21,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 75 21 e0 <unknown>

st1b    {za0h.b[w13, 7]}, p2, [x12, x11]
// CHECK-INST: st1b    {za0h.b[w13, 7]}, p2, [x12, x11]
// CHECK-ENCODING: [0x87,0x29,0x2b,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 29 2b e0 <unknown>

st1b    za0h.b[w12, 0], p0, [x0, x0]
// CHECK-INST: st1b    {za0h.b[w12, 0]}, p0, [x0, x0]
// CHECK-ENCODING: [0x00,0x00,0x20,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 00 20 e0 <unknown>

st1b    za0h.b[w14, 5], p5, [x10, x21]
// CHECK-INST: st1b    {za0h.b[w14, 5]}, p5, [x10, x21]
// CHECK-ENCODING: [0x45,0x55,0x35,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 55 35 e0 <unknown>

st1b    za0h.b[w15, 7], p3, [x13, x8]
// CHECK-INST: st1b    {za0h.b[w15, 7]}, p3, [x13, x8]
// CHECK-ENCODING: [0xa7,0x6d,0x28,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 6d 28 e0 <unknown>

st1b    za0h.b[w15, 15], p7, [sp]
// CHECK-INST: st1b    {za0h.b[w15, 15]}, p7, [sp]
// CHECK-ENCODING: [0xef,0x7f,0x3f,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef 7f 3f e0 <unknown>

st1b    za0h.b[w12, 5], p3, [x17, x16]
// CHECK-INST: st1b    {za0h.b[w12, 5]}, p3, [x17, x16]
// CHECK-ENCODING: [0x25,0x0e,0x30,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 0e 30 e0 <unknown>

st1b    za0h.b[w12, 1], p1, [x1, x30]
// CHECK-INST: st1b    {za0h.b[w12, 1]}, p1, [x1, x30]
// CHECK-ENCODING: [0x21,0x04,0x3e,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 04 3e e0 <unknown>

st1b    za0h.b[w14, 8], p5, [x19, x20]
// CHECK-INST: st1b    {za0h.b[w14, 8]}, p5, [x19, x20]
// CHECK-ENCODING: [0x68,0x56,0x34,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 56 34 e0 <unknown>

st1b    za0h.b[w12, 0], p6, [x12, x2]
// CHECK-INST: st1b    {za0h.b[w12, 0]}, p6, [x12, x2]
// CHECK-ENCODING: [0x80,0x19,0x22,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 19 22 e0 <unknown>

st1b    za0h.b[w14, 1], p2, [x1, x26]
// CHECK-INST: st1b    {za0h.b[w14, 1]}, p2, [x1, x26]
// CHECK-ENCODING: [0x21,0x48,0x3a,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 48 3a e0 <unknown>

st1b    za0h.b[w12, 13], p2, [x22, x30]
// CHECK-INST: st1b    {za0h.b[w12, 13]}, p2, [x22, x30]
// CHECK-ENCODING: [0xcd,0x0a,0x3e,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 0a 3e e0 <unknown>

st1b    za0h.b[w15, 2], p5, [x9, x1]
// CHECK-INST: st1b    {za0h.b[w15, 2]}, p5, [x9, x1]
// CHECK-ENCODING: [0x22,0x75,0x21,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 75 21 e0 <unknown>

st1b    za0h.b[w13, 7], p2, [x12, x11]
// CHECK-INST: st1b    {za0h.b[w13, 7]}, p2, [x12, x11]
// CHECK-ENCODING: [0x87,0x29,0x2b,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 29 2b e0 <unknown>

// --------------------------------------------------------------------------//
// Vertical

st1b    {za0v.b[w12, 0]}, p0, [x0, x0]
// CHECK-INST: st1b    {za0v.b[w12, 0]}, p0, [x0, x0]
// CHECK-ENCODING: [0x00,0x80,0x20,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 80 20 e0 <unknown>

st1b    {za0v.b[w14, 5]}, p5, [x10, x21]
// CHECK-INST: st1b    {za0v.b[w14, 5]}, p5, [x10, x21]
// CHECK-ENCODING: [0x45,0xd5,0x35,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 d5 35 e0 <unknown>

st1b    {za0v.b[w15, 7]}, p3, [x13, x8]
// CHECK-INST: st1b    {za0v.b[w15, 7]}, p3, [x13, x8]
// CHECK-ENCODING: [0xa7,0xed,0x28,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 ed 28 e0 <unknown>

st1b    {za0v.b[w15, 15]}, p7, [sp]
// CHECK-INST: st1b    {za0v.b[w15, 15]}, p7, [sp]
// CHECK-ENCODING: [0xef,0xff,0x3f,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef ff 3f e0 <unknown>

st1b    {za0v.b[w12, 5]}, p3, [x17, x16]
// CHECK-INST: st1b    {za0v.b[w12, 5]}, p3, [x17, x16]
// CHECK-ENCODING: [0x25,0x8e,0x30,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 8e 30 e0 <unknown>

st1b    {za0v.b[w12, 1]}, p1, [x1, x30]
// CHECK-INST: st1b    {za0v.b[w12, 1]}, p1, [x1, x30]
// CHECK-ENCODING: [0x21,0x84,0x3e,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 84 3e e0 <unknown>

st1b    {za0v.b[w14, 8]}, p5, [x19, x20]
// CHECK-INST: st1b    {za0v.b[w14, 8]}, p5, [x19, x20]
// CHECK-ENCODING: [0x68,0xd6,0x34,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 d6 34 e0 <unknown>

st1b    {za0v.b[w12, 0]}, p6, [x12, x2]
// CHECK-INST: st1b    {za0v.b[w12, 0]}, p6, [x12, x2]
// CHECK-ENCODING: [0x80,0x99,0x22,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 99 22 e0 <unknown>

st1b    {za0v.b[w14, 1]}, p2, [x1, x26]
// CHECK-INST: st1b    {za0v.b[w14, 1]}, p2, [x1, x26]
// CHECK-ENCODING: [0x21,0xc8,0x3a,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 c8 3a e0 <unknown>

st1b    {za0v.b[w12, 13]}, p2, [x22, x30]
// CHECK-INST: st1b    {za0v.b[w12, 13]}, p2, [x22, x30]
// CHECK-ENCODING: [0xcd,0x8a,0x3e,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 8a 3e e0 <unknown>

st1b    {za0v.b[w15, 2]}, p5, [x9, x1]
// CHECK-INST: st1b    {za0v.b[w15, 2]}, p5, [x9, x1]
// CHECK-ENCODING: [0x22,0xf5,0x21,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 f5 21 e0 <unknown>

st1b    {za0v.b[w13, 7]}, p2, [x12, x11]
// CHECK-INST: st1b    {za0v.b[w13, 7]}, p2, [x12, x11]
// CHECK-ENCODING: [0x87,0xa9,0x2b,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 a9 2b e0 <unknown>

st1b    za0v.b[w12, 0], p0, [x0, x0]
// CHECK-INST: st1b    {za0v.b[w12, 0]}, p0, [x0, x0]
// CHECK-ENCODING: [0x00,0x80,0x20,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 80 20 e0 <unknown>

st1b    za0v.b[w14, 5], p5, [x10, x21]
// CHECK-INST: st1b    {za0v.b[w14, 5]}, p5, [x10, x21]
// CHECK-ENCODING: [0x45,0xd5,0x35,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 d5 35 e0 <unknown>

st1b    za0v.b[w15, 7], p3, [x13, x8]
// CHECK-INST: st1b    {za0v.b[w15, 7]}, p3, [x13, x8]
// CHECK-ENCODING: [0xa7,0xed,0x28,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 ed 28 e0 <unknown>

st1b    za0v.b[w15, 15], p7, [sp]
// CHECK-INST: st1b    {za0v.b[w15, 15]}, p7, [sp]
// CHECK-ENCODING: [0xef,0xff,0x3f,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef ff 3f e0 <unknown>

st1b    za0v.b[w12, 5], p3, [x17, x16]
// CHECK-INST: st1b    {za0v.b[w12, 5]}, p3, [x17, x16]
// CHECK-ENCODING: [0x25,0x8e,0x30,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 8e 30 e0 <unknown>

st1b    za0v.b[w12, 1], p1, [x1, x30]
// CHECK-INST: st1b    {za0v.b[w12, 1]}, p1, [x1, x30]
// CHECK-ENCODING: [0x21,0x84,0x3e,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 84 3e e0 <unknown>

st1b    za0v.b[w14, 8], p5, [x19, x20]
// CHECK-INST: st1b    {za0v.b[w14, 8]}, p5, [x19, x20]
// CHECK-ENCODING: [0x68,0xd6,0x34,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 d6 34 e0 <unknown>

st1b    za0v.b[w12, 0], p6, [x12, x2]
// CHECK-INST: st1b    {za0v.b[w12, 0]}, p6, [x12, x2]
// CHECK-ENCODING: [0x80,0x99,0x22,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 99 22 e0 <unknown>

st1b    za0v.b[w14, 1], p2, [x1, x26]
// CHECK-INST: st1b    {za0v.b[w14, 1]}, p2, [x1, x26]
// CHECK-ENCODING: [0x21,0xc8,0x3a,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 c8 3a e0 <unknown>

st1b    za0v.b[w12, 13], p2, [x22, x30]
// CHECK-INST: st1b    {za0v.b[w12, 13]}, p2, [x22, x30]
// CHECK-ENCODING: [0xcd,0x8a,0x3e,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 8a 3e e0 <unknown>

st1b    za0v.b[w15, 2], p5, [x9, x1]
// CHECK-INST: st1b    {za0v.b[w15, 2]}, p5, [x9, x1]
// CHECK-ENCODING: [0x22,0xf5,0x21,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 f5 21 e0 <unknown>

st1b    za0v.b[w13, 7], p2, [x12, x11]
// CHECK-INST: st1b    {za0v.b[w13, 7]}, p2, [x12, x11]
// CHECK-ENCODING: [0x87,0xa9,0x2b,0xe0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 a9 2b e0 <unknown>
