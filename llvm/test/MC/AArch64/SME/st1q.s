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

st1q    {za0h.q[w12, 0]}, p0, [x0, x0, lsl #4]
// CHECK-INST: st1q    {za0h.q[w12, 0]}, p0, [x0, x0, lsl #4]
// CHECK-ENCODING: [0x00,0x00,0xe0,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 00 e0 e1 <unknown>

st1q    {za5h.q[w14, 0]}, p5, [x10, x21, lsl #4]
// CHECK-INST: st1q    {za5h.q[w14, 0]}, p5, [x10, x21, lsl #4]
// CHECK-ENCODING: [0x45,0x55,0xf5,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 55 f5 e1 <unknown>

st1q    {za7h.q[w15, 0]}, p3, [x13, x8, lsl #4]
// CHECK-INST: st1q    {za7h.q[w15, 0]}, p3, [x13, x8, lsl #4]
// CHECK-ENCODING: [0xa7,0x6d,0xe8,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 6d e8 e1 <unknown>

st1q    {za15h.q[w15, 0]}, p7, [sp]
// CHECK-INST: st1q    {za15h.q[w15, 0]}, p7, [sp]
// CHECK-ENCODING: [0xef,0x7f,0xff,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef 7f ff e1 <unknown>

st1q    {za5h.q[w12, 0]}, p3, [x17, x16, lsl #4]
// CHECK-INST: st1q    {za5h.q[w12, 0]}, p3, [x17, x16, lsl #4]
// CHECK-ENCODING: [0x25,0x0e,0xf0,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 0e f0 e1 <unknown>

st1q    {za1h.q[w12, 0]}, p1, [x1, x30, lsl #4]
// CHECK-INST: st1q    {za1h.q[w12, 0]}, p1, [x1, x30, lsl #4]
// CHECK-ENCODING: [0x21,0x04,0xfe,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 04 fe e1 <unknown>

st1q    {za8h.q[w14, 0]}, p5, [x19, x20, lsl #4]
// CHECK-INST: st1q    {za8h.q[w14, 0]}, p5, [x19, x20, lsl #4]
// CHECK-ENCODING: [0x68,0x56,0xf4,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 56 f4 e1 <unknown>

st1q    {za0h.q[w12, 0]}, p6, [x12, x2, lsl #4]
// CHECK-INST: st1q    {za0h.q[w12, 0]}, p6, [x12, x2, lsl #4]
// CHECK-ENCODING: [0x80,0x19,0xe2,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 19 e2 e1 <unknown>

st1q    {za1h.q[w14, 0]}, p2, [x1, x26, lsl #4]
// CHECK-INST: st1q    {za1h.q[w14, 0]}, p2, [x1, x26, lsl #4]
// CHECK-ENCODING: [0x21,0x48,0xfa,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 48 fa e1 <unknown>

st1q    {za13h.q[w12, 0]}, p2, [x22, x30, lsl #4]
// CHECK-INST: st1q    {za13h.q[w12, 0]}, p2, [x22, x30, lsl #4]
// CHECK-ENCODING: [0xcd,0x0a,0xfe,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 0a fe e1 <unknown>

st1q    {za2h.q[w15, 0]}, p5, [x9, x1, lsl #4]
// CHECK-INST: st1q    {za2h.q[w15, 0]}, p5, [x9, x1, lsl #4]
// CHECK-ENCODING: [0x22,0x75,0xe1,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 75 e1 e1 <unknown>

st1q    {za7h.q[w13, 0]}, p2, [x12, x11, lsl #4]
// CHECK-INST: st1q    {za7h.q[w13, 0]}, p2, [x12, x11, lsl #4]
// CHECK-ENCODING: [0x87,0x29,0xeb,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 29 eb e1 <unknown>

st1q    za0h.q[w12, 0], p0, [x0, x0, lsl #4]
// CHECK-INST: st1q    {za0h.q[w12, 0]}, p0, [x0, x0, lsl #4]
// CHECK-ENCODING: [0x00,0x00,0xe0,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 00 e0 e1 <unknown>

st1q    za5h.q[w14, 0], p5, [x10, x21, lsl #4]
// CHECK-INST: st1q    {za5h.q[w14, 0]}, p5, [x10, x21, lsl #4]
// CHECK-ENCODING: [0x45,0x55,0xf5,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 55 f5 e1 <unknown>

st1q    za7h.q[w15, 0], p3, [x13, x8, lsl #4]
// CHECK-INST: st1q    {za7h.q[w15, 0]}, p3, [x13, x8, lsl #4]
// CHECK-ENCODING: [0xa7,0x6d,0xe8,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 6d e8 e1 <unknown>

st1q    za15h.q[w15, 0], p7, [sp]
// CHECK-INST: st1q    {za15h.q[w15, 0]}, p7, [sp]
// CHECK-ENCODING: [0xef,0x7f,0xff,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef 7f ff e1 <unknown>

st1q    za5h.q[w12, 0], p3, [x17, x16, lsl #4]
// CHECK-INST: st1q    {za5h.q[w12, 0]}, p3, [x17, x16, lsl #4]
// CHECK-ENCODING: [0x25,0x0e,0xf0,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 0e f0 e1 <unknown>

st1q    za1h.q[w12, 0], p1, [x1, x30, lsl #4]
// CHECK-INST: st1q    {za1h.q[w12, 0]}, p1, [x1, x30, lsl #4]
// CHECK-ENCODING: [0x21,0x04,0xfe,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 04 fe e1 <unknown>

st1q    za8h.q[w14, 0], p5, [x19, x20, lsl #4]
// CHECK-INST: st1q    {za8h.q[w14, 0]}, p5, [x19, x20, lsl #4]
// CHECK-ENCODING: [0x68,0x56,0xf4,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 56 f4 e1 <unknown>

st1q    za0h.q[w12, 0], p6, [x12, x2, lsl #4]
// CHECK-INST: st1q    {za0h.q[w12, 0]}, p6, [x12, x2, lsl #4]
// CHECK-ENCODING: [0x80,0x19,0xe2,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 19 e2 e1 <unknown>

st1q    za1h.q[w14, 0], p2, [x1, x26, lsl #4]
// CHECK-INST: st1q    {za1h.q[w14, 0]}, p2, [x1, x26, lsl #4]
// CHECK-ENCODING: [0x21,0x48,0xfa,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 48 fa e1 <unknown>

st1q    za13h.q[w12, 0], p2, [x22, x30, lsl #4]
// CHECK-INST: st1q    {za13h.q[w12, 0]}, p2, [x22, x30, lsl #4]
// CHECK-ENCODING: [0xcd,0x0a,0xfe,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 0a fe e1 <unknown>

st1q    za2h.q[w15, 0], p5, [x9, x1, lsl #4]
// CHECK-INST: st1q    {za2h.q[w15, 0]}, p5, [x9, x1, lsl #4]
// CHECK-ENCODING: [0x22,0x75,0xe1,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 75 e1 e1 <unknown>

st1q    za7h.q[w13, 0], p2, [x12, x11, lsl #4]
// CHECK-INST: st1q    {za7h.q[w13, 0]}, p2, [x12, x11, lsl #4]
// CHECK-ENCODING: [0x87,0x29,0xeb,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 29 eb e1 <unknown>

// --------------------------------------------------------------------------//
// Vertical

st1q    {za0v.q[w12, 0]}, p0, [x0, x0, lsl #4]
// CHECK-INST: st1q    {za0v.q[w12, 0]}, p0, [x0, x0, lsl #4]
// CHECK-ENCODING: [0x00,0x80,0xe0,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 80 e0 e1 <unknown>

st1q    {za5v.q[w14, 0]}, p5, [x10, x21, lsl #4]
// CHECK-INST: st1q    {za5v.q[w14, 0]}, p5, [x10, x21, lsl #4]
// CHECK-ENCODING: [0x45,0xd5,0xf5,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 d5 f5 e1 <unknown>

st1q    {za7v.q[w15, 0]}, p3, [x13, x8, lsl #4]
// CHECK-INST: st1q    {za7v.q[w15, 0]}, p3, [x13, x8, lsl #4]
// CHECK-ENCODING: [0xa7,0xed,0xe8,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 ed e8 e1 <unknown>

st1q    {za15v.q[w15, 0]}, p7, [sp]
// CHECK-INST: st1q    {za15v.q[w15, 0]}, p7, [sp]
// CHECK-ENCODING: [0xef,0xff,0xff,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef ff ff e1 <unknown>

st1q    {za5v.q[w12, 0]}, p3, [x17, x16, lsl #4]
// CHECK-INST: st1q    {za5v.q[w12, 0]}, p3, [x17, x16, lsl #4]
// CHECK-ENCODING: [0x25,0x8e,0xf0,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 8e f0 e1 <unknown>

st1q    {za1v.q[w12, 0]}, p1, [x1, x30, lsl #4]
// CHECK-INST: st1q    {za1v.q[w12, 0]}, p1, [x1, x30, lsl #4]
// CHECK-ENCODING: [0x21,0x84,0xfe,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 84 fe e1 <unknown>

st1q    {za8v.q[w14, 0]}, p5, [x19, x20, lsl #4]
// CHECK-INST: st1q    {za8v.q[w14, 0]}, p5, [x19, x20, lsl #4]
// CHECK-ENCODING: [0x68,0xd6,0xf4,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 d6 f4 e1 <unknown>

st1q    {za0v.q[w12, 0]}, p6, [x12, x2, lsl #4]
// CHECK-INST: st1q    {za0v.q[w12, 0]}, p6, [x12, x2, lsl #4]
// CHECK-ENCODING: [0x80,0x99,0xe2,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 99 e2 e1 <unknown>

st1q    {za1v.q[w14, 0]}, p2, [x1, x26, lsl #4]
// CHECK-INST: st1q    {za1v.q[w14, 0]}, p2, [x1, x26, lsl #4]
// CHECK-ENCODING: [0x21,0xc8,0xfa,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 c8 fa e1 <unknown>

st1q    {za13v.q[w12, 0]}, p2, [x22, x30, lsl #4]
// CHECK-INST: st1q    {za13v.q[w12, 0]}, p2, [x22, x30, lsl #4]
// CHECK-ENCODING: [0xcd,0x8a,0xfe,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 8a fe e1 <unknown>

st1q    {za2v.q[w15, 0]}, p5, [x9, x1, lsl #4]
// CHECK-INST: st1q    {za2v.q[w15, 0]}, p5, [x9, x1, lsl #4]
// CHECK-ENCODING: [0x22,0xf5,0xe1,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 f5 e1 e1 <unknown>

st1q    {za7v.q[w13, 0]}, p2, [x12, x11, lsl #4]
// CHECK-INST: st1q    {za7v.q[w13, 0]}, p2, [x12, x11, lsl #4]
// CHECK-ENCODING: [0x87,0xa9,0xeb,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 a9 eb e1 <unknown>

st1q    za0v.q[w12, 0], p0, [x0, x0, lsl #4]
// CHECK-INST: st1q    {za0v.q[w12, 0]}, p0, [x0, x0, lsl #4]
// CHECK-ENCODING: [0x00,0x80,0xe0,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 80 e0 e1 <unknown>

st1q    za5v.q[w14, 0], p5, [x10, x21, lsl #4]
// CHECK-INST: st1q    {za5v.q[w14, 0]}, p5, [x10, x21, lsl #4]
// CHECK-ENCODING: [0x45,0xd5,0xf5,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 d5 f5 e1 <unknown>

st1q    za7v.q[w15, 0], p3, [x13, x8, lsl #4]
// CHECK-INST: st1q    {za7v.q[w15, 0]}, p3, [x13, x8, lsl #4]
// CHECK-ENCODING: [0xa7,0xed,0xe8,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 ed e8 e1 <unknown>

st1q    za15v.q[w15, 0], p7, [sp]
// CHECK-INST: st1q    {za15v.q[w15, 0]}, p7, [sp]
// CHECK-ENCODING: [0xef,0xff,0xff,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef ff ff e1 <unknown>

st1q    za5v.q[w12, 0], p3, [x17, x16, lsl #4]
// CHECK-INST: st1q    {za5v.q[w12, 0]}, p3, [x17, x16, lsl #4]
// CHECK-ENCODING: [0x25,0x8e,0xf0,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 8e f0 e1 <unknown>

st1q    za1v.q[w12, 0], p1, [x1, x30, lsl #4]
// CHECK-INST: st1q    {za1v.q[w12, 0]}, p1, [x1, x30, lsl #4]
// CHECK-ENCODING: [0x21,0x84,0xfe,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 84 fe e1 <unknown>

st1q    za8v.q[w14, 0], p5, [x19, x20, lsl #4]
// CHECK-INST: st1q    {za8v.q[w14, 0]}, p5, [x19, x20, lsl #4]
// CHECK-ENCODING: [0x68,0xd6,0xf4,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 d6 f4 e1 <unknown>

st1q    za0v.q[w12, 0], p6, [x12, x2, lsl #4]
// CHECK-INST: st1q    {za0v.q[w12, 0]}, p6, [x12, x2, lsl #4]
// CHECK-ENCODING: [0x80,0x99,0xe2,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 99 e2 e1 <unknown>

st1q    za1v.q[w14, 0], p2, [x1, x26, lsl #4]
// CHECK-INST: st1q    {za1v.q[w14, 0]}, p2, [x1, x26, lsl #4]
// CHECK-ENCODING: [0x21,0xc8,0xfa,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 c8 fa e1 <unknown>

st1q    za13v.q[w12, 0], p2, [x22, x30, lsl #4]
// CHECK-INST: st1q    {za13v.q[w12, 0]}, p2, [x22, x30, lsl #4]
// CHECK-ENCODING: [0xcd,0x8a,0xfe,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 8a fe e1 <unknown>

st1q    za2v.q[w15, 0], p5, [x9, x1, lsl #4]
// CHECK-INST: st1q    {za2v.q[w15, 0]}, p5, [x9, x1, lsl #4]
// CHECK-ENCODING: [0x22,0xf5,0xe1,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 f5 e1 e1 <unknown>

st1q    za7v.q[w13, 0], p2, [x12, x11, lsl #4]
// CHECK-INST: st1q    {za7v.q[w13, 0]}, p2, [x12, x11, lsl #4]
// CHECK-ENCODING: [0x87,0xa9,0xeb,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 a9 eb e1 <unknown>
