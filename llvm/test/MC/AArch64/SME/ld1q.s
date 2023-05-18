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

ld1q    {za0h.q[w12, 0]}, p0/z, [x0, x0, lsl #4]
// CHECK-INST: ld1q    {za0h.q[w12, 0]}, p0/z, [x0, x0, lsl #4]
// CHECK-ENCODING: [0x00,0x00,0xc0,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1c00000 <unknown>

ld1q    {za5h.q[w14, 0]}, p5/z, [x10, x21, lsl #4]
// CHECK-INST: ld1q    {za5h.q[w14, 0]}, p5/z, [x10, x21, lsl #4]
// CHECK-ENCODING: [0x45,0x55,0xd5,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1d55545 <unknown>

ld1q    {za7h.q[w15, 0]}, p3/z, [x13, x8, lsl #4]
// CHECK-INST: ld1q    {za7h.q[w15, 0]}, p3/z, [x13, x8, lsl #4]
// CHECK-ENCODING: [0xa7,0x6d,0xc8,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1c86da7 <unknown>

ld1q    {za15h.q[w15, 0]}, p7/z, [sp]
// CHECK-INST: ld1q    {za15h.q[w15, 0]}, p7/z, [sp]
// CHECK-ENCODING: [0xef,0x7f,0xdf,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1df7fef <unknown>

ld1q    {za5h.q[w12, 0]}, p3/z, [x17, x16, lsl #4]
// CHECK-INST: ld1q    {za5h.q[w12, 0]}, p3/z, [x17, x16, lsl #4]
// CHECK-ENCODING: [0x25,0x0e,0xd0,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1d00e25 <unknown>

ld1q    {za1h.q[w12, 0]}, p1/z, [x1, x30, lsl #4]
// CHECK-INST: ld1q    {za1h.q[w12, 0]}, p1/z, [x1, x30, lsl #4]
// CHECK-ENCODING: [0x21,0x04,0xde,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1de0421 <unknown>

ld1q    {za8h.q[w14, 0]}, p5/z, [x19, x20, lsl #4]
// CHECK-INST: ld1q    {za8h.q[w14, 0]}, p5/z, [x19, x20, lsl #4]
// CHECK-ENCODING: [0x68,0x56,0xd4,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1d45668 <unknown>

ld1q    {za0h.q[w12, 0]}, p6/z, [x12, x2, lsl #4]
// CHECK-INST: ld1q    {za0h.q[w12, 0]}, p6/z, [x12, x2, lsl #4]
// CHECK-ENCODING: [0x80,0x19,0xc2,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1c21980 <unknown>

ld1q    {za1h.q[w14, 0]}, p2/z, [x1, x26, lsl #4]
// CHECK-INST: ld1q    {za1h.q[w14, 0]}, p2/z, [x1, x26, lsl #4]
// CHECK-ENCODING: [0x21,0x48,0xda,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1da4821 <unknown>

ld1q    {za13h.q[w12, 0]}, p2/z, [x22, x30, lsl #4]
// CHECK-INST: ld1q    {za13h.q[w12, 0]}, p2/z, [x22, x30, lsl #4]
// CHECK-ENCODING: [0xcd,0x0a,0xde,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1de0acd <unknown>

ld1q    {za2h.q[w15, 0]}, p5/z, [x9, x1, lsl #4]
// CHECK-INST: ld1q    {za2h.q[w15, 0]}, p5/z, [x9, x1, lsl #4]
// CHECK-ENCODING: [0x22,0x75,0xc1,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1c17522 <unknown>

ld1q    {za7h.q[w13, 0]}, p2/z, [x12, x11, lsl #4]
// CHECK-INST: ld1q    {za7h.q[w13, 0]}, p2/z, [x12, x11, lsl #4]
// CHECK-ENCODING: [0x87,0x29,0xcb,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1cb2987 <unknown>

ld1q    za0h.q[w12, 0], p0/z, [x0, x0, lsl #4]
// CHECK-INST: ld1q    {za0h.q[w12, 0]}, p0/z, [x0, x0, lsl #4]
// CHECK-ENCODING: [0x00,0x00,0xc0,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1c00000 <unknown>

ld1q    za5h.q[w14, 0], p5/z, [x10, x21, lsl #4]
// CHECK-INST: ld1q    {za5h.q[w14, 0]}, p5/z, [x10, x21, lsl #4]
// CHECK-ENCODING: [0x45,0x55,0xd5,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1d55545 <unknown>

ld1q    za7h.q[w15, 0], p3/z, [x13, x8, lsl #4]
// CHECK-INST: ld1q    {za7h.q[w15, 0]}, p3/z, [x13, x8, lsl #4]
// CHECK-ENCODING: [0xa7,0x6d,0xc8,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1c86da7 <unknown>

ld1q    za15h.q[w15, 0], p7/z, [sp]
// CHECK-INST: ld1q    {za15h.q[w15, 0]}, p7/z, [sp]
// CHECK-ENCODING: [0xef,0x7f,0xdf,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1df7fef <unknown>

ld1q    za5h.q[w12, 0], p3/z, [x17, x16, lsl #4]
// CHECK-INST: ld1q    {za5h.q[w12, 0]}, p3/z, [x17, x16, lsl #4]
// CHECK-ENCODING: [0x25,0x0e,0xd0,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1d00e25 <unknown>

ld1q    za1h.q[w12, 0], p1/z, [x1, x30, lsl #4]
// CHECK-INST: ld1q    {za1h.q[w12, 0]}, p1/z, [x1, x30, lsl #4]
// CHECK-ENCODING: [0x21,0x04,0xde,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1de0421 <unknown>

ld1q    za8h.q[w14, 0], p5/z, [x19, x20, lsl #4]
// CHECK-INST: ld1q    {za8h.q[w14, 0]}, p5/z, [x19, x20, lsl #4]
// CHECK-ENCODING: [0x68,0x56,0xd4,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1d45668 <unknown>

ld1q    za0h.q[w12, 0], p6/z, [x12, x2, lsl #4]
// CHECK-INST: ld1q    {za0h.q[w12, 0]}, p6/z, [x12, x2, lsl #4]
// CHECK-ENCODING: [0x80,0x19,0xc2,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1c21980 <unknown>

ld1q    za1h.q[w14, 0], p2/z, [x1, x26, lsl #4]
// CHECK-INST: ld1q    {za1h.q[w14, 0]}, p2/z, [x1, x26, lsl #4]
// CHECK-ENCODING: [0x21,0x48,0xda,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1da4821 <unknown>

ld1q    za13h.q[w12, 0], p2/z, [x22, x30, lsl #4]
// CHECK-INST: ld1q    {za13h.q[w12, 0]}, p2/z, [x22, x30, lsl #4]
// CHECK-ENCODING: [0xcd,0x0a,0xde,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1de0acd <unknown>

ld1q    za2h.q[w15, 0], p5/z, [x9, x1, lsl #4]
// CHECK-INST: ld1q    {za2h.q[w15, 0]}, p5/z, [x9, x1, lsl #4]
// CHECK-ENCODING: [0x22,0x75,0xc1,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1c17522 <unknown>

ld1q    za7h.q[w13, 0], p2/z, [x12, x11, lsl #4]
// CHECK-INST: ld1q    {za7h.q[w13, 0]}, p2/z, [x12, x11, lsl #4]
// CHECK-ENCODING: [0x87,0x29,0xcb,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1cb2987 <unknown>

// --------------------------------------------------------------------------//
// Vertical

ld1q    {za0v.q[w12, 0]}, p0/z, [x0, x0, lsl #4]
// CHECK-INST: ld1q    {za0v.q[w12, 0]}, p0/z, [x0, x0, lsl #4]
// CHECK-ENCODING: [0x00,0x80,0xc0,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1c08000 <unknown>

ld1q    {za5v.q[w14, 0]}, p5/z, [x10, x21, lsl #4]
// CHECK-INST: ld1q    {za5v.q[w14, 0]}, p5/z, [x10, x21, lsl #4]
// CHECK-ENCODING: [0x45,0xd5,0xd5,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1d5d545 <unknown>

ld1q    {za7v.q[w15, 0]}, p3/z, [x13, x8, lsl #4]
// CHECK-INST: ld1q    {za7v.q[w15, 0]}, p3/z, [x13, x8, lsl #4]
// CHECK-ENCODING: [0xa7,0xed,0xc8,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1c8eda7 <unknown>

ld1q    {za15v.q[w15, 0]}, p7/z, [sp]
// CHECK-INST: ld1q    {za15v.q[w15, 0]}, p7/z, [sp]
// CHECK-ENCODING: [0xef,0xff,0xdf,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1dfffef <unknown>

ld1q    {za5v.q[w12, 0]}, p3/z, [x17, x16, lsl #4]
// CHECK-INST: ld1q    {za5v.q[w12, 0]}, p3/z, [x17, x16, lsl #4]
// CHECK-ENCODING: [0x25,0x8e,0xd0,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1d08e25 <unknown>

ld1q    {za1v.q[w12, 0]}, p1/z, [x1, x30, lsl #4]
// CHECK-INST: ld1q    {za1v.q[w12, 0]}, p1/z, [x1, x30, lsl #4]
// CHECK-ENCODING: [0x21,0x84,0xde,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1de8421 <unknown>

ld1q    {za8v.q[w14, 0]}, p5/z, [x19, x20, lsl #4]
// CHECK-INST: ld1q    {za8v.q[w14, 0]}, p5/z, [x19, x20, lsl #4]
// CHECK-ENCODING: [0x68,0xd6,0xd4,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1d4d668 <unknown>

ld1q    {za0v.q[w12, 0]}, p6/z, [x12, x2, lsl #4]
// CHECK-INST: ld1q    {za0v.q[w12, 0]}, p6/z, [x12, x2, lsl #4]
// CHECK-ENCODING: [0x80,0x99,0xc2,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1c29980 <unknown>

ld1q    {za1v.q[w14, 0]}, p2/z, [x1, x26, lsl #4]
// CHECK-INST: ld1q    {za1v.q[w14, 0]}, p2/z, [x1, x26, lsl #4]
// CHECK-ENCODING: [0x21,0xc8,0xda,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1dac821 <unknown>

ld1q    {za13v.q[w12, 0]}, p2/z, [x22, x30, lsl #4]
// CHECK-INST: ld1q    {za13v.q[w12, 0]}, p2/z, [x22, x30, lsl #4]
// CHECK-ENCODING: [0xcd,0x8a,0xde,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1de8acd <unknown>

ld1q    {za2v.q[w15, 0]}, p5/z, [x9, x1, lsl #4]
// CHECK-INST: ld1q    {za2v.q[w15, 0]}, p5/z, [x9, x1, lsl #4]
// CHECK-ENCODING: [0x22,0xf5,0xc1,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1c1f522 <unknown>

ld1q    {za7v.q[w13, 0]}, p2/z, [x12, x11, lsl #4]
// CHECK-INST: ld1q    {za7v.q[w13, 0]}, p2/z, [x12, x11, lsl #4]
// CHECK-ENCODING: [0x87,0xa9,0xcb,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1cba987 <unknown>

ld1q    za0v.q[w12, 0], p0/z, [x0, x0, lsl #4]
// CHECK-INST: ld1q    {za0v.q[w12, 0]}, p0/z, [x0, x0, lsl #4]
// CHECK-ENCODING: [0x00,0x80,0xc0,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1c08000 <unknown>

ld1q    za5v.q[w14, 0], p5/z, [x10, x21, lsl #4]
// CHECK-INST: ld1q    {za5v.q[w14, 0]}, p5/z, [x10, x21, lsl #4]
// CHECK-ENCODING: [0x45,0xd5,0xd5,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1d5d545 <unknown>

ld1q    za7v.q[w15, 0], p3/z, [x13, x8, lsl #4]
// CHECK-INST: ld1q    {za7v.q[w15, 0]}, p3/z, [x13, x8, lsl #4]
// CHECK-ENCODING: [0xa7,0xed,0xc8,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1c8eda7 <unknown>

ld1q    za15v.q[w15, 0], p7/z, [sp]
// CHECK-INST: ld1q    {za15v.q[w15, 0]}, p7/z, [sp]
// CHECK-ENCODING: [0xef,0xff,0xdf,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1dfffef <unknown>

ld1q    za5v.q[w12, 0], p3/z, [x17, x16, lsl #4]
// CHECK-INST: ld1q    {za5v.q[w12, 0]}, p3/z, [x17, x16, lsl #4]
// CHECK-ENCODING: [0x25,0x8e,0xd0,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1d08e25 <unknown>

ld1q    za1v.q[w12, 0], p1/z, [x1, x30, lsl #4]
// CHECK-INST: ld1q    {za1v.q[w12, 0]}, p1/z, [x1, x30, lsl #4]
// CHECK-ENCODING: [0x21,0x84,0xde,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1de8421 <unknown>

ld1q    za8v.q[w14, 0], p5/z, [x19, x20, lsl #4]
// CHECK-INST: ld1q    {za8v.q[w14, 0]}, p5/z, [x19, x20, lsl #4]
// CHECK-ENCODING: [0x68,0xd6,0xd4,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1d4d668 <unknown>

ld1q    za0v.q[w12, 0], p6/z, [x12, x2, lsl #4]
// CHECK-INST: ld1q    {za0v.q[w12, 0]}, p6/z, [x12, x2, lsl #4]
// CHECK-ENCODING: [0x80,0x99,0xc2,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1c29980 <unknown>

ld1q    za1v.q[w14, 0], p2/z, [x1, x26, lsl #4]
// CHECK-INST: ld1q    {za1v.q[w14, 0]}, p2/z, [x1, x26, lsl #4]
// CHECK-ENCODING: [0x21,0xc8,0xda,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1dac821 <unknown>

ld1q    za13v.q[w12, 0], p2/z, [x22, x30, lsl #4]
// CHECK-INST: ld1q    {za13v.q[w12, 0]}, p2/z, [x22, x30, lsl #4]
// CHECK-ENCODING: [0xcd,0x8a,0xde,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1de8acd <unknown>

ld1q    za2v.q[w15, 0], p5/z, [x9, x1, lsl #4]
// CHECK-INST: ld1q    {za2v.q[w15, 0]}, p5/z, [x9, x1, lsl #4]
// CHECK-ENCODING: [0x22,0xf5,0xc1,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1c1f522 <unknown>

ld1q    za7v.q[w13, 0], p2/z, [x12, x11, lsl #4]
// CHECK-INST: ld1q    {za7v.q[w13, 0]}, p2/z, [x12, x11, lsl #4]
// CHECK-ENCODING: [0x87,0xa9,0xcb,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e1cba987 <unknown>
