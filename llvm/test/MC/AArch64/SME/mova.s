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
// Extract, tile to vector, horizontal, 8-bit

mova    z0.b, p0/m, za0h.b[w12, 0]
// CHECK-INST: mov     z0.b, p0/m, za0h.b[w12, 0]
// CHECK-ENCODING: [0x00,0x00,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 00 02 c0 <unknown>

mova    z21.b, p5/m, za0h.b[w14, 10]
// CHECK-INST: mov     z21.b, p5/m, za0h.b[w14, 10]
// CHECK-ENCODING: [0x55,0x55,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 55 55 02 c0 <unknown>

mova    z23.b, p3/m, za0h.b[w15, 13]
// CHECK-INST: mov     z23.b, p3/m, za0h.b[w15, 13]
// CHECK-ENCODING: [0xb7,0x6d,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: b7 6d 02 c0 <unknown>

mova    z31.b, p7/m, za0h.b[w15, 15]
// CHECK-INST: mov     z31.b, p7/m, za0h.b[w15, 15]
// CHECK-ENCODING: [0xff,0x7d,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ff 7d 02 c0 <unknown>

mova    z5.b, p3/m, za0h.b[w12, 1]
// CHECK-INST: mov     z5.b, p3/m, za0h.b[w12, 1]
// CHECK-ENCODING: [0x25,0x0c,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 0c 02 c0 <unknown>

mova    z1.b, p1/m, za0h.b[w12, 1]
// CHECK-INST: mov     z1.b, p1/m, za0h.b[w12, 1]
// CHECK-ENCODING: [0x21,0x04,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 04 02 c0 <unknown>

mova    z24.b, p5/m, za0h.b[w14, 3]
// CHECK-INST: mov     z24.b, p5/m, za0h.b[w14, 3]
// CHECK-ENCODING: [0x78,0x54,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 78 54 02 c0 <unknown>

mova    z0.b, p6/m, za0h.b[w12, 12]
// CHECK-INST: mov     z0.b, p6/m, za0h.b[w12, 12]
// CHECK-ENCODING: [0x80,0x19,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 19 02 c0 <unknown>

mova    z17.b, p2/m, za0h.b[w14, 1]
// CHECK-INST: mov     z17.b, p2/m, za0h.b[w14, 1]
// CHECK-ENCODING: [0x31,0x48,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 31 48 02 c0 <unknown>

mova    z29.b, p2/m, za0h.b[w12, 6]
// CHECK-INST: mov     z29.b, p2/m, za0h.b[w12, 6]
// CHECK-ENCODING: [0xdd,0x08,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: dd 08 02 c0 <unknown>

mova    z2.b, p5/m, za0h.b[w15, 9]
// CHECK-INST: mov     z2.b, p5/m, za0h.b[w15, 9]
// CHECK-ENCODING: [0x22,0x75,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 75 02 c0 <unknown>

mova    z7.b, p2/m, za0h.b[w13, 12]
// CHECK-INST: mov     z7.b, p2/m, za0h.b[w13, 12]
// CHECK-ENCODING: [0x87,0x29,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 29 02 c0 <unknown>

// Aliases

mov     z0.b, p0/m, za0h.b[w12, 0]
// CHECK-INST: mov     z0.b, p0/m, za0h.b[w12, 0]
// CHECK-ENCODING: [0x00,0x00,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 00 02 c0 <unknown>

mov     z21.b, p5/m, za0h.b[w14, 10]
// CHECK-INST: mov     z21.b, p5/m, za0h.b[w14, 10]
// CHECK-ENCODING: [0x55,0x55,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 55 55 02 c0 <unknown>

mov     z23.b, p3/m, za0h.b[w15, 13]
// CHECK-INST: mov     z23.b, p3/m, za0h.b[w15, 13]
// CHECK-ENCODING: [0xb7,0x6d,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: b7 6d 02 c0 <unknown>

mov     z31.b, p7/m, za0h.b[w15, 15]
// CHECK-INST: mov     z31.b, p7/m, za0h.b[w15, 15]
// CHECK-ENCODING: [0xff,0x7d,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ff 7d 02 c0 <unknown>

mov     z5.b, p3/m, za0h.b[w12, 1]
// CHECK-INST: mov     z5.b, p3/m, za0h.b[w12, 1]
// CHECK-ENCODING: [0x25,0x0c,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 0c 02 c0 <unknown>

mov     z1.b, p1/m, za0h.b[w12, 1]
// CHECK-INST: mov     z1.b, p1/m, za0h.b[w12, 1]
// CHECK-ENCODING: [0x21,0x04,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 04 02 c0 <unknown>

mov     z24.b, p5/m, za0h.b[w14, 3]
// CHECK-INST: mov     z24.b, p5/m, za0h.b[w14, 3]
// CHECK-ENCODING: [0x78,0x54,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 78 54 02 c0 <unknown>

mov     z0.b, p6/m, za0h.b[w12, 12]
// CHECK-INST: mov     z0.b, p6/m, za0h.b[w12, 12]
// CHECK-ENCODING: [0x80,0x19,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 19 02 c0 <unknown>

mov     z17.b, p2/m, za0h.b[w14, 1]
// CHECK-INST: mov     z17.b, p2/m, za0h.b[w14, 1]
// CHECK-ENCODING: [0x31,0x48,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 31 48 02 c0 <unknown>

mov     z29.b, p2/m, za0h.b[w12, 6]
// CHECK-INST: mov     z29.b, p2/m, za0h.b[w12, 6]
// CHECK-ENCODING: [0xdd,0x08,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: dd 08 02 c0 <unknown>

mov     z2.b, p5/m, za0h.b[w15, 9]
// CHECK-INST: mov     z2.b, p5/m, za0h.b[w15, 9]
// CHECK-ENCODING: [0x22,0x75,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 75 02 c0 <unknown>

mov     z7.b, p2/m, za0h.b[w13, 12]
// CHECK-INST: mov     z7.b, p2/m, za0h.b[w13, 12]
// CHECK-ENCODING: [0x87,0x29,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 29 02 c0 <unknown>

// --------------------------------------------------------------------------//
// Extract, tile to vector, vertical, 8-bit

mova    z0.b, p0/m, za0v.b[w12, 0]
// CHECK-INST: mov     z0.b, p0/m, za0v.b[w12, 0]
// CHECK-ENCODING: [0x00,0x80,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 80 02 c0 <unknown>

mova    z21.b, p5/m, za0v.b[w14, 10]
// CHECK-INST: mov     z21.b, p5/m, za0v.b[w14, 10]
// CHECK-ENCODING: [0x55,0xd5,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 55 d5 02 c0 <unknown>

mova    z23.b, p3/m, za0v.b[w15, 13]
// CHECK-INST: mov     z23.b, p3/m, za0v.b[w15, 13]
// CHECK-ENCODING: [0xb7,0xed,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: b7 ed 02 c0 <unknown>

mova    z31.b, p7/m, za0v.b[w15, 15]
// CHECK-INST: mov     z31.b, p7/m, za0v.b[w15, 15]
// CHECK-ENCODING: [0xff,0xfd,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ff fd 02 c0 <unknown>

mova    z5.b, p3/m, za0v.b[w12, 1]
// CHECK-INST: mov     z5.b, p3/m, za0v.b[w12, 1]
// CHECK-ENCODING: [0x25,0x8c,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 8c 02 c0 <unknown>

mova    z1.b, p1/m, za0v.b[w12, 1]
// CHECK-INST: mov     z1.b, p1/m, za0v.b[w12, 1]
// CHECK-ENCODING: [0x21,0x84,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 84 02 c0 <unknown>

mova    z24.b, p5/m, za0v.b[w14, 3]
// CHECK-INST: mov     z24.b, p5/m, za0v.b[w14, 3]
// CHECK-ENCODING: [0x78,0xd4,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 78 d4 02 c0 <unknown>

mova    z0.b, p6/m, za0v.b[w12, 12]
// CHECK-INST: mov     z0.b, p6/m, za0v.b[w12, 12]
// CHECK-ENCODING: [0x80,0x99,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 99 02 c0 <unknown>

mova    z17.b, p2/m, za0v.b[w14, 1]
// CHECK-INST: mov     z17.b, p2/m, za0v.b[w14, 1]
// CHECK-ENCODING: [0x31,0xc8,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 31 c8 02 c0 <unknown>

mova    z29.b, p2/m, za0v.b[w12, 6]
// CHECK-INST: mov     z29.b, p2/m, za0v.b[w12, 6]
// CHECK-ENCODING: [0xdd,0x88,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: dd 88 02 c0 <unknown>

mova    z2.b, p5/m, za0v.b[w15, 9]
// CHECK-INST: mov     z2.b, p5/m, za0v.b[w15, 9]
// CHECK-ENCODING: [0x22,0xf5,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 f5 02 c0 <unknown>

mova    z7.b, p2/m, za0v.b[w13, 12]
// CHECK-INST: mov     z7.b, p2/m, za0v.b[w13, 12]
// CHECK-ENCODING: [0x87,0xa9,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 a9 02 c0 <unknown>

// Aliases

mov     z0.b, p0/m, za0v.b[w12, 0]
// CHECK-INST: mov     z0.b, p0/m, za0v.b[w12, 0]
// CHECK-ENCODING: [0x00,0x80,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 80 02 c0 <unknown>

mov     z21.b, p5/m, za0v.b[w14, 10]
// CHECK-INST: mov     z21.b, p5/m, za0v.b[w14, 10]
// CHECK-ENCODING: [0x55,0xd5,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 55 d5 02 c0 <unknown>

mov     z23.b, p3/m, za0v.b[w15, 13]
// CHECK-INST: mov     z23.b, p3/m, za0v.b[w15, 13]
// CHECK-ENCODING: [0xb7,0xed,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: b7 ed 02 c0 <unknown>

mov     z31.b, p7/m, za0v.b[w15, 15]
// CHECK-INST: mov     z31.b, p7/m, za0v.b[w15, 15]
// CHECK-ENCODING: [0xff,0xfd,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ff fd 02 c0 <unknown>

mov     z5.b, p3/m, za0v.b[w12, 1]
// CHECK-INST: mov     z5.b, p3/m, za0v.b[w12, 1]
// CHECK-ENCODING: [0x25,0x8c,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 8c 02 c0 <unknown>

mov     z1.b, p1/m, za0v.b[w12, 1]
// CHECK-INST: mov     z1.b, p1/m, za0v.b[w12, 1]
// CHECK-ENCODING: [0x21,0x84,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 84 02 c0 <unknown>

mov     z24.b, p5/m, za0v.b[w14, 3]
// CHECK-INST: mov     z24.b, p5/m, za0v.b[w14, 3]
// CHECK-ENCODING: [0x78,0xd4,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 78 d4 02 c0 <unknown>

mov     z0.b, p6/m, za0v.b[w12, 12]
// CHECK-INST: mov     z0.b, p6/m, za0v.b[w12, 12]
// CHECK-ENCODING: [0x80,0x99,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 99 02 c0 <unknown>

mov     z17.b, p2/m, za0v.b[w14, 1]
// CHECK-INST: mov     z17.b, p2/m, za0v.b[w14, 1]
// CHECK-ENCODING: [0x31,0xc8,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 31 c8 02 c0 <unknown>

mov     z29.b, p2/m, za0v.b[w12, 6]
// CHECK-INST: mov     z29.b, p2/m, za0v.b[w12, 6]
// CHECK-ENCODING: [0xdd,0x88,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: dd 88 02 c0 <unknown>

mov     z2.b, p5/m, za0v.b[w15, 9]
// CHECK-INST: mov     z2.b, p5/m, za0v.b[w15, 9]
// CHECK-ENCODING: [0x22,0xf5,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 f5 02 c0 <unknown>

mov     z7.b, p2/m, za0v.b[w13, 12]
// CHECK-INST: mov     z7.b, p2/m, za0v.b[w13, 12]
// CHECK-ENCODING: [0x87,0xa9,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 a9 02 c0 <unknown>

// --------------------------------------------------------------------------//
// Extract, tile to vector, horizontal, 16-bit

mova    z0.h, p0/m, za0h.h[w12, 0]
// CHECK-INST: mov     z0.h, p0/m, za0h.h[w12, 0]
// CHECK-ENCODING: [0x00,0x00,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 00 42 c0 <unknown>

mova    z21.h, p5/m, za1h.h[w14, 2]
// CHECK-INST: mov     z21.h, p5/m, za1h.h[w14, 2]
// CHECK-ENCODING: [0x55,0x55,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 55 55 42 c0 <unknown>

mova    z23.h, p3/m, za1h.h[w15, 5]
// CHECK-INST: mov     z23.h, p3/m, za1h.h[w15, 5]
// CHECK-ENCODING: [0xb7,0x6d,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: b7 6d 42 c0 <unknown>

mova    z31.h, p7/m, za1h.h[w15, 7]
// CHECK-INST: mov     z31.h, p7/m, za1h.h[w15, 7]
// CHECK-ENCODING: [0xff,0x7d,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ff 7d 42 c0 <unknown>

mova    z5.h, p3/m, za0h.h[w12, 1]
// CHECK-INST: mov     z5.h, p3/m, za0h.h[w12, 1]
// CHECK-ENCODING: [0x25,0x0c,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 0c 42 c0 <unknown>

mova    z1.h, p1/m, za0h.h[w12, 1]
// CHECK-INST: mov     z1.h, p1/m, za0h.h[w12, 1]
// CHECK-ENCODING: [0x21,0x04,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 04 42 c0 <unknown>

mova    z24.h, p5/m, za0h.h[w14, 3]
// CHECK-INST: mov     z24.h, p5/m, za0h.h[w14, 3]
// CHECK-ENCODING: [0x78,0x54,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 78 54 42 c0 <unknown>

mova    z0.h, p6/m, za1h.h[w12, 4]
// CHECK-INST: mov     z0.h, p6/m, za1h.h[w12, 4]
// CHECK-ENCODING: [0x80,0x19,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 19 42 c0 <unknown>

mova    z17.h, p2/m, za0h.h[w14, 1]
// CHECK-INST: mov     z17.h, p2/m, za0h.h[w14, 1]
// CHECK-ENCODING: [0x31,0x48,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 31 48 42 c0 <unknown>

mova    z29.h, p2/m, za0h.h[w12, 6]
// CHECK-INST: mov     z29.h, p2/m, za0h.h[w12, 6]
// CHECK-ENCODING: [0xdd,0x08,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: dd 08 42 c0 <unknown>

mova    z2.h, p5/m, za1h.h[w15, 1]
// CHECK-INST: mov     z2.h, p5/m, za1h.h[w15, 1]
// CHECK-ENCODING: [0x22,0x75,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 75 42 c0 <unknown>

mova    z7.h, p2/m, za1h.h[w13, 4]
// CHECK-INST: mov     z7.h, p2/m, za1h.h[w13, 4]
// CHECK-ENCODING: [0x87,0x29,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 29 42 c0 <unknown>

// Aliases

mov     z0.h, p0/m, za0h.h[w12, 0]
// CHECK-INST: mov     z0.h, p0/m, za0h.h[w12, 0]
// CHECK-ENCODING: [0x00,0x00,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 00 42 c0 <unknown>

mov     z21.h, p5/m, za1h.h[w14, 2]
// CHECK-INST: mov     z21.h, p5/m, za1h.h[w14, 2]
// CHECK-ENCODING: [0x55,0x55,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 55 55 42 c0 <unknown>

mov     z23.h, p3/m, za1h.h[w15, 5]
// CHECK-INST: mov     z23.h, p3/m, za1h.h[w15, 5]
// CHECK-ENCODING: [0xb7,0x6d,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: b7 6d 42 c0 <unknown>

mov     z31.h, p7/m, za1h.h[w15, 7]
// CHECK-INST: mov     z31.h, p7/m, za1h.h[w15, 7]
// CHECK-ENCODING: [0xff,0x7d,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ff 7d 42 c0 <unknown>

mov     z5.h, p3/m, za0h.h[w12, 1]
// CHECK-INST: mov     z5.h, p3/m, za0h.h[w12, 1]
// CHECK-ENCODING: [0x25,0x0c,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 0c 42 c0 <unknown>

mov     z1.h, p1/m, za0h.h[w12, 1]
// CHECK-INST: mov     z1.h, p1/m, za0h.h[w12, 1]
// CHECK-ENCODING: [0x21,0x04,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 04 42 c0 <unknown>

mov     z24.h, p5/m, za0h.h[w14, 3]
// CHECK-INST: mov     z24.h, p5/m, za0h.h[w14, 3]
// CHECK-ENCODING: [0x78,0x54,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 78 54 42 c0 <unknown>

mov     z0.h, p6/m, za1h.h[w12, 4]
// CHECK-INST: mov     z0.h, p6/m, za1h.h[w12, 4]
// CHECK-ENCODING: [0x80,0x19,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 19 42 c0 <unknown>

mov     z17.h, p2/m, za0h.h[w14, 1]
// CHECK-INST: mov     z17.h, p2/m, za0h.h[w14, 1]
// CHECK-ENCODING: [0x31,0x48,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 31 48 42 c0 <unknown>

mov     z29.h, p2/m, za0h.h[w12, 6]
// CHECK-INST: mov     z29.h, p2/m, za0h.h[w12, 6]
// CHECK-ENCODING: [0xdd,0x08,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: dd 08 42 c0 <unknown>

mov     z2.h, p5/m, za1h.h[w15, 1]
// CHECK-INST: mov     z2.h, p5/m, za1h.h[w15, 1]
// CHECK-ENCODING: [0x22,0x75,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 75 42 c0 <unknown>

mov     z7.h, p2/m, za1h.h[w13, 4]
// CHECK-INST: mov     z7.h, p2/m, za1h.h[w13, 4]
// CHECK-ENCODING: [0x87,0x29,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 29 42 c0 <unknown>

// --------------------------------------------------------------------------//
// Extract, tile to vector, vertical, 16-bit

mova    z0.h, p0/m, za0v.h[w12, 0]
// CHECK-INST: mov     z0.h, p0/m, za0v.h[w12, 0]
// CHECK-ENCODING: [0x00,0x80,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 80 42 c0 <unknown>

mova    z21.h, p5/m, za1v.h[w14, 2]
// CHECK-INST: mov     z21.h, p5/m, za1v.h[w14, 2]
// CHECK-ENCODING: [0x55,0xd5,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 55 d5 42 c0 <unknown>

mova    z23.h, p3/m, za1v.h[w15, 5]
// CHECK-INST: mov     z23.h, p3/m, za1v.h[w15, 5]
// CHECK-ENCODING: [0xb7,0xed,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: b7 ed 42 c0 <unknown>

mova    z31.h, p7/m, za1v.h[w15, 7]
// CHECK-INST: mov     z31.h, p7/m, za1v.h[w15, 7]
// CHECK-ENCODING: [0xff,0xfd,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ff fd 42 c0 <unknown>

mova    z5.h, p3/m, za0v.h[w12, 1]
// CHECK-INST: mov     z5.h, p3/m, za0v.h[w12, 1]
// CHECK-ENCODING: [0x25,0x8c,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 8c 42 c0 <unknown>

mova    z1.h, p1/m, za0v.h[w12, 1]
// CHECK-INST: mov     z1.h, p1/m, za0v.h[w12, 1]
// CHECK-ENCODING: [0x21,0x84,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 84 42 c0 <unknown>

mova    z24.h, p5/m, za0v.h[w14, 3]
// CHECK-INST: mov     z24.h, p5/m, za0v.h[w14, 3]
// CHECK-ENCODING: [0x78,0xd4,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 78 d4 42 c0 <unknown>

mova    z0.h, p6/m, za1v.h[w12, 4]
// CHECK-INST: mov     z0.h, p6/m, za1v.h[w12, 4]
// CHECK-ENCODING: [0x80,0x99,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 99 42 c0 <unknown>

mova    z17.h, p2/m, za0v.h[w14, 1]
// CHECK-INST: mov     z17.h, p2/m, za0v.h[w14, 1]
// CHECK-ENCODING: [0x31,0xc8,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 31 c8 42 c0 <unknown>

mova    z29.h, p2/m, za0v.h[w12, 6]
// CHECK-INST: mov     z29.h, p2/m, za0v.h[w12, 6]
// CHECK-ENCODING: [0xdd,0x88,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: dd 88 42 c0 <unknown>

mova    z2.h, p5/m, za1v.h[w15, 1]
// CHECK-INST: mov     z2.h, p5/m, za1v.h[w15, 1]
// CHECK-ENCODING: [0x22,0xf5,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 f5 42 c0 <unknown>

mova    z7.h, p2/m, za1v.h[w13, 4]
// CHECK-INST: mov     z7.h, p2/m, za1v.h[w13, 4]
// CHECK-ENCODING: [0x87,0xa9,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 a9 42 c0 <unknown>

// Aliases

mov     z0.h, p0/m, za0v.h[w12, 0]
// CHECK-INST: mov     z0.h, p0/m, za0v.h[w12, 0]
// CHECK-ENCODING: [0x00,0x80,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 80 42 c0 <unknown>

mov     z21.h, p5/m, za1v.h[w14, 2]
// CHECK-INST: mov     z21.h, p5/m, za1v.h[w14, 2]
// CHECK-ENCODING: [0x55,0xd5,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 55 d5 42 c0 <unknown>

mov     z23.h, p3/m, za1v.h[w15, 5]
// CHECK-INST: mov     z23.h, p3/m, za1v.h[w15, 5]
// CHECK-ENCODING: [0xb7,0xed,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: b7 ed 42 c0 <unknown>

mov     z31.h, p7/m, za1v.h[w15, 7]
// CHECK-INST: mov     z31.h, p7/m, za1v.h[w15, 7]
// CHECK-ENCODING: [0xff,0xfd,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ff fd 42 c0 <unknown>

mov     z5.h, p3/m, za0v.h[w12, 1]
// CHECK-INST: mov     z5.h, p3/m, za0v.h[w12, 1]
// CHECK-ENCODING: [0x25,0x8c,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 8c 42 c0 <unknown>

mov     z1.h, p1/m, za0v.h[w12, 1]
// CHECK-INST: mov     z1.h, p1/m, za0v.h[w12, 1]
// CHECK-ENCODING: [0x21,0x84,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 84 42 c0 <unknown>

mov     z24.h, p5/m, za0v.h[w14, 3]
// CHECK-INST: mov     z24.h, p5/m, za0v.h[w14, 3]
// CHECK-ENCODING: [0x78,0xd4,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 78 d4 42 c0 <unknown>

mov     z0.h, p6/m, za1v.h[w12, 4]
// CHECK-INST: mov     z0.h, p6/m, za1v.h[w12, 4]
// CHECK-ENCODING: [0x80,0x99,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 99 42 c0 <unknown>

mov     z17.h, p2/m, za0v.h[w14, 1]
// CHECK-INST: mov     z17.h, p2/m, za0v.h[w14, 1]
// CHECK-ENCODING: [0x31,0xc8,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 31 c8 42 c0 <unknown>

mov     z29.h, p2/m, za0v.h[w12, 6]
// CHECK-INST: mov     z29.h, p2/m, za0v.h[w12, 6]
// CHECK-ENCODING: [0xdd,0x88,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: dd 88 42 c0 <unknown>

mov     z2.h, p5/m, za1v.h[w15, 1]
// CHECK-INST: mov     z2.h, p5/m, za1v.h[w15, 1]
// CHECK-ENCODING: [0x22,0xf5,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 f5 42 c0 <unknown>

mov     z7.h, p2/m, za1v.h[w13, 4]
// CHECK-INST: mov     z7.h, p2/m, za1v.h[w13, 4]
// CHECK-ENCODING: [0x87,0xa9,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 a9 42 c0 <unknown>

// --------------------------------------------------------------------------//
// Extract, tile to vector, horizontal, 32-bit

mova    z0.s, p0/m, za0h.s[w12, 0]
// CHECK-INST: mov     z0.s, p0/m, za0h.s[w12, 0]
// CHECK-ENCODING: [0x00,0x00,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 00 82 c0 <unknown>

mova    z21.s, p5/m, za2h.s[w14, 2]
// CHECK-INST: mov     z21.s, p5/m, za2h.s[w14, 2]
// CHECK-ENCODING: [0x55,0x55,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 55 55 82 c0 <unknown>

mova    z23.s, p3/m, za3h.s[w15, 1]
// CHECK-INST: mov     z23.s, p3/m, za3h.s[w15, 1]
// CHECK-ENCODING: [0xb7,0x6d,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: b7 6d 82 c0 <unknown>

mova    z31.s, p7/m, za3h.s[w15, 3]
// CHECK-INST: mov     z31.s, p7/m, za3h.s[w15, 3]
// CHECK-ENCODING: [0xff,0x7d,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ff 7d 82 c0 <unknown>

mova    z5.s, p3/m, za0h.s[w12, 1]
// CHECK-INST: mov     z5.s, p3/m, za0h.s[w12, 1]
// CHECK-ENCODING: [0x25,0x0c,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 0c 82 c0 <unknown>

mova    z1.s, p1/m, za0h.s[w12, 1]
// CHECK-INST: mov     z1.s, p1/m, za0h.s[w12, 1]
// CHECK-ENCODING: [0x21,0x04,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 04 82 c0 <unknown>

mova    z24.s, p5/m, za0h.s[w14, 3]
// CHECK-INST: mov     z24.s, p5/m, za0h.s[w14, 3]
// CHECK-ENCODING: [0x78,0x54,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 78 54 82 c0 <unknown>

mova    z0.s, p6/m, za3h.s[w12, 0]
// CHECK-INST: mov     z0.s, p6/m, za3h.s[w12, 0]
// CHECK-ENCODING: [0x80,0x19,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 19 82 c0 <unknown>

mova    z17.s, p2/m, za0h.s[w14, 1]
// CHECK-INST: mov     z17.s, p2/m, za0h.s[w14, 1]
// CHECK-ENCODING: [0x31,0x48,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 31 48 82 c0 <unknown>

mova    z29.s, p2/m, za1h.s[w12, 2]
// CHECK-INST: mov     z29.s, p2/m, za1h.s[w12, 2]
// CHECK-ENCODING: [0xdd,0x08,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: dd 08 82 c0 <unknown>

mova    z2.s, p5/m, za2h.s[w15, 1]
// CHECK-INST: mov     z2.s, p5/m, za2h.s[w15, 1]
// CHECK-ENCODING: [0x22,0x75,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 75 82 c0 <unknown>

mova    z7.s, p2/m, za3h.s[w13, 0]
// CHECK-INST: mov     z7.s, p2/m, za3h.s[w13, 0]
// CHECK-ENCODING: [0x87,0x29,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 29 82 c0 <unknown>

// Aliases

mov     z0.s, p0/m, za0h.s[w12, 0]
// CHECK-INST: mov     z0.s, p0/m, za0h.s[w12, 0]
// CHECK-ENCODING: [0x00,0x00,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 00 82 c0 <unknown>

mov     z21.s, p5/m, za2h.s[w14, 2]
// CHECK-INST: mov     z21.s, p5/m, za2h.s[w14, 2]
// CHECK-ENCODING: [0x55,0x55,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 55 55 82 c0 <unknown>

mov     z23.s, p3/m, za3h.s[w15, 1]
// CHECK-INST: mov     z23.s, p3/m, za3h.s[w15, 1]
// CHECK-ENCODING: [0xb7,0x6d,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: b7 6d 82 c0 <unknown>

mov     z31.s, p7/m, za3h.s[w15, 3]
// CHECK-INST: mov     z31.s, p7/m, za3h.s[w15, 3]
// CHECK-ENCODING: [0xff,0x7d,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ff 7d 82 c0 <unknown>

mov     z5.s, p3/m, za0h.s[w12, 1]
// CHECK-INST: mov     z5.s, p3/m, za0h.s[w12, 1]
// CHECK-ENCODING: [0x25,0x0c,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 0c 82 c0 <unknown>

mov     z1.s, p1/m, za0h.s[w12, 1]
// CHECK-INST: mov     z1.s, p1/m, za0h.s[w12, 1]
// CHECK-ENCODING: [0x21,0x04,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 04 82 c0 <unknown>

mov     z24.s, p5/m, za0h.s[w14, 3]
// CHECK-INST: mov     z24.s, p5/m, za0h.s[w14, 3]
// CHECK-ENCODING: [0x78,0x54,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 78 54 82 c0 <unknown>

mov     z0.s, p6/m, za3h.s[w12, 0]
// CHECK-INST: mov     z0.s, p6/m, za3h.s[w12, 0]
// CHECK-ENCODING: [0x80,0x19,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 19 82 c0 <unknown>

mov     z17.s, p2/m, za0h.s[w14, 1]
// CHECK-INST: mov     z17.s, p2/m, za0h.s[w14, 1]
// CHECK-ENCODING: [0x31,0x48,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 31 48 82 c0 <unknown>

mov     z29.s, p2/m, za1h.s[w12, 2]
// CHECK-INST: mov     z29.s, p2/m, za1h.s[w12, 2]
// CHECK-ENCODING: [0xdd,0x08,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: dd 08 82 c0 <unknown>

mov     z2.s, p5/m, za2h.s[w15, 1]
// CHECK-INST: mov     z2.s, p5/m, za2h.s[w15, 1]
// CHECK-ENCODING: [0x22,0x75,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 75 82 c0 <unknown>

mov     z7.s, p2/m, za3h.s[w13, 0]
// CHECK-INST: mov     z7.s, p2/m, za3h.s[w13, 0]
// CHECK-ENCODING: [0x87,0x29,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 29 82 c0 <unknown>

// --------------------------------------------------------------------------//
// Extract, tile to vector, vertical, 32-bit

mova    z0.s, p0/m, za0v.s[w12, 0]
// CHECK-INST: mov     z0.s, p0/m, za0v.s[w12, 0]
// CHECK-ENCODING: [0x00,0x80,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 80 82 c0 <unknown>

mova    z21.s, p5/m, za2v.s[w14, 2]
// CHECK-INST: mov     z21.s, p5/m, za2v.s[w14, 2]
// CHECK-ENCODING: [0x55,0xd5,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 55 d5 82 c0 <unknown>

mova    z23.s, p3/m, za3v.s[w15, 1]
// CHECK-INST: mov     z23.s, p3/m, za3v.s[w15, 1]
// CHECK-ENCODING: [0xb7,0xed,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: b7 ed 82 c0 <unknown>

mova    z31.s, p7/m, za3v.s[w15, 3]
// CHECK-INST: mov     z31.s, p7/m, za3v.s[w15, 3]
// CHECK-ENCODING: [0xff,0xfd,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ff fd 82 c0 <unknown>

mova    z5.s, p3/m, za0v.s[w12, 1]
// CHECK-INST: mov     z5.s, p3/m, za0v.s[w12, 1]
// CHECK-ENCODING: [0x25,0x8c,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 8c 82 c0 <unknown>

mova    z1.s, p1/m, za0v.s[w12, 1]
// CHECK-INST: mov     z1.s, p1/m, za0v.s[w12, 1]
// CHECK-ENCODING: [0x21,0x84,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 84 82 c0 <unknown>

mova    z24.s, p5/m, za0v.s[w14, 3]
// CHECK-INST: mov     z24.s, p5/m, za0v.s[w14, 3]
// CHECK-ENCODING: [0x78,0xd4,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 78 d4 82 c0 <unknown>

mova    z0.s, p6/m, za3v.s[w12, 0]
// CHECK-INST: mov     z0.s, p6/m, za3v.s[w12, 0]
// CHECK-ENCODING: [0x80,0x99,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 99 82 c0 <unknown>

mova    z17.s, p2/m, za0v.s[w14, 1]
// CHECK-INST: mov     z17.s, p2/m, za0v.s[w14, 1]
// CHECK-ENCODING: [0x31,0xc8,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 31 c8 82 c0 <unknown>

mova    z29.s, p2/m, za1v.s[w12, 2]
// CHECK-INST: mov     z29.s, p2/m, za1v.s[w12, 2]
// CHECK-ENCODING: [0xdd,0x88,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: dd 88 82 c0 <unknown>

mova    z2.s, p5/m, za2v.s[w15, 1]
// CHECK-INST: mov     z2.s, p5/m, za2v.s[w15, 1]
// CHECK-ENCODING: [0x22,0xf5,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 f5 82 c0 <unknown>

mova    z7.s, p2/m, za3v.s[w13, 0]
// CHECK-INST: mov     z7.s, p2/m, za3v.s[w13, 0]
// CHECK-ENCODING: [0x87,0xa9,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 a9 82 c0 <unknown>

// Aliases

mov     z0.s, p0/m, za0v.s[w12, 0]
// CHECK-INST: mov     z0.s, p0/m, za0v.s[w12, 0]
// CHECK-ENCODING: [0x00,0x80,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 80 82 c0 <unknown>

mov     z21.s, p5/m, za2v.s[w14, 2]
// CHECK-INST: mov     z21.s, p5/m, za2v.s[w14, 2]
// CHECK-ENCODING: [0x55,0xd5,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 55 d5 82 c0 <unknown>

mov     z23.s, p3/m, za3v.s[w15, 1]
// CHECK-INST: mov     z23.s, p3/m, za3v.s[w15, 1]
// CHECK-ENCODING: [0xb7,0xed,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: b7 ed 82 c0 <unknown>

mov     z31.s, p7/m, za3v.s[w15, 3]
// CHECK-INST: mov     z31.s, p7/m, za3v.s[w15, 3]
// CHECK-ENCODING: [0xff,0xfd,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ff fd 82 c0 <unknown>

mov     z5.s, p3/m, za0v.s[w12, 1]
// CHECK-INST: mov     z5.s, p3/m, za0v.s[w12, 1]
// CHECK-ENCODING: [0x25,0x8c,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 8c 82 c0 <unknown>

mov     z1.s, p1/m, za0v.s[w12, 1]
// CHECK-INST: mov     z1.s, p1/m, za0v.s[w12, 1]
// CHECK-ENCODING: [0x21,0x84,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 84 82 c0 <unknown>

mov     z24.s, p5/m, za0v.s[w14, 3]
// CHECK-INST: mov     z24.s, p5/m, za0v.s[w14, 3]
// CHECK-ENCODING: [0x78,0xd4,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 78 d4 82 c0 <unknown>

mov     z0.s, p6/m, za3v.s[w12, 0]
// CHECK-INST: mov     z0.s, p6/m, za3v.s[w12, 0]
// CHECK-ENCODING: [0x80,0x99,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 99 82 c0 <unknown>

mov     z17.s, p2/m, za0v.s[w14, 1]
// CHECK-INST: mov     z17.s, p2/m, za0v.s[w14, 1]
// CHECK-ENCODING: [0x31,0xc8,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 31 c8 82 c0 <unknown>

mov     z29.s, p2/m, za1v.s[w12, 2]
// CHECK-INST: mov     z29.s, p2/m, za1v.s[w12, 2]
// CHECK-ENCODING: [0xdd,0x88,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: dd 88 82 c0 <unknown>

mov     z2.s, p5/m, za2v.s[w15, 1]
// CHECK-INST: mov     z2.s, p5/m, za2v.s[w15, 1]
// CHECK-ENCODING: [0x22,0xf5,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 f5 82 c0 <unknown>

mov     z7.s, p2/m, za3v.s[w13, 0]
// CHECK-INST: mov     z7.s, p2/m, za3v.s[w13, 0]
// CHECK-ENCODING: [0x87,0xa9,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 a9 82 c0 <unknown>

// --------------------------------------------------------------------------//
// Extract, tile to vector, horizontal, 64-bit

mova    z0.d, p0/m, za0h.d[w12, 0]
// CHECK-INST: mov     z0.d, p0/m, za0h.d[w12, 0]
// CHECK-ENCODING: [0x00,0x00,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 00 c2 c0 <unknown>

mova    z21.d, p5/m, za5h.d[w14, 0]
// CHECK-INST: mov     z21.d, p5/m, za5h.d[w14, 0]
// CHECK-ENCODING: [0x55,0x55,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 55 55 c2 c0 <unknown>

mova    z23.d, p3/m, za6h.d[w15, 1]
// CHECK-INST: mov     z23.d, p3/m, za6h.d[w15, 1]
// CHECK-ENCODING: [0xb7,0x6d,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: b7 6d c2 c0 <unknown>

mova    z31.d, p7/m, za7h.d[w15, 1]
// CHECK-INST: mov     z31.d, p7/m, za7h.d[w15, 1]
// CHECK-ENCODING: [0xff,0x7d,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ff 7d c2 c0 <unknown>

mova    z5.d, p3/m, za0h.d[w12, 1]
// CHECK-INST: mov     z5.d, p3/m, za0h.d[w12, 1]
// CHECK-ENCODING: [0x25,0x0c,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 0c c2 c0 <unknown>

mova    z1.d, p1/m, za0h.d[w12, 1]
// CHECK-INST: mov     z1.d, p1/m, za0h.d[w12, 1]
// CHECK-ENCODING: [0x21,0x04,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 04 c2 c0 <unknown>

mova    z24.d, p5/m, za1h.d[w14, 1]
// CHECK-INST: mov     z24.d, p5/m, za1h.d[w14, 1]
// CHECK-ENCODING: [0x78,0x54,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 78 54 c2 c0 <unknown>

mova    z0.d, p6/m, za6h.d[w12, 0]
// CHECK-INST: mov     z0.d, p6/m, za6h.d[w12, 0]
// CHECK-ENCODING: [0x80,0x19,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 19 c2 c0 <unknown>

mova    z17.d, p2/m, za0h.d[w14, 1]
// CHECK-INST: mov     z17.d, p2/m, za0h.d[w14, 1]
// CHECK-ENCODING: [0x31,0x48,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 31 48 c2 c0 <unknown>

mova    z29.d, p2/m, za3h.d[w12, 0]
// CHECK-INST: mov     z29.d, p2/m, za3h.d[w12, 0]
// CHECK-ENCODING: [0xdd,0x08,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: dd 08 c2 c0 <unknown>

mova    z2.d, p5/m, za4h.d[w15, 1]
// CHECK-INST: mov     z2.d, p5/m, za4h.d[w15, 1]
// CHECK-ENCODING: [0x22,0x75,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 75 c2 c0 <unknown>

mova    z7.d, p2/m, za6h.d[w13, 0]
// CHECK-INST: mov     z7.d, p2/m, za6h.d[w13, 0]
// CHECK-ENCODING: [0x87,0x29,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 29 c2 c0 <unknown>

// Aliases

mov     z0.d, p0/m, za0h.d[w12, 0]
// CHECK-INST: mov     z0.d, p0/m, za0h.d[w12, 0]
// CHECK-ENCODING: [0x00,0x00,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 00 c2 c0 <unknown>

mov     z21.d, p5/m, za5h.d[w14, 0]
// CHECK-INST: mov     z21.d, p5/m, za5h.d[w14, 0]
// CHECK-ENCODING: [0x55,0x55,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 55 55 c2 c0 <unknown>

mov     z23.d, p3/m, za6h.d[w15, 1]
// CHECK-INST: mov     z23.d, p3/m, za6h.d[w15, 1]
// CHECK-ENCODING: [0xb7,0x6d,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: b7 6d c2 c0 <unknown>

mov     z31.d, p7/m, za7h.d[w15, 1]
// CHECK-INST: mov     z31.d, p7/m, za7h.d[w15, 1]
// CHECK-ENCODING: [0xff,0x7d,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ff 7d c2 c0 <unknown>

mov     z5.d, p3/m, za0h.d[w12, 1]
// CHECK-INST: mov     z5.d, p3/m, za0h.d[w12, 1]
// CHECK-ENCODING: [0x25,0x0c,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 0c c2 c0 <unknown>

mov     z1.d, p1/m, za0h.d[w12, 1]
// CHECK-INST: mov     z1.d, p1/m, za0h.d[w12, 1]
// CHECK-ENCODING: [0x21,0x04,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 04 c2 c0 <unknown>

mov     z24.d, p5/m, za1h.d[w14, 1]
// CHECK-INST: mov     z24.d, p5/m, za1h.d[w14, 1]
// CHECK-ENCODING: [0x78,0x54,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 78 54 c2 c0 <unknown>

mov     z0.d, p6/m, za6h.d[w12, 0]
// CHECK-INST: mov     z0.d, p6/m, za6h.d[w12, 0]
// CHECK-ENCODING: [0x80,0x19,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 19 c2 c0 <unknown>

mov     z17.d, p2/m, za0h.d[w14, 1]
// CHECK-INST: mov     z17.d, p2/m, za0h.d[w14, 1]
// CHECK-ENCODING: [0x31,0x48,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 31 48 c2 c0 <unknown>

mov     z29.d, p2/m, za3h.d[w12, 0]
// CHECK-INST: mov     z29.d, p2/m, za3h.d[w12, 0]
// CHECK-ENCODING: [0xdd,0x08,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: dd 08 c2 c0 <unknown>

mov     z2.d, p5/m, za4h.d[w15, 1]
// CHECK-INST: mov     z2.d, p5/m, za4h.d[w15, 1]
// CHECK-ENCODING: [0x22,0x75,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 75 c2 c0 <unknown>

mov     z7.d, p2/m, za6h.d[w13, 0]
// CHECK-INST: mov     z7.d, p2/m, za6h.d[w13, 0]
// CHECK-ENCODING: [0x87,0x29,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 29 c2 c0 <unknown>

// --------------------------------------------------------------------------//
// Extract, tile to vector, vertical, 64-bit

mova    z0.d, p0/m, za0v.d[w12, 0]
// CHECK-INST: mov     z0.d, p0/m, za0v.d[w12, 0]
// CHECK-ENCODING: [0x00,0x80,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 80 c2 c0 <unknown>

mova    z21.d, p5/m, za5v.d[w14, 0]
// CHECK-INST: mov     z21.d, p5/m, za5v.d[w14, 0]
// CHECK-ENCODING: [0x55,0xd5,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 55 d5 c2 c0 <unknown>

mova    z23.d, p3/m, za6v.d[w15, 1]
// CHECK-INST: mov     z23.d, p3/m, za6v.d[w15, 1]
// CHECK-ENCODING: [0xb7,0xed,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: b7 ed c2 c0 <unknown>

mova    z31.d, p7/m, za7v.d[w15, 1]
// CHECK-INST: mov     z31.d, p7/m, za7v.d[w15, 1]
// CHECK-ENCODING: [0xff,0xfd,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ff fd c2 c0 <unknown>

mova    z5.d, p3/m, za0v.d[w12, 1]
// CHECK-INST: mov     z5.d, p3/m, za0v.d[w12, 1]
// CHECK-ENCODING: [0x25,0x8c,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 8c c2 c0 <unknown>

mova    z1.d, p1/m, za0v.d[w12, 1]
// CHECK-INST: mov     z1.d, p1/m, za0v.d[w12, 1]
// CHECK-ENCODING: [0x21,0x84,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 84 c2 c0 <unknown>

mova    z24.d, p5/m, za1v.d[w14, 1]
// CHECK-INST: mov     z24.d, p5/m, za1v.d[w14, 1]
// CHECK-ENCODING: [0x78,0xd4,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 78 d4 c2 c0 <unknown>

mova    z0.d, p6/m, za6v.d[w12, 0]
// CHECK-INST: mov     z0.d, p6/m, za6v.d[w12, 0]
// CHECK-ENCODING: [0x80,0x99,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 99 c2 c0 <unknown>

mova    z17.d, p2/m, za0v.d[w14, 1]
// CHECK-INST: mov     z17.d, p2/m, za0v.d[w14, 1]
// CHECK-ENCODING: [0x31,0xc8,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 31 c8 c2 c0 <unknown>

mova    z29.d, p2/m, za3v.d[w12, 0]
// CHECK-INST: mov     z29.d, p2/m, za3v.d[w12, 0]
// CHECK-ENCODING: [0xdd,0x88,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: dd 88 c2 c0 <unknown>

mova    z2.d, p5/m, za4v.d[w15, 1]
// CHECK-INST: mov     z2.d, p5/m, za4v.d[w15, 1]
// CHECK-ENCODING: [0x22,0xf5,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 f5 c2 c0 <unknown>

mova    z7.d, p2/m, za6v.d[w13, 0]
// CHECK-INST: mov     z7.d, p2/m, za6v.d[w13, 0]
// CHECK-ENCODING: [0x87,0xa9,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 a9 c2 c0 <unknown>

// Aliases

mov     z0.d, p0/m, za0v.d[w12, 0]
// CHECK-INST: mov     z0.d, p0/m, za0v.d[w12, 0]
// CHECK-ENCODING: [0x00,0x80,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 80 c2 c0 <unknown>

mov     z21.d, p5/m, za5v.d[w14, 0]
// CHECK-INST: mov     z21.d, p5/m, za5v.d[w14, 0]
// CHECK-ENCODING: [0x55,0xd5,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 55 d5 c2 c0 <unknown>

mov     z23.d, p3/m, za6v.d[w15, 1]
// CHECK-INST: mov     z23.d, p3/m, za6v.d[w15, 1]
// CHECK-ENCODING: [0xb7,0xed,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: b7 ed c2 c0 <unknown>

mov     z31.d, p7/m, za7v.d[w15, 1]
// CHECK-INST: mov     z31.d, p7/m, za7v.d[w15, 1]
// CHECK-ENCODING: [0xff,0xfd,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ff fd c2 c0 <unknown>

mov     z5.d, p3/m, za0v.d[w12, 1]
// CHECK-INST: mov     z5.d, p3/m, za0v.d[w12, 1]
// CHECK-ENCODING: [0x25,0x8c,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 8c c2 c0 <unknown>

mov     z1.d, p1/m, za0v.d[w12, 1]
// CHECK-INST: mov     z1.d, p1/m, za0v.d[w12, 1]
// CHECK-ENCODING: [0x21,0x84,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 84 c2 c0 <unknown>

mov     z24.d, p5/m, za1v.d[w14, 1]
// CHECK-INST: mov     z24.d, p5/m, za1v.d[w14, 1]
// CHECK-ENCODING: [0x78,0xd4,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 78 d4 c2 c0 <unknown>

mov     z0.d, p6/m, za6v.d[w12, 0]
// CHECK-INST: mov     z0.d, p6/m, za6v.d[w12, 0]
// CHECK-ENCODING: [0x80,0x99,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 99 c2 c0 <unknown>

mov     z17.d, p2/m, za0v.d[w14, 1]
// CHECK-INST: mov     z17.d, p2/m, za0v.d[w14, 1]
// CHECK-ENCODING: [0x31,0xc8,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 31 c8 c2 c0 <unknown>

mov     z29.d, p2/m, za3v.d[w12, 0]
// CHECK-INST: mov     z29.d, p2/m, za3v.d[w12, 0]
// CHECK-ENCODING: [0xdd,0x88,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: dd 88 c2 c0 <unknown>

mov     z2.d, p5/m, za4v.d[w15, 1]
// CHECK-INST: mov     z2.d, p5/m, za4v.d[w15, 1]
// CHECK-ENCODING: [0x22,0xf5,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 f5 c2 c0 <unknown>

mov     z7.d, p2/m, za6v.d[w13, 0]
// CHECK-INST: mov     z7.d, p2/m, za6v.d[w13, 0]
// CHECK-ENCODING: [0x87,0xa9,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 a9 c2 c0 <unknown>

// --------------------------------------------------------------------------//
// Extract, tile to vector, horizontal, 128-bit

mova    z0.q, p0/m, za0h.q[w12, 0]
// CHECK-INST: mov     z0.q, p0/m, za0h.q[w12, 0]
// CHECK-ENCODING: [0x00,0x00,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 00 c3 c0 <unknown>

mova    z21.q, p5/m, za10h.q[w14, 0]
// CHECK-INST: mov     z21.q, p5/m, za10h.q[w14, 0]
// CHECK-ENCODING: [0x55,0x55,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 55 55 c3 c0 <unknown>

mova    z23.q, p3/m, za13h.q[w15, 0]
// CHECK-INST: mov     z23.q, p3/m, za13h.q[w15, 0]
// CHECK-ENCODING: [0xb7,0x6d,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: b7 6d c3 c0 <unknown>

mova    z31.q, p7/m, za15h.q[w15, 0]
// CHECK-INST: mov     z31.q, p7/m, za15h.q[w15, 0]
// CHECK-ENCODING: [0xff,0x7d,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ff 7d c3 c0 <unknown>

mova    z5.q, p3/m, za1h.q[w12, 0]
// CHECK-INST: mov     z5.q, p3/m, za1h.q[w12, 0]
// CHECK-ENCODING: [0x25,0x0c,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 0c c3 c0 <unknown>

mova    z1.q, p1/m, za1h.q[w12, 0]
// CHECK-INST: mov     z1.q, p1/m, za1h.q[w12, 0]
// CHECK-ENCODING: [0x21,0x04,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 04 c3 c0 <unknown>

mova    z24.q, p5/m, za3h.q[w14, 0]
// CHECK-INST: mov     z24.q, p5/m, za3h.q[w14, 0]
// CHECK-ENCODING: [0x78,0x54,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 78 54 c3 c0 <unknown>

mova    z0.q, p6/m, za12h.q[w12, 0]
// CHECK-INST: mov     z0.q, p6/m, za12h.q[w12, 0]
// CHECK-ENCODING: [0x80,0x19,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 19 c3 c0 <unknown>

mova    z17.q, p2/m, za1h.q[w14, 0]
// CHECK-INST: mov     z17.q, p2/m, za1h.q[w14, 0]
// CHECK-ENCODING: [0x31,0x48,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 31 48 c3 c0 <unknown>

mova    z29.q, p2/m, za6h.q[w12, 0]
// CHECK-INST: mov     z29.q, p2/m, za6h.q[w12, 0]
// CHECK-ENCODING: [0xdd,0x08,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: dd 08 c3 c0 <unknown>

mova    z2.q, p5/m, za9h.q[w15, 0]
// CHECK-INST: mov     z2.q, p5/m, za9h.q[w15, 0]
// CHECK-ENCODING: [0x22,0x75,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 75 c3 c0 <unknown>

mova    z7.q, p2/m, za12h.q[w13, 0]
// CHECK-INST: mov     z7.q, p2/m, za12h.q[w13, 0]
// CHECK-ENCODING: [0x87,0x29,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 29 c3 c0 <unknown>

// Aliases

mov     z0.q, p0/m, za0h.q[w12, 0]
// CHECK-INST: mov     z0.q, p0/m, za0h.q[w12, 0]
// CHECK-ENCODING: [0x00,0x00,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 00 c3 c0 <unknown>

mov     z21.q, p5/m, za10h.q[w14, 0]
// CHECK-INST: mov     z21.q, p5/m, za10h.q[w14, 0]
// CHECK-ENCODING: [0x55,0x55,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 55 55 c3 c0 <unknown>

mov     z23.q, p3/m, za13h.q[w15, 0]
// CHECK-INST: mov     z23.q, p3/m, za13h.q[w15, 0]
// CHECK-ENCODING: [0xb7,0x6d,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: b7 6d c3 c0 <unknown>

mov     z31.q, p7/m, za15h.q[w15, 0]
// CHECK-INST: mov     z31.q, p7/m, za15h.q[w15, 0]
// CHECK-ENCODING: [0xff,0x7d,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ff 7d c3 c0 <unknown>

mov     z5.q, p3/m, za1h.q[w12, 0]
// CHECK-INST: mov     z5.q, p3/m, za1h.q[w12, 0]
// CHECK-ENCODING: [0x25,0x0c,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 0c c3 c0 <unknown>

mov     z1.q, p1/m, za1h.q[w12, 0]
// CHECK-INST: mov     z1.q, p1/m, za1h.q[w12, 0]
// CHECK-ENCODING: [0x21,0x04,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 04 c3 c0 <unknown>

mov     z24.q, p5/m, za3h.q[w14, 0]
// CHECK-INST: mov     z24.q, p5/m, za3h.q[w14, 0]
// CHECK-ENCODING: [0x78,0x54,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 78 54 c3 c0 <unknown>

mov     z0.q, p6/m, za12h.q[w12, 0]
// CHECK-INST: mov     z0.q, p6/m, za12h.q[w12, 0]
// CHECK-ENCODING: [0x80,0x19,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 19 c3 c0 <unknown>

mov     z17.q, p2/m, za1h.q[w14, 0]
// CHECK-INST: mov     z17.q, p2/m, za1h.q[w14, 0]
// CHECK-ENCODING: [0x31,0x48,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 31 48 c3 c0 <unknown>

mov     z29.q, p2/m, za6h.q[w12, 0]
// CHECK-INST: mov     z29.q, p2/m, za6h.q[w12, 0]
// CHECK-ENCODING: [0xdd,0x08,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: dd 08 c3 c0 <unknown>

mov     z2.q, p5/m, za9h.q[w15, 0]
// CHECK-INST: mov     z2.q, p5/m, za9h.q[w15, 0]
// CHECK-ENCODING: [0x22,0x75,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 75 c3 c0 <unknown>

mov     z7.q, p2/m, za12h.q[w13, 0]
// CHECK-INST: mov     z7.q, p2/m, za12h.q[w13, 0]
// CHECK-ENCODING: [0x87,0x29,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 29 c3 c0 <unknown>

// --------------------------------------------------------------------------//
// Extract, tile to vector, vertical, 128-bit

mova    z0.q, p0/m, za0v.q[w12, 0]
// CHECK-INST: mov     z0.q, p0/m, za0v.q[w12, 0]
// CHECK-ENCODING: [0x00,0x80,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 80 c3 c0 <unknown>

mova    z21.q, p5/m, za10v.q[w14, 0]
// CHECK-INST: mov     z21.q, p5/m, za10v.q[w14, 0]
// CHECK-ENCODING: [0x55,0xd5,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 55 d5 c3 c0 <unknown>

mova    z23.q, p3/m, za13v.q[w15, 0]
// CHECK-INST: mov     z23.q, p3/m, za13v.q[w15, 0]
// CHECK-ENCODING: [0xb7,0xed,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: b7 ed c3 c0 <unknown>

mova    z31.q, p7/m, za15v.q[w15, 0]
// CHECK-INST: mov     z31.q, p7/m, za15v.q[w15, 0]
// CHECK-ENCODING: [0xff,0xfd,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ff fd c3 c0 <unknown>

mova    z5.q, p3/m, za1v.q[w12, 0]
// CHECK-INST: mov     z5.q, p3/m, za1v.q[w12, 0]
// CHECK-ENCODING: [0x25,0x8c,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 8c c3 c0 <unknown>

mova    z1.q, p1/m, za1v.q[w12, 0]
// CHECK-INST: mov     z1.q, p1/m, za1v.q[w12, 0]
// CHECK-ENCODING: [0x21,0x84,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 84 c3 c0 <unknown>

mova    z24.q, p5/m, za3v.q[w14, 0]
// CHECK-INST: mov     z24.q, p5/m, za3v.q[w14, 0]
// CHECK-ENCODING: [0x78,0xd4,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 78 d4 c3 c0 <unknown>

mova    z0.q, p6/m, za12v.q[w12, 0]
// CHECK-INST: mov     z0.q, p6/m, za12v.q[w12, 0]
// CHECK-ENCODING: [0x80,0x99,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 99 c3 c0 <unknown>

mova    z17.q, p2/m, za1v.q[w14, 0]
// CHECK-INST: mov     z17.q, p2/m, za1v.q[w14, 0]
// CHECK-ENCODING: [0x31,0xc8,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 31 c8 c3 c0 <unknown>

mova    z29.q, p2/m, za6v.q[w12, 0]
// CHECK-INST: mov     z29.q, p2/m, za6v.q[w12, 0]
// CHECK-ENCODING: [0xdd,0x88,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: dd 88 c3 c0 <unknown>

mova    z2.q, p5/m, za9v.q[w15, 0]
// CHECK-INST: mov     z2.q, p5/m, za9v.q[w15, 0]
// CHECK-ENCODING: [0x22,0xf5,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 f5 c3 c0 <unknown>

mova    z7.q, p2/m, za12v.q[w13, 0]
// CHECK-INST: mov     z7.q, p2/m, za12v.q[w13, 0]
// CHECK-ENCODING: [0x87,0xa9,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 a9 c3 c0 <unknown>

// Aliases

mov     z0.q, p0/m, za0v.q[w12, 0]
// CHECK-INST: mov     z0.q, p0/m, za0v.q[w12, 0]
// CHECK-ENCODING: [0x00,0x80,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 80 c3 c0 <unknown>

mov     z21.q, p5/m, za10v.q[w14, 0]
// CHECK-INST: mov     z21.q, p5/m, za10v.q[w14, 0]
// CHECK-ENCODING: [0x55,0xd5,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 55 d5 c3 c0 <unknown>

mov     z23.q, p3/m, za13v.q[w15, 0]
// CHECK-INST: mov     z23.q, p3/m, za13v.q[w15, 0]
// CHECK-ENCODING: [0xb7,0xed,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: b7 ed c3 c0 <unknown>

mov     z31.q, p7/m, za15v.q[w15, 0]
// CHECK-INST: mov     z31.q, p7/m, za15v.q[w15, 0]
// CHECK-ENCODING: [0xff,0xfd,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ff fd c3 c0 <unknown>

mov     z5.q, p3/m, za1v.q[w12, 0]
// CHECK-INST: mov     z5.q, p3/m, za1v.q[w12, 0]
// CHECK-ENCODING: [0x25,0x8c,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 8c c3 c0 <unknown>

mov     z1.q, p1/m, za1v.q[w12, 0]
// CHECK-INST: mov     z1.q, p1/m, za1v.q[w12, 0]
// CHECK-ENCODING: [0x21,0x84,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 84 c3 c0 <unknown>

mov     z24.q, p5/m, za3v.q[w14, 0]
// CHECK-INST: mov     z24.q, p5/m, za3v.q[w14, 0]
// CHECK-ENCODING: [0x78,0xd4,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 78 d4 c3 c0 <unknown>

mov     z0.q, p6/m, za12v.q[w12, 0]
// CHECK-INST: mov     z0.q, p6/m, za12v.q[w12, 0]
// CHECK-ENCODING: [0x80,0x99,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 99 c3 c0 <unknown>

mov     z17.q, p2/m, za1v.q[w14, 0]
// CHECK-INST: mov     z17.q, p2/m, za1v.q[w14, 0]
// CHECK-ENCODING: [0x31,0xc8,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 31 c8 c3 c0 <unknown>

mov     z29.q, p2/m, za6v.q[w12, 0]
// CHECK-INST: mov     z29.q, p2/m, za6v.q[w12, 0]
// CHECK-ENCODING: [0xdd,0x88,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: dd 88 c3 c0 <unknown>

mov     z2.q, p5/m, za9v.q[w15, 0]
// CHECK-INST: mov     z2.q, p5/m, za9v.q[w15, 0]
// CHECK-ENCODING: [0x22,0xf5,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 f5 c3 c0 <unknown>

mov     z7.q, p2/m, za12v.q[w13, 0]
// CHECK-INST: mov     z7.q, p2/m, za12v.q[w13, 0]
// CHECK-ENCODING: [0x87,0xa9,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 a9 c3 c0 <unknown>

// --------------------------------------------------------------------------//
// Insert, vector to tile, horizontal, 8-bit

mova    za0h.b[w12, 0], p0/m, z0.b
// CHECK-INST: mov     za0h.b[w12, 0], p0/m, z0.b
// CHECK-ENCODING: [0x00,0x00,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 00 00 c0 <unknown>

mova    za0h.b[w14, 5], p5/m, z10.b
// CHECK-INST: mov     za0h.b[w14, 5], p5/m, z10.b
// CHECK-ENCODING: [0x45,0x55,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 55 00 c0 <unknown>

mova    za0h.b[w15, 7], p3/m, z13.b
// CHECK-INST: mov     za0h.b[w15, 7], p3/m, z13.b
// CHECK-ENCODING: [0xa7,0x6d,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 6d 00 c0 <unknown>

mova    za0h.b[w15, 15], p7/m, z31.b
// CHECK-INST: mov     za0h.b[w15, 15], p7/m, z31.b
// CHECK-ENCODING: [0xef,0x7f,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef 7f 00 c0 <unknown>

mova    za0h.b[w12, 5], p3/m, z17.b
// CHECK-INST: mov     za0h.b[w12, 5], p3/m, z17.b
// CHECK-ENCODING: [0x25,0x0e,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 0e 00 c0 <unknown>

mova    za0h.b[w12, 1], p1/m, z1.b
// CHECK-INST: mov     za0h.b[w12, 1], p1/m, z1.b
// CHECK-ENCODING: [0x21,0x04,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 04 00 c0 <unknown>

mova    za0h.b[w14, 8], p5/m, z19.b
// CHECK-INST: mov     za0h.b[w14, 8], p5/m, z19.b
// CHECK-ENCODING: [0x68,0x56,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 56 00 c0 <unknown>

mova    za0h.b[w12, 0], p6/m, z12.b
// CHECK-INST: mov     za0h.b[w12, 0], p6/m, z12.b
// CHECK-ENCODING: [0x80,0x19,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 19 00 c0 <unknown>

mova    za0h.b[w14, 1], p2/m, z1.b
// CHECK-INST: mov     za0h.b[w14, 1], p2/m, z1.b
// CHECK-ENCODING: [0x21,0x48,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 48 00 c0 <unknown>

mova    za0h.b[w12, 13], p2/m, z22.b
// CHECK-INST: mov     za0h.b[w12, 13], p2/m, z22.b
// CHECK-ENCODING: [0xcd,0x0a,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 0a 00 c0 <unknown>

mova    za0h.b[w15, 2], p5/m, z9.b
// CHECK-INST: mov     za0h.b[w15, 2], p5/m, z9.b
// CHECK-ENCODING: [0x22,0x75,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 75 00 c0 <unknown>

mova    za0h.b[w13, 7], p2/m, z12.b
// CHECK-INST: mov     za0h.b[w13, 7], p2/m, z12.b
// CHECK-ENCODING: [0x87,0x29,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 29 00 c0 <unknown>

// Aliases

mov     za0h.b[w12, 0], p0/m, z0.b
// CHECK-INST: mov     za0h.b[w12, 0], p0/m, z0.b
// CHECK-ENCODING: [0x00,0x00,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 00 00 c0 <unknown>

mov     za0h.b[w14, 5], p5/m, z10.b
// CHECK-INST: mov     za0h.b[w14, 5], p5/m, z10.b
// CHECK-ENCODING: [0x45,0x55,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 55 00 c0 <unknown>

mov     za0h.b[w15, 7], p3/m, z13.b
// CHECK-INST: mov     za0h.b[w15, 7], p3/m, z13.b
// CHECK-ENCODING: [0xa7,0x6d,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 6d 00 c0 <unknown>

mov     za0h.b[w15, 15], p7/m, z31.b
// CHECK-INST: mov     za0h.b[w15, 15], p7/m, z31.b
// CHECK-ENCODING: [0xef,0x7f,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef 7f 00 c0 <unknown>

mov     za0h.b[w12, 5], p3/m, z17.b
// CHECK-INST: mov     za0h.b[w12, 5], p3/m, z17.b
// CHECK-ENCODING: [0x25,0x0e,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 0e 00 c0 <unknown>

mov     za0h.b[w12, 1], p1/m, z1.b
// CHECK-INST: mov     za0h.b[w12, 1], p1/m, z1.b
// CHECK-ENCODING: [0x21,0x04,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 04 00 c0 <unknown>

mov     za0h.b[w14, 8], p5/m, z19.b
// CHECK-INST: mov     za0h.b[w14, 8], p5/m, z19.b
// CHECK-ENCODING: [0x68,0x56,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 56 00 c0 <unknown>

mov     za0h.b[w12, 0], p6/m, z12.b
// CHECK-INST: mov     za0h.b[w12, 0], p6/m, z12.b
// CHECK-ENCODING: [0x80,0x19,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 19 00 c0 <unknown>

mov     za0h.b[w14, 1], p2/m, z1.b
// CHECK-INST: mov     za0h.b[w14, 1], p2/m, z1.b
// CHECK-ENCODING: [0x21,0x48,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 48 00 c0 <unknown>

mov     za0h.b[w12, 13], p2/m, z22.b
// CHECK-INST: mov     za0h.b[w12, 13], p2/m, z22.b
// CHECK-ENCODING: [0xcd,0x0a,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 0a 00 c0 <unknown>

mov     za0h.b[w15, 2], p5/m, z9.b
// CHECK-INST: mov     za0h.b[w15, 2], p5/m, z9.b
// CHECK-ENCODING: [0x22,0x75,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 75 00 c0 <unknown>

mov     za0h.b[w13, 7], p2/m, z12.b
// CHECK-INST: mov     za0h.b[w13, 7], p2/m, z12.b
// CHECK-ENCODING: [0x87,0x29,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 29 00 c0 <unknown>

// --------------------------------------------------------------------------//
// Insert, vector to tile, vertical, 8-bit

mova    za0v.b[w12, 0], p0/m, z0.b
// CHECK-INST: mov     za0v.b[w12, 0], p0/m, z0.b
// CHECK-ENCODING: [0x00,0x80,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 80 00 c0 <unknown>

mova    za0v.b[w14, 5], p5/m, z10.b
// CHECK-INST: mov     za0v.b[w14, 5], p5/m, z10.b
// CHECK-ENCODING: [0x45,0xd5,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 d5 00 c0 <unknown>

mova    za0v.b[w15, 7], p3/m, z13.b
// CHECK-INST: mov     za0v.b[w15, 7], p3/m, z13.b
// CHECK-ENCODING: [0xa7,0xed,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 ed 00 c0 <unknown>

mova    za0v.b[w15, 15], p7/m, z31.b
// CHECK-INST: mov     za0v.b[w15, 15], p7/m, z31.b
// CHECK-ENCODING: [0xef,0xff,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef ff 00 c0 <unknown>

mova    za0v.b[w12, 5], p3/m, z17.b
// CHECK-INST: mov     za0v.b[w12, 5], p3/m, z17.b
// CHECK-ENCODING: [0x25,0x8e,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 8e 00 c0 <unknown>

mova    za0v.b[w12, 1], p1/m, z1.b
// CHECK-INST: mov     za0v.b[w12, 1], p1/m, z1.b
// CHECK-ENCODING: [0x21,0x84,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 84 00 c0 <unknown>

mova    za0v.b[w14, 8], p5/m, z19.b
// CHECK-INST: mov     za0v.b[w14, 8], p5/m, z19.b
// CHECK-ENCODING: [0x68,0xd6,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 d6 00 c0 <unknown>

mova    za0v.b[w12, 0], p6/m, z12.b
// CHECK-INST: mov     za0v.b[w12, 0], p6/m, z12.b
// CHECK-ENCODING: [0x80,0x99,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 99 00 c0 <unknown>

mova    za0v.b[w14, 1], p2/m, z1.b
// CHECK-INST: mov     za0v.b[w14, 1], p2/m, z1.b
// CHECK-ENCODING: [0x21,0xc8,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 c8 00 c0 <unknown>

mova    za0v.b[w12, 13], p2/m, z22.b
// CHECK-INST: mov     za0v.b[w12, 13], p2/m, z22.b
// CHECK-ENCODING: [0xcd,0x8a,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 8a 00 c0 <unknown>

mova    za0v.b[w15, 2], p5/m, z9.b
// CHECK-INST: mov     za0v.b[w15, 2], p5/m, z9.b
// CHECK-ENCODING: [0x22,0xf5,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 f5 00 c0 <unknown>

mova    za0v.b[w13, 7], p2/m, z12.b
// CHECK-INST: mov     za0v.b[w13, 7], p2/m, z12.b
// CHECK-ENCODING: [0x87,0xa9,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 a9 00 c0 <unknown>

// Aliases

mov     za0v.b[w12, 0], p0/m, z0.b
// CHECK-INST: mov     za0v.b[w12, 0], p0/m, z0.b
// CHECK-ENCODING: [0x00,0x80,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 80 00 c0 <unknown>

mov     za0v.b[w14, 5], p5/m, z10.b
// CHECK-INST: mov     za0v.b[w14, 5], p5/m, z10.b
// CHECK-ENCODING: [0x45,0xd5,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 d5 00 c0 <unknown>

mov     za0v.b[w15, 7], p3/m, z13.b
// CHECK-INST: mov     za0v.b[w15, 7], p3/m, z13.b
// CHECK-ENCODING: [0xa7,0xed,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 ed 00 c0 <unknown>

mov     za0v.b[w15, 15], p7/m, z31.b
// CHECK-INST: mov     za0v.b[w15, 15], p7/m, z31.b
// CHECK-ENCODING: [0xef,0xff,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef ff 00 c0 <unknown>

mov     za0v.b[w12, 5], p3/m, z17.b
// CHECK-INST: mov     za0v.b[w12, 5], p3/m, z17.b
// CHECK-ENCODING: [0x25,0x8e,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 8e 00 c0 <unknown>

mov     za0v.b[w12, 1], p1/m, z1.b
// CHECK-INST: mov     za0v.b[w12, 1], p1/m, z1.b
// CHECK-ENCODING: [0x21,0x84,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 84 00 c0 <unknown>

mov     za0v.b[w14, 8], p5/m, z19.b
// CHECK-INST: mov     za0v.b[w14, 8], p5/m, z19.b
// CHECK-ENCODING: [0x68,0xd6,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 d6 00 c0 <unknown>

mov     za0v.b[w12, 0], p6/m, z12.b
// CHECK-INST: mov     za0v.b[w12, 0], p6/m, z12.b
// CHECK-ENCODING: [0x80,0x99,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 99 00 c0 <unknown>

mov     za0v.b[w14, 1], p2/m, z1.b
// CHECK-INST: mov     za0v.b[w14, 1], p2/m, z1.b
// CHECK-ENCODING: [0x21,0xc8,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 c8 00 c0 <unknown>

mov     za0v.b[w12, 13], p2/m, z22.b
// CHECK-INST: mov     za0v.b[w12, 13], p2/m, z22.b
// CHECK-ENCODING: [0xcd,0x8a,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 8a 00 c0 <unknown>

mov     za0v.b[w15, 2], p5/m, z9.b
// CHECK-INST: mov     za0v.b[w15, 2], p5/m, z9.b
// CHECK-ENCODING: [0x22,0xf5,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 f5 00 c0 <unknown>

mov     za0v.b[w13, 7], p2/m, z12.b
// CHECK-INST: mov     za0v.b[w13, 7], p2/m, z12.b
// CHECK-ENCODING: [0x87,0xa9,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 a9 00 c0 <unknown>

// --------------------------------------------------------------------------//
// Insert, vector to tile, horizontal, 16-bit

mova    za0h.h[w12, 0], p0/m, z0.h
// CHECK-INST: mov     za0h.h[w12, 0], p0/m, z0.h
// CHECK-ENCODING: [0x00,0x00,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 00 40 c0 <unknown>

mova    za0h.h[w14, 5], p5/m, z10.h
// CHECK-INST: mov     za0h.h[w14, 5], p5/m, z10.h
// CHECK-ENCODING: [0x45,0x55,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 55 40 c0 <unknown>

mova    za0h.h[w15, 7], p3/m, z13.h
// CHECK-INST: mov     za0h.h[w15, 7], p3/m, z13.h
// CHECK-ENCODING: [0xa7,0x6d,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 6d 40 c0 <unknown>

mova    za1h.h[w15, 7], p7/m, z31.h
// CHECK-INST: mov     za1h.h[w15, 7], p7/m, z31.h
// CHECK-ENCODING: [0xef,0x7f,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef 7f 40 c0 <unknown>

mova    za0h.h[w12, 5], p3/m, z17.h
// CHECK-INST: mov     za0h.h[w12, 5], p3/m, z17.h
// CHECK-ENCODING: [0x25,0x0e,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 0e 40 c0 <unknown>

mova    za0h.h[w12, 1], p1/m, z1.h
// CHECK-INST: mov     za0h.h[w12, 1], p1/m, z1.h
// CHECK-ENCODING: [0x21,0x04,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 04 40 c0 <unknown>

mova    za1h.h[w14, 0], p5/m, z19.h
// CHECK-INST: mov     za1h.h[w14, 0], p5/m, z19.h
// CHECK-ENCODING: [0x68,0x56,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 56 40 c0 <unknown>

mova    za0h.h[w12, 0], p6/m, z12.h
// CHECK-INST: mov     za0h.h[w12, 0], p6/m, z12.h
// CHECK-ENCODING: [0x80,0x19,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 19 40 c0 <unknown>

mova    za0h.h[w14, 1], p2/m, z1.h
// CHECK-INST: mov     za0h.h[w14, 1], p2/m, z1.h
// CHECK-ENCODING: [0x21,0x48,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 48 40 c0 <unknown>

mova    za1h.h[w12, 5], p2/m, z22.h
// CHECK-INST: mov     za1h.h[w12, 5], p2/m, z22.h
// CHECK-ENCODING: [0xcd,0x0a,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 0a 40 c0 <unknown>

mova    za0h.h[w15, 2], p5/m, z9.h
// CHECK-INST: mov     za0h.h[w15, 2], p5/m, z9.h
// CHECK-ENCODING: [0x22,0x75,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 75 40 c0 <unknown>

mova    za0h.h[w13, 7], p2/m, z12.h
// CHECK-INST: mov     za0h.h[w13, 7], p2/m, z12.h
// CHECK-ENCODING: [0x87,0x29,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 29 40 c0 <unknown>

// Aliases

mov     za0h.h[w12, 0], p0/m, z0.h
// CHECK-INST: mov     za0h.h[w12, 0], p0/m, z0.h
// CHECK-ENCODING: [0x00,0x00,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 00 40 c0 <unknown>

mov     za0h.h[w14, 5], p5/m, z10.h
// CHECK-INST: mov     za0h.h[w14, 5], p5/m, z10.h
// CHECK-ENCODING: [0x45,0x55,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 55 40 c0 <unknown>

mov     za0h.h[w15, 7], p3/m, z13.h
// CHECK-INST: mov     za0h.h[w15, 7], p3/m, z13.h
// CHECK-ENCODING: [0xa7,0x6d,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 6d 40 c0 <unknown>

mov     za1h.h[w15, 7], p7/m, z31.h
// CHECK-INST: mov     za1h.h[w15, 7], p7/m, z31.h
// CHECK-ENCODING: [0xef,0x7f,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef 7f 40 c0 <unknown>

mov     za0h.h[w12, 5], p3/m, z17.h
// CHECK-INST: mov     za0h.h[w12, 5], p3/m, z17.h
// CHECK-ENCODING: [0x25,0x0e,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 0e 40 c0 <unknown>

mov     za0h.h[w12, 1], p1/m, z1.h
// CHECK-INST: mov     za0h.h[w12, 1], p1/m, z1.h
// CHECK-ENCODING: [0x21,0x04,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 04 40 c0 <unknown>

mov     za1h.h[w14, 0], p5/m, z19.h
// CHECK-INST: mov     za1h.h[w14, 0], p5/m, z19.h
// CHECK-ENCODING: [0x68,0x56,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 56 40 c0 <unknown>

mov     za0h.h[w12, 0], p6/m, z12.h
// CHECK-INST: mov     za0h.h[w12, 0], p6/m, z12.h
// CHECK-ENCODING: [0x80,0x19,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 19 40 c0 <unknown>

mov     za0h.h[w14, 1], p2/m, z1.h
// CHECK-INST: mov     za0h.h[w14, 1], p2/m, z1.h
// CHECK-ENCODING: [0x21,0x48,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 48 40 c0 <unknown>

mov     za1h.h[w12, 5], p2/m, z22.h
// CHECK-INST: mov     za1h.h[w12, 5], p2/m, z22.h
// CHECK-ENCODING: [0xcd,0x0a,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 0a 40 c0 <unknown>

mov     za0h.h[w15, 2], p5/m, z9.h
// CHECK-INST: mov     za0h.h[w15, 2], p5/m, z9.h
// CHECK-ENCODING: [0x22,0x75,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 75 40 c0 <unknown>

mov     za0h.h[w13, 7], p2/m, z12.h
// CHECK-INST: mov     za0h.h[w13, 7], p2/m, z12.h
// CHECK-ENCODING: [0x87,0x29,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 29 40 c0 <unknown>

// --------------------------------------------------------------------------//
// Insert, vector to tile, vertical, 16-bit

mova    za0v.h[w12, 0], p0/m, z0.h
// CHECK-INST: mov     za0v.h[w12, 0], p0/m, z0.h
// CHECK-ENCODING: [0x00,0x80,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 80 40 c0 <unknown>

mova    za0v.h[w14, 5], p5/m, z10.h
// CHECK-INST: mov     za0v.h[w14, 5], p5/m, z10.h
// CHECK-ENCODING: [0x45,0xd5,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 d5 40 c0 <unknown>

mova    za0v.h[w15, 7], p3/m, z13.h
// CHECK-INST: mov     za0v.h[w15, 7], p3/m, z13.h
// CHECK-ENCODING: [0xa7,0xed,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 ed 40 c0 <unknown>

mova    za1v.h[w15, 7], p7/m, z31.h
// CHECK-INST: mov     za1v.h[w15, 7], p7/m, z31.h
// CHECK-ENCODING: [0xef,0xff,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef ff 40 c0 <unknown>

mova    za0v.h[w12, 5], p3/m, z17.h
// CHECK-INST: mov     za0v.h[w12, 5], p3/m, z17.h
// CHECK-ENCODING: [0x25,0x8e,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 8e 40 c0 <unknown>

mova    za0v.h[w12, 1], p1/m, z1.h
// CHECK-INST: mov     za0v.h[w12, 1], p1/m, z1.h
// CHECK-ENCODING: [0x21,0x84,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 84 40 c0 <unknown>

mova    za1v.h[w14, 0], p5/m, z19.h
// CHECK-INST: mov     za1v.h[w14, 0], p5/m, z19.h
// CHECK-ENCODING: [0x68,0xd6,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 d6 40 c0 <unknown>

mova    za0v.h[w12, 0], p6/m, z12.h
// CHECK-INST: mov     za0v.h[w12, 0], p6/m, z12.h
// CHECK-ENCODING: [0x80,0x99,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 99 40 c0 <unknown>

mova    za0v.h[w14, 1], p2/m, z1.h
// CHECK-INST: mov     za0v.h[w14, 1], p2/m, z1.h
// CHECK-ENCODING: [0x21,0xc8,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 c8 40 c0 <unknown>

mova    za1v.h[w12, 5], p2/m, z22.h
// CHECK-INST: mov     za1v.h[w12, 5], p2/m, z22.h
// CHECK-ENCODING: [0xcd,0x8a,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 8a 40 c0 <unknown>

mova    za0v.h[w15, 2], p5/m, z9.h
// CHECK-INST: mov     za0v.h[w15, 2], p5/m, z9.h
// CHECK-ENCODING: [0x22,0xf5,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 f5 40 c0 <unknown>

mova    za0v.h[w13, 7], p2/m, z12.h
// CHECK-INST: mov     za0v.h[w13, 7], p2/m, z12.h
// CHECK-ENCODING: [0x87,0xa9,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 a9 40 c0 <unknown>

// Aliases

mov     za0v.h[w12, 0], p0/m, z0.h
// CHECK-INST: mov     za0v.h[w12, 0], p0/m, z0.h
// CHECK-ENCODING: [0x00,0x80,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 80 40 c0 <unknown>

mov     za0v.h[w14, 5], p5/m, z10.h
// CHECK-INST: mov     za0v.h[w14, 5], p5/m, z10.h
// CHECK-ENCODING: [0x45,0xd5,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 d5 40 c0 <unknown>

mov     za0v.h[w15, 7], p3/m, z13.h
// CHECK-INST: mov     za0v.h[w15, 7], p3/m, z13.h
// CHECK-ENCODING: [0xa7,0xed,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 ed 40 c0 <unknown>

mov     za1v.h[w15, 7], p7/m, z31.h
// CHECK-INST: mov     za1v.h[w15, 7], p7/m, z31.h
// CHECK-ENCODING: [0xef,0xff,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef ff 40 c0 <unknown>

mov     za0v.h[w12, 5], p3/m, z17.h
// CHECK-INST: mov     za0v.h[w12, 5], p3/m, z17.h
// CHECK-ENCODING: [0x25,0x8e,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 8e 40 c0 <unknown>

mov     za0v.h[w12, 1], p1/m, z1.h
// CHECK-INST: mov     za0v.h[w12, 1], p1/m, z1.h
// CHECK-ENCODING: [0x21,0x84,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 84 40 c0 <unknown>

mov     za1v.h[w14, 0], p5/m, z19.h
// CHECK-INST: mov     za1v.h[w14, 0], p5/m, z19.h
// CHECK-ENCODING: [0x68,0xd6,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 d6 40 c0 <unknown>

mov     za0v.h[w12, 0], p6/m, z12.h
// CHECK-INST: mov     za0v.h[w12, 0], p6/m, z12.h
// CHECK-ENCODING: [0x80,0x99,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 99 40 c0 <unknown>

mov     za0v.h[w14, 1], p2/m, z1.h
// CHECK-INST: mov     za0v.h[w14, 1], p2/m, z1.h
// CHECK-ENCODING: [0x21,0xc8,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 c8 40 c0 <unknown>

mov     za1v.h[w12, 5], p2/m, z22.h
// CHECK-INST: mov     za1v.h[w12, 5], p2/m, z22.h
// CHECK-ENCODING: [0xcd,0x8a,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 8a 40 c0 <unknown>

mov     za0v.h[w15, 2], p5/m, z9.h
// CHECK-INST: mov     za0v.h[w15, 2], p5/m, z9.h
// CHECK-ENCODING: [0x22,0xf5,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 f5 40 c0 <unknown>

mov     za0v.h[w13, 7], p2/m, z12.h
// CHECK-INST: mov     za0v.h[w13, 7], p2/m, z12.h
// CHECK-ENCODING: [0x87,0xa9,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 a9 40 c0 <unknown>

// --------------------------------------------------------------------------//
// Insert, vector to tile, horizontal, 32-bit

mova    za0h.s[w12, 0], p0/m, z0.s
// CHECK-INST: mov     za0h.s[w12, 0], p0/m, z0.s
// CHECK-ENCODING: [0x00,0x00,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 00 80 c0 <unknown>

mova    za1h.s[w14, 1], p5/m, z10.s
// CHECK-INST: mov     za1h.s[w14, 1], p5/m, z10.s
// CHECK-ENCODING: [0x45,0x55,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 55 80 c0 <unknown>

mova    za1h.s[w15, 3], p3/m, z13.s
// CHECK-INST: mov     za1h.s[w15, 3], p3/m, z13.s
// CHECK-ENCODING: [0xa7,0x6d,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 6d 80 c0 <unknown>

mova    za3h.s[w15, 3], p7/m, z31.s
// CHECK-INST: mov     za3h.s[w15, 3], p7/m, z31.s
// CHECK-ENCODING: [0xef,0x7f,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef 7f 80 c0 <unknown>

mova    za1h.s[w12, 1], p3/m, z17.s
// CHECK-INST: mov     za1h.s[w12, 1], p3/m, z17.s
// CHECK-ENCODING: [0x25,0x0e,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 0e 80 c0 <unknown>

mova    za0h.s[w12, 1], p1/m, z1.s
// CHECK-INST: mov     za0h.s[w12, 1], p1/m, z1.s
// CHECK-ENCODING: [0x21,0x04,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 04 80 c0 <unknown>

mova    za2h.s[w14, 0], p5/m, z19.s
// CHECK-INST: mov     za2h.s[w14, 0], p5/m, z19.s
// CHECK-ENCODING: [0x68,0x56,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 56 80 c0 <unknown>

mova    za0h.s[w12, 0], p6/m, z12.s
// CHECK-INST: mov     za0h.s[w12, 0], p6/m, z12.s
// CHECK-ENCODING: [0x80,0x19,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 19 80 c0 <unknown>

mova    za0h.s[w14, 1], p2/m, z1.s
// CHECK-INST: mov     za0h.s[w14, 1], p2/m, z1.s
// CHECK-ENCODING: [0x21,0x48,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 48 80 c0 <unknown>

mova    za3h.s[w12, 1], p2/m, z22.s
// CHECK-INST: mov     za3h.s[w12, 1], p2/m, z22.s
// CHECK-ENCODING: [0xcd,0x0a,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 0a 80 c0 <unknown>

mova    za0h.s[w15, 2], p5/m, z9.s
// CHECK-INST: mov     za0h.s[w15, 2], p5/m, z9.s
// CHECK-ENCODING: [0x22,0x75,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 75 80 c0 <unknown>

mova    za1h.s[w13, 3], p2/m, z12.s
// CHECK-INST: mov     za1h.s[w13, 3], p2/m, z12.s
// CHECK-ENCODING: [0x87,0x29,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 29 80 c0 <unknown>

// Aliases

mov     za0h.s[w12, 0], p0/m, z0.s
// CHECK-INST: mov     za0h.s[w12, 0], p0/m, z0.s
// CHECK-ENCODING: [0x00,0x00,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 00 80 c0 <unknown>

mov     za1h.s[w14, 1], p5/m, z10.s
// CHECK-INST: mov     za1h.s[w14, 1], p5/m, z10.s
// CHECK-ENCODING: [0x45,0x55,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 55 80 c0 <unknown>

mov     za1h.s[w15, 3], p3/m, z13.s
// CHECK-INST: mov     za1h.s[w15, 3], p3/m, z13.s
// CHECK-ENCODING: [0xa7,0x6d,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 6d 80 c0 <unknown>

mov     za3h.s[w15, 3], p7/m, z31.s
// CHECK-INST: mov     za3h.s[w15, 3], p7/m, z31.s
// CHECK-ENCODING: [0xef,0x7f,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef 7f 80 c0 <unknown>

mov     za1h.s[w12, 1], p3/m, z17.s
// CHECK-INST: mov     za1h.s[w12, 1], p3/m, z17.s
// CHECK-ENCODING: [0x25,0x0e,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 0e 80 c0 <unknown>

mov     za0h.s[w12, 1], p1/m, z1.s
// CHECK-INST: mov     za0h.s[w12, 1], p1/m, z1.s
// CHECK-ENCODING: [0x21,0x04,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 04 80 c0 <unknown>

mov     za2h.s[w14, 0], p5/m, z19.s
// CHECK-INST: mov     za2h.s[w14, 0], p5/m, z19.s
// CHECK-ENCODING: [0x68,0x56,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 56 80 c0 <unknown>

mov     za0h.s[w12, 0], p6/m, z12.s
// CHECK-INST: mov     za0h.s[w12, 0], p6/m, z12.s
// CHECK-ENCODING: [0x80,0x19,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 19 80 c0 <unknown>

mov     za0h.s[w14, 1], p2/m, z1.s
// CHECK-INST: mov     za0h.s[w14, 1], p2/m, z1.s
// CHECK-ENCODING: [0x21,0x48,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 48 80 c0 <unknown>

mov     za3h.s[w12, 1], p2/m, z22.s
// CHECK-INST: mov     za3h.s[w12, 1], p2/m, z22.s
// CHECK-ENCODING: [0xcd,0x0a,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 0a 80 c0 <unknown>

mov     za0h.s[w15, 2], p5/m, z9.s
// CHECK-INST: mov     za0h.s[w15, 2], p5/m, z9.s
// CHECK-ENCODING: [0x22,0x75,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 75 80 c0 <unknown>

mov     za1h.s[w13, 3], p2/m, z12.s
// CHECK-INST: mov     za1h.s[w13, 3], p2/m, z12.s
// CHECK-ENCODING: [0x87,0x29,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 29 80 c0 <unknown>

// --------------------------------------------------------------------------//
// Insert, vector to tile, vertical, 32-bit

mova    za0v.s[w12, 0], p0/m, z0.s
// CHECK-INST: mov     za0v.s[w12, 0], p0/m, z0.s
// CHECK-ENCODING: [0x00,0x80,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 80 80 c0 <unknown>

mova    za1v.s[w14, 1], p5/m, z10.s
// CHECK-INST: mov     za1v.s[w14, 1], p5/m, z10.s
// CHECK-ENCODING: [0x45,0xd5,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 d5 80 c0 <unknown>

mova    za1v.s[w15, 3], p3/m, z13.s
// CHECK-INST: mov     za1v.s[w15, 3], p3/m, z13.s
// CHECK-ENCODING: [0xa7,0xed,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 ed 80 c0 <unknown>

mova    za3v.s[w15, 3], p7/m, z31.s
// CHECK-INST: mov     za3v.s[w15, 3], p7/m, z31.s
// CHECK-ENCODING: [0xef,0xff,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef ff 80 c0 <unknown>

mova    za1v.s[w12, 1], p3/m, z17.s
// CHECK-INST: mov     za1v.s[w12, 1], p3/m, z17.s
// CHECK-ENCODING: [0x25,0x8e,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 8e 80 c0 <unknown>

mova    za0v.s[w12, 1], p1/m, z1.s
// CHECK-INST: mov     za0v.s[w12, 1], p1/m, z1.s
// CHECK-ENCODING: [0x21,0x84,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 84 80 c0 <unknown>

mova    za2v.s[w14, 0], p5/m, z19.s
// CHECK-INST: mov     za2v.s[w14, 0], p5/m, z19.s
// CHECK-ENCODING: [0x68,0xd6,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 d6 80 c0 <unknown>

mova    za0v.s[w12, 0], p6/m, z12.s
// CHECK-INST: mov     za0v.s[w12, 0], p6/m, z12.s
// CHECK-ENCODING: [0x80,0x99,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 99 80 c0 <unknown>

mova    za0v.s[w14, 1], p2/m, z1.s
// CHECK-INST: mov     za0v.s[w14, 1], p2/m, z1.s
// CHECK-ENCODING: [0x21,0xc8,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 c8 80 c0 <unknown>

mova    za3v.s[w12, 1], p2/m, z22.s
// CHECK-INST: mov     za3v.s[w12, 1], p2/m, z22.s
// CHECK-ENCODING: [0xcd,0x8a,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 8a 80 c0 <unknown>

mova    za0v.s[w15, 2], p5/m, z9.s
// CHECK-INST: mov     za0v.s[w15, 2], p5/m, z9.s
// CHECK-ENCODING: [0x22,0xf5,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 f5 80 c0 <unknown>

mova    za1v.s[w13, 3], p2/m, z12.s
// CHECK-INST: mov     za1v.s[w13, 3], p2/m, z12.s
// CHECK-ENCODING: [0x87,0xa9,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 a9 80 c0 <unknown>

// Aliases

mov     za0v.s[w12, 0], p0/m, z0.s
// CHECK-INST: mov     za0v.s[w12, 0], p0/m, z0.s
// CHECK-ENCODING: [0x00,0x80,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 80 80 c0 <unknown>

mov     za1v.s[w14, 1], p5/m, z10.s
// CHECK-INST: mov     za1v.s[w14, 1], p5/m, z10.s
// CHECK-ENCODING: [0x45,0xd5,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 d5 80 c0 <unknown>

mov     za1v.s[w15, 3], p3/m, z13.s
// CHECK-INST: mov     za1v.s[w15, 3], p3/m, z13.s
// CHECK-ENCODING: [0xa7,0xed,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 ed 80 c0 <unknown>

mov     za3v.s[w15, 3], p7/m, z31.s
// CHECK-INST: mov     za3v.s[w15, 3], p7/m, z31.s
// CHECK-ENCODING: [0xef,0xff,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef ff 80 c0 <unknown>

mov     za1v.s[w12, 1], p3/m, z17.s
// CHECK-INST: mov     za1v.s[w12, 1], p3/m, z17.s
// CHECK-ENCODING: [0x25,0x8e,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 8e 80 c0 <unknown>

mov     za0v.s[w12, 1], p1/m, z1.s
// CHECK-INST: mov     za0v.s[w12, 1], p1/m, z1.s
// CHECK-ENCODING: [0x21,0x84,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 84 80 c0 <unknown>

mov     za2v.s[w14, 0], p5/m, z19.s
// CHECK-INST: mov     za2v.s[w14, 0], p5/m, z19.s
// CHECK-ENCODING: [0x68,0xd6,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 d6 80 c0 <unknown>

mov     za0v.s[w12, 0], p6/m, z12.s
// CHECK-INST: mov     za0v.s[w12, 0], p6/m, z12.s
// CHECK-ENCODING: [0x80,0x99,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 99 80 c0 <unknown>

mov     za0v.s[w14, 1], p2/m, z1.s
// CHECK-INST: mov     za0v.s[w14, 1], p2/m, z1.s
// CHECK-ENCODING: [0x21,0xc8,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 c8 80 c0 <unknown>

mov     za3v.s[w12, 1], p2/m, z22.s
// CHECK-INST: mov     za3v.s[w12, 1], p2/m, z22.s
// CHECK-ENCODING: [0xcd,0x8a,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 8a 80 c0 <unknown>

mov     za0v.s[w15, 2], p5/m, z9.s
// CHECK-INST: mov     za0v.s[w15, 2], p5/m, z9.s
// CHECK-ENCODING: [0x22,0xf5,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 f5 80 c0 <unknown>

mov     za1v.s[w13, 3], p2/m, z12.s
// CHECK-INST: mov     za1v.s[w13, 3], p2/m, z12.s
// CHECK-ENCODING: [0x87,0xa9,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 a9 80 c0 <unknown>

// --------------------------------------------------------------------------//
// Insert, vector to tile, horizontal, 64-bit

mova    za0h.d[w12, 0], p0/m, z0.d
// CHECK-INST: mov     za0h.d[w12, 0], p0/m, z0.d
// CHECK-ENCODING: [0x00,0x00,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 00 c0 c0 <unknown>

mova    za2h.d[w14, 1], p5/m, z10.d
// CHECK-INST: mov     za2h.d[w14, 1], p5/m, z10.d
// CHECK-ENCODING: [0x45,0x55,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 55 c0 c0 <unknown>

mova    za3h.d[w15, 1], p3/m, z13.d
// CHECK-INST: mov     za3h.d[w15, 1], p3/m, z13.d
// CHECK-ENCODING: [0xa7,0x6d,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 6d c0 c0 <unknown>

mova    za7h.d[w15, 1], p7/m, z31.d
// CHECK-INST: mov     za7h.d[w15, 1], p7/m, z31.d
// CHECK-ENCODING: [0xef,0x7f,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef 7f c0 c0 <unknown>

mova    za2h.d[w12, 1], p3/m, z17.d
// CHECK-INST: mov     za2h.d[w12, 1], p3/m, z17.d
// CHECK-ENCODING: [0x25,0x0e,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 0e c0 c0 <unknown>

mova    za0h.d[w12, 1], p1/m, z1.d
// CHECK-INST: mov     za0h.d[w12, 1], p1/m, z1.d
// CHECK-ENCODING: [0x21,0x04,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 04 c0 c0 <unknown>

mova    za4h.d[w14, 0], p5/m, z19.d
// CHECK-INST: mov     za4h.d[w14, 0], p5/m, z19.d
// CHECK-ENCODING: [0x68,0x56,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 56 c0 c0 <unknown>

mova    za0h.d[w12, 0], p6/m, z12.d
// CHECK-INST: mov     za0h.d[w12, 0], p6/m, z12.d
// CHECK-ENCODING: [0x80,0x19,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 19 c0 c0 <unknown>

mova    za0h.d[w14, 1], p2/m, z1.d
// CHECK-INST: mov     za0h.d[w14, 1], p2/m, z1.d
// CHECK-ENCODING: [0x21,0x48,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 48 c0 c0 <unknown>

mova    za6h.d[w12, 1], p2/m, z22.d
// CHECK-INST: mov     za6h.d[w12, 1], p2/m, z22.d
// CHECK-ENCODING: [0xcd,0x0a,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 0a c0 c0 <unknown>

mova    za1h.d[w15, 0], p5/m, z9.d
// CHECK-INST: mov     za1h.d[w15, 0], p5/m, z9.d
// CHECK-ENCODING: [0x22,0x75,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 75 c0 c0 <unknown>

mova    za3h.d[w13, 1], p2/m, z12.d
// CHECK-INST: mov     za3h.d[w13, 1], p2/m, z12.d
// CHECK-ENCODING: [0x87,0x29,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 29 c0 c0 <unknown>

// Aliases

mov     za0h.d[w12, 0], p0/m, z0.d
// CHECK-INST: mov     za0h.d[w12, 0], p0/m, z0.d
// CHECK-ENCODING: [0x00,0x00,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 00 c0 c0 <unknown>

mov     za2h.d[w14, 1], p5/m, z10.d
// CHECK-INST: mov     za2h.d[w14, 1], p5/m, z10.d
// CHECK-ENCODING: [0x45,0x55,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 55 c0 c0 <unknown>

mov     za3h.d[w15, 1], p3/m, z13.d
// CHECK-INST: mov     za3h.d[w15, 1], p3/m, z13.d
// CHECK-ENCODING: [0xa7,0x6d,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 6d c0 c0 <unknown>

mov     za7h.d[w15, 1], p7/m, z31.d
// CHECK-INST: mov     za7h.d[w15, 1], p7/m, z31.d
// CHECK-ENCODING: [0xef,0x7f,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef 7f c0 c0 <unknown>

mov     za2h.d[w12, 1], p3/m, z17.d
// CHECK-INST: mov     za2h.d[w12, 1], p3/m, z17.d
// CHECK-ENCODING: [0x25,0x0e,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 0e c0 c0 <unknown>

mov     za0h.d[w12, 1], p1/m, z1.d
// CHECK-INST: mov     za0h.d[w12, 1], p1/m, z1.d
// CHECK-ENCODING: [0x21,0x04,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 04 c0 c0 <unknown>

mov     za4h.d[w14, 0], p5/m, z19.d
// CHECK-INST: mov     za4h.d[w14, 0], p5/m, z19.d
// CHECK-ENCODING: [0x68,0x56,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 56 c0 c0 <unknown>

mov     za0h.d[w12, 0], p6/m, z12.d
// CHECK-INST: mov     za0h.d[w12, 0], p6/m, z12.d
// CHECK-ENCODING: [0x80,0x19,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 19 c0 c0 <unknown>

mov     za0h.d[w14, 1], p2/m, z1.d
// CHECK-INST: mov     za0h.d[w14, 1], p2/m, z1.d
// CHECK-ENCODING: [0x21,0x48,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 48 c0 c0 <unknown>

mov     za6h.d[w12, 1], p2/m, z22.d
// CHECK-INST: mov     za6h.d[w12, 1], p2/m, z22.d
// CHECK-ENCODING: [0xcd,0x0a,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 0a c0 c0 <unknown>

mov     za1h.d[w15, 0], p5/m, z9.d
// CHECK-INST: mov     za1h.d[w15, 0], p5/m, z9.d
// CHECK-ENCODING: [0x22,0x75,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 75 c0 c0 <unknown>

mov     za3h.d[w13, 1], p2/m, z12.d
// CHECK-INST: mov     za3h.d[w13, 1], p2/m, z12.d
// CHECK-ENCODING: [0x87,0x29,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 29 c0 c0 <unknown>

// --------------------------------------------------------------------------//
// Insert, vector to tile, vertical, 64-bit

mova    za0v.d[w12, 0], p0/m, z0.d
// CHECK-INST: mov     za0v.d[w12, 0], p0/m, z0.d
// CHECK-ENCODING: [0x00,0x80,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 80 c0 c0 <unknown>

mova    za2v.d[w14, 1], p5/m, z10.d
// CHECK-INST: mov     za2v.d[w14, 1], p5/m, z10.d
// CHECK-ENCODING: [0x45,0xd5,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 d5 c0 c0 <unknown>

mova    za3v.d[w15, 1], p3/m, z13.d
// CHECK-INST: mov     za3v.d[w15, 1], p3/m, z13.d
// CHECK-ENCODING: [0xa7,0xed,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 ed c0 c0 <unknown>

mova    za7v.d[w15, 1], p7/m, z31.d
// CHECK-INST: mov     za7v.d[w15, 1], p7/m, z31.d
// CHECK-ENCODING: [0xef,0xff,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef ff c0 c0 <unknown>

mova    za2v.d[w12, 1], p3/m, z17.d
// CHECK-INST: mov     za2v.d[w12, 1], p3/m, z17.d
// CHECK-ENCODING: [0x25,0x8e,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 8e c0 c0 <unknown>

mova    za0v.d[w12, 1], p1/m, z1.d
// CHECK-INST: mov     za0v.d[w12, 1], p1/m, z1.d
// CHECK-ENCODING: [0x21,0x84,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 84 c0 c0 <unknown>

mova    za4v.d[w14, 0], p5/m, z19.d
// CHECK-INST: mov     za4v.d[w14, 0], p5/m, z19.d
// CHECK-ENCODING: [0x68,0xd6,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 d6 c0 c0 <unknown>

mova    za0v.d[w12, 0], p6/m, z12.d
// CHECK-INST: mov     za0v.d[w12, 0], p6/m, z12.d
// CHECK-ENCODING: [0x80,0x99,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 99 c0 c0 <unknown>

mova    za0v.d[w14, 1], p2/m, z1.d
// CHECK-INST: mov     za0v.d[w14, 1], p2/m, z1.d
// CHECK-ENCODING: [0x21,0xc8,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 c8 c0 c0 <unknown>

mova    za6v.d[w12, 1], p2/m, z22.d
// CHECK-INST: mov     za6v.d[w12, 1], p2/m, z22.d
// CHECK-ENCODING: [0xcd,0x8a,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 8a c0 c0 <unknown>

mova    za1v.d[w15, 0], p5/m, z9.d
// CHECK-INST: mov     za1v.d[w15, 0], p5/m, z9.d
// CHECK-ENCODING: [0x22,0xf5,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 f5 c0 c0 <unknown>

mova    za3v.d[w13, 1], p2/m, z12.d
// CHECK-INST: mov     za3v.d[w13, 1], p2/m, z12.d
// CHECK-ENCODING: [0x87,0xa9,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 a9 c0 c0 <unknown>

// Aliases

mov     za0v.d[w12, 0], p0/m, z0.d
// CHECK-INST: mov     za0v.d[w12, 0], p0/m, z0.d
// CHECK-ENCODING: [0x00,0x80,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 80 c0 c0 <unknown>

mov     za2v.d[w14, 1], p5/m, z10.d
// CHECK-INST: mov     za2v.d[w14, 1], p5/m, z10.d
// CHECK-ENCODING: [0x45,0xd5,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 d5 c0 c0 <unknown>

mov     za3v.d[w15, 1], p3/m, z13.d
// CHECK-INST: mov     za3v.d[w15, 1], p3/m, z13.d
// CHECK-ENCODING: [0xa7,0xed,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 ed c0 c0 <unknown>

mov     za7v.d[w15, 1], p7/m, z31.d
// CHECK-INST: mov     za7v.d[w15, 1], p7/m, z31.d
// CHECK-ENCODING: [0xef,0xff,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef ff c0 c0 <unknown>

mov     za2v.d[w12, 1], p3/m, z17.d
// CHECK-INST: mov     za2v.d[w12, 1], p3/m, z17.d
// CHECK-ENCODING: [0x25,0x8e,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 8e c0 c0 <unknown>

mov     za0v.d[w12, 1], p1/m, z1.d
// CHECK-INST: mov     za0v.d[w12, 1], p1/m, z1.d
// CHECK-ENCODING: [0x21,0x84,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 84 c0 c0 <unknown>

mov     za4v.d[w14, 0], p5/m, z19.d
// CHECK-INST: mov     za4v.d[w14, 0], p5/m, z19.d
// CHECK-ENCODING: [0x68,0xd6,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 d6 c0 c0 <unknown>

mov     za0v.d[w12, 0], p6/m, z12.d
// CHECK-INST: mov     za0v.d[w12, 0], p6/m, z12.d
// CHECK-ENCODING: [0x80,0x99,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 99 c0 c0 <unknown>

mov     za0v.d[w14, 1], p2/m, z1.d
// CHECK-INST: mov     za0v.d[w14, 1], p2/m, z1.d
// CHECK-ENCODING: [0x21,0xc8,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 c8 c0 c0 <unknown>

mov     za6v.d[w12, 1], p2/m, z22.d
// CHECK-INST: mov     za6v.d[w12, 1], p2/m, z22.d
// CHECK-ENCODING: [0xcd,0x8a,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 8a c0 c0 <unknown>

mov     za1v.d[w15, 0], p5/m, z9.d
// CHECK-INST: mov     za1v.d[w15, 0], p5/m, z9.d
// CHECK-ENCODING: [0x22,0xf5,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 f5 c0 c0 <unknown>

mov     za3v.d[w13, 1], p2/m, z12.d
// CHECK-INST: mov     za3v.d[w13, 1], p2/m, z12.d
// CHECK-ENCODING: [0x87,0xa9,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 a9 c0 c0 <unknown>

// --------------------------------------------------------------------------//
// Insert, vector to tile, horizontal, 128-bit

mova    za0h.q[w12, 0], p0/m, z0.q
// CHECK-INST: mov     za0h.q[w12, 0], p0/m, z0.q
// CHECK-ENCODING: [0x00,0x00,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 00 c1 c0 <unknown>

mova    za5h.q[w14, 0], p5/m, z10.q
// CHECK-INST: mov     za5h.q[w14, 0], p5/m, z10.q
// CHECK-ENCODING: [0x45,0x55,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 55 c1 c0 <unknown>

mova    za7h.q[w15, 0], p3/m, z13.q
// CHECK-INST: mov     za7h.q[w15, 0], p3/m, z13.q
// CHECK-ENCODING: [0xa7,0x6d,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 6d c1 c0 <unknown>

mova    za15h.q[w15, 0], p7/m, z31.q
// CHECK-INST: mov     za15h.q[w15, 0], p7/m, z31.q
// CHECK-ENCODING: [0xef,0x7f,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef 7f c1 c0 <unknown>

mova    za5h.q[w12, 0], p3/m, z17.q
// CHECK-INST: mov     za5h.q[w12, 0], p3/m, z17.q
// CHECK-ENCODING: [0x25,0x0e,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 0e c1 c0 <unknown>

mova    za1h.q[w12, 0], p1/m, z1.q
// CHECK-INST: mov     za1h.q[w12, 0], p1/m, z1.q
// CHECK-ENCODING: [0x21,0x04,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 04 c1 c0 <unknown>

mova    za8h.q[w14, 0], p5/m, z19.q
// CHECK-INST: mov     za8h.q[w14, 0], p5/m, z19.q
// CHECK-ENCODING: [0x68,0x56,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 56 c1 c0 <unknown>

mova    za0h.q[w12, 0], p6/m, z12.q
// CHECK-INST: mov     za0h.q[w12, 0], p6/m, z12.q
// CHECK-ENCODING: [0x80,0x19,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 19 c1 c0 <unknown>

mova    za1h.q[w14, 0], p2/m, z1.q
// CHECK-INST: mov     za1h.q[w14, 0], p2/m, z1.q
// CHECK-ENCODING: [0x21,0x48,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 48 c1 c0 <unknown>

mova    za13h.q[w12, 0], p2/m, z22.q
// CHECK-INST: mov     za13h.q[w12, 0], p2/m, z22.q
// CHECK-ENCODING: [0xcd,0x0a,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 0a c1 c0 <unknown>

mova    za2h.q[w15, 0], p5/m, z9.q
// CHECK-INST: mov     za2h.q[w15, 0], p5/m, z9.q
// CHECK-ENCODING: [0x22,0x75,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 75 c1 c0 <unknown>

mova    za7h.q[w13, 0], p2/m, z12.q
// CHECK-INST: mov     za7h.q[w13, 0], p2/m, z12.q
// CHECK-ENCODING: [0x87,0x29,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 29 c1 c0 <unknown>

// Aliases

mov     za0h.q[w12, 0], p0/m, z0.q
// CHECK-INST: mov     za0h.q[w12, 0], p0/m, z0.q
// CHECK-ENCODING: [0x00,0x00,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 00 c1 c0 <unknown>

mov     za5h.q[w14, 0], p5/m, z10.q
// CHECK-INST: mov     za5h.q[w14, 0], p5/m, z10.q
// CHECK-ENCODING: [0x45,0x55,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 55 c1 c0 <unknown>

mov     za7h.q[w15, 0], p3/m, z13.q
// CHECK-INST: mov     za7h.q[w15, 0], p3/m, z13.q
// CHECK-ENCODING: [0xa7,0x6d,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 6d c1 c0 <unknown>

mov     za15h.q[w15, 0], p7/m, z31.q
// CHECK-INST: mov     za15h.q[w15, 0], p7/m, z31.q
// CHECK-ENCODING: [0xef,0x7f,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef 7f c1 c0 <unknown>

mov     za5h.q[w12, 0], p3/m, z17.q
// CHECK-INST: mov     za5h.q[w12, 0], p3/m, z17.q
// CHECK-ENCODING: [0x25,0x0e,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 0e c1 c0 <unknown>

mov     za1h.q[w12, 0], p1/m, z1.q
// CHECK-INST: mov     za1h.q[w12, 0], p1/m, z1.q
// CHECK-ENCODING: [0x21,0x04,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 04 c1 c0 <unknown>

mov     za8h.q[w14, 0], p5/m, z19.q
// CHECK-INST: mov     za8h.q[w14, 0], p5/m, z19.q
// CHECK-ENCODING: [0x68,0x56,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 56 c1 c0 <unknown>

mov     za0h.q[w12, 0], p6/m, z12.q
// CHECK-INST: mov     za0h.q[w12, 0], p6/m, z12.q
// CHECK-ENCODING: [0x80,0x19,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 19 c1 c0 <unknown>

mov     za1h.q[w14, 0], p2/m, z1.q
// CHECK-INST: mov     za1h.q[w14, 0], p2/m, z1.q
// CHECK-ENCODING: [0x21,0x48,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 48 c1 c0 <unknown>

mov     za13h.q[w12, 0], p2/m, z22.q
// CHECK-INST: mov     za13h.q[w12, 0], p2/m, z22.q
// CHECK-ENCODING: [0xcd,0x0a,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 0a c1 c0 <unknown>

mov     za2h.q[w15, 0], p5/m, z9.q
// CHECK-INST: mov     za2h.q[w15, 0], p5/m, z9.q
// CHECK-ENCODING: [0x22,0x75,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 75 c1 c0 <unknown>

mov     za7h.q[w13, 0], p2/m, z12.q
// CHECK-INST: mov     za7h.q[w13, 0], p2/m, z12.q
// CHECK-ENCODING: [0x87,0x29,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 29 c1 c0 <unknown>

// --------------------------------------------------------------------------//
// Insert, vector to tile, vertical, 128-bit

mova    za0v.q[w12, 0], p0/m, z0.q
// CHECK-INST: mov     za0v.q[w12, 0], p0/m, z0.q
// CHECK-ENCODING: [0x00,0x80,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 80 c1 c0 <unknown>

mova    za5v.q[w14, 0], p5/m, z10.q
// CHECK-INST: mov     za5v.q[w14, 0], p5/m, z10.q
// CHECK-ENCODING: [0x45,0xd5,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 d5 c1 c0 <unknown>

mova    za7v.q[w15, 0], p3/m, z13.q
// CHECK-INST: mov     za7v.q[w15, 0], p3/m, z13.q
// CHECK-ENCODING: [0xa7,0xed,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 ed c1 c0 <unknown>

mova    za15v.q[w15, 0], p7/m, z31.q
// CHECK-INST: mov     za15v.q[w15, 0], p7/m, z31.q
// CHECK-ENCODING: [0xef,0xff,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef ff c1 c0 <unknown>

mova    za5v.q[w12, 0], p3/m, z17.q
// CHECK-INST: mov     za5v.q[w12, 0], p3/m, z17.q
// CHECK-ENCODING: [0x25,0x8e,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 8e c1 c0 <unknown>

mova    za1v.q[w12, 0], p1/m, z1.q
// CHECK-INST: mov     za1v.q[w12, 0], p1/m, z1.q
// CHECK-ENCODING: [0x21,0x84,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 84 c1 c0 <unknown>

mova    za8v.q[w14, 0], p5/m, z19.q
// CHECK-INST: mov     za8v.q[w14, 0], p5/m, z19.q
// CHECK-ENCODING: [0x68,0xd6,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 d6 c1 c0 <unknown>

mova    za0v.q[w12, 0], p6/m, z12.q
// CHECK-INST: mov     za0v.q[w12, 0], p6/m, z12.q
// CHECK-ENCODING: [0x80,0x99,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 99 c1 c0 <unknown>

mova    za1v.q[w14, 0], p2/m, z1.q
// CHECK-INST: mov     za1v.q[w14, 0], p2/m, z1.q
// CHECK-ENCODING: [0x21,0xc8,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 c8 c1 c0 <unknown>

mova    za13v.q[w12, 0], p2/m, z22.q
// CHECK-INST: mov     za13v.q[w12, 0], p2/m, z22.q
// CHECK-ENCODING: [0xcd,0x8a,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 8a c1 c0 <unknown>

mova    za2v.q[w15, 0], p5/m, z9.q
// CHECK-INST: mov     za2v.q[w15, 0], p5/m, z9.q
// CHECK-ENCODING: [0x22,0xf5,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 f5 c1 c0 <unknown>

mova    za7v.q[w13, 0], p2/m, z12.q
// CHECK-INST: mov     za7v.q[w13, 0], p2/m, z12.q
// CHECK-ENCODING: [0x87,0xa9,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 a9 c1 c0 <unknown>

// Aliases

mov     za0v.q[w12, 0], p0/m, z0.q
// CHECK-INST: mov     za0v.q[w12, 0], p0/m, z0.q
// CHECK-ENCODING: [0x00,0x80,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 80 c1 c0 <unknown>

mov     za5v.q[w14, 0], p5/m, z10.q
// CHECK-INST: mov     za5v.q[w14, 0], p5/m, z10.q
// CHECK-ENCODING: [0x45,0xd5,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 d5 c1 c0 <unknown>

mov     za7v.q[w15, 0], p3/m, z13.q
// CHECK-INST: mov     za7v.q[w15, 0], p3/m, z13.q
// CHECK-ENCODING: [0xa7,0xed,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 ed c1 c0 <unknown>

mov     za15v.q[w15, 0], p7/m, z31.q
// CHECK-INST: mov     za15v.q[w15, 0], p7/m, z31.q
// CHECK-ENCODING: [0xef,0xff,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef ff c1 c0 <unknown>

mov     za5v.q[w12, 0], p3/m, z17.q
// CHECK-INST: mov     za5v.q[w12, 0], p3/m, z17.q
// CHECK-ENCODING: [0x25,0x8e,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 8e c1 c0 <unknown>

mov     za1v.q[w12, 0], p1/m, z1.q
// CHECK-INST: mov     za1v.q[w12, 0], p1/m, z1.q
// CHECK-ENCODING: [0x21,0x84,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 84 c1 c0 <unknown>

mov     za8v.q[w14, 0], p5/m, z19.q
// CHECK-INST: mov     za8v.q[w14, 0], p5/m, z19.q
// CHECK-ENCODING: [0x68,0xd6,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 d6 c1 c0 <unknown>

mov     za0v.q[w12, 0], p6/m, z12.q
// CHECK-INST: mov     za0v.q[w12, 0], p6/m, z12.q
// CHECK-ENCODING: [0x80,0x99,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 99 c1 c0 <unknown>

mov     za1v.q[w14, 0], p2/m, z1.q
// CHECK-INST: mov     za1v.q[w14, 0], p2/m, z1.q
// CHECK-ENCODING: [0x21,0xc8,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 c8 c1 c0 <unknown>

mov     za13v.q[w12, 0], p2/m, z22.q
// CHECK-INST: mov     za13v.q[w12, 0], p2/m, z22.q
// CHECK-ENCODING: [0xcd,0x8a,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 8a c1 c0 <unknown>

mov     za2v.q[w15, 0], p5/m, z9.q
// CHECK-INST: mov     za2v.q[w15, 0], p5/m, z9.q
// CHECK-ENCODING: [0x22,0xf5,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 f5 c1 c0 <unknown>

mov     za7v.q[w13, 0], p2/m, z12.q
// CHECK-INST: mov     za7v.q[w13, 0], p2/m, z12.q
// CHECK-ENCODING: [0x87,0xa9,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 a9 c1 c0 <unknown>
