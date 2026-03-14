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
// CHECK-UNKNOWN: c0020000 <unknown>

mova    z21.b, p5/m, za0h.b[w14, 10]
// CHECK-INST: mov     z21.b, p5/m, za0h.b[w14, 10]
// CHECK-ENCODING: [0x55,0x55,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0025555 <unknown>

mova    z23.b, p3/m, za0h.b[w15, 13]
// CHECK-INST: mov     z23.b, p3/m, za0h.b[w15, 13]
// CHECK-ENCODING: [0xb7,0x6d,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0026db7 <unknown>

mova    z31.b, p7/m, za0h.b[w15, 15]
// CHECK-INST: mov     z31.b, p7/m, za0h.b[w15, 15]
// CHECK-ENCODING: [0xff,0x7d,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0027dff <unknown>

mova    z5.b, p3/m, za0h.b[w12, 1]
// CHECK-INST: mov     z5.b, p3/m, za0h.b[w12, 1]
// CHECK-ENCODING: [0x25,0x0c,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0020c25 <unknown>

mova    z1.b, p1/m, za0h.b[w12, 1]
// CHECK-INST: mov     z1.b, p1/m, za0h.b[w12, 1]
// CHECK-ENCODING: [0x21,0x04,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0020421 <unknown>

mova    z24.b, p5/m, za0h.b[w14, 3]
// CHECK-INST: mov     z24.b, p5/m, za0h.b[w14, 3]
// CHECK-ENCODING: [0x78,0x54,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0025478 <unknown>

mova    z0.b, p6/m, za0h.b[w12, 12]
// CHECK-INST: mov     z0.b, p6/m, za0h.b[w12, 12]
// CHECK-ENCODING: [0x80,0x19,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0021980 <unknown>

mova    z17.b, p2/m, za0h.b[w14, 1]
// CHECK-INST: mov     z17.b, p2/m, za0h.b[w14, 1]
// CHECK-ENCODING: [0x31,0x48,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0024831 <unknown>

mova    z29.b, p2/m, za0h.b[w12, 6]
// CHECK-INST: mov     z29.b, p2/m, za0h.b[w12, 6]
// CHECK-ENCODING: [0xdd,0x08,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c00208dd <unknown>

mova    z2.b, p5/m, za0h.b[w15, 9]
// CHECK-INST: mov     z2.b, p5/m, za0h.b[w15, 9]
// CHECK-ENCODING: [0x22,0x75,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0027522 <unknown>

mova    z7.b, p2/m, za0h.b[w13, 12]
// CHECK-INST: mov     z7.b, p2/m, za0h.b[w13, 12]
// CHECK-ENCODING: [0x87,0x29,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0022987 <unknown>

// Aliases

mov     z0.b, p0/m, za0h.b[w12, 0]
// CHECK-INST: mov     z0.b, p0/m, za0h.b[w12, 0]
// CHECK-ENCODING: [0x00,0x00,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0020000 <unknown>

mov     z21.b, p5/m, za0h.b[w14, 10]
// CHECK-INST: mov     z21.b, p5/m, za0h.b[w14, 10]
// CHECK-ENCODING: [0x55,0x55,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0025555 <unknown>

mov     z23.b, p3/m, za0h.b[w15, 13]
// CHECK-INST: mov     z23.b, p3/m, za0h.b[w15, 13]
// CHECK-ENCODING: [0xb7,0x6d,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0026db7 <unknown>

mov     z31.b, p7/m, za0h.b[w15, 15]
// CHECK-INST: mov     z31.b, p7/m, za0h.b[w15, 15]
// CHECK-ENCODING: [0xff,0x7d,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0027dff <unknown>

mov     z5.b, p3/m, za0h.b[w12, 1]
// CHECK-INST: mov     z5.b, p3/m, za0h.b[w12, 1]
// CHECK-ENCODING: [0x25,0x0c,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0020c25 <unknown>

mov     z1.b, p1/m, za0h.b[w12, 1]
// CHECK-INST: mov     z1.b, p1/m, za0h.b[w12, 1]
// CHECK-ENCODING: [0x21,0x04,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0020421 <unknown>

mov     z24.b, p5/m, za0h.b[w14, 3]
// CHECK-INST: mov     z24.b, p5/m, za0h.b[w14, 3]
// CHECK-ENCODING: [0x78,0x54,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0025478 <unknown>

mov     z0.b, p6/m, za0h.b[w12, 12]
// CHECK-INST: mov     z0.b, p6/m, za0h.b[w12, 12]
// CHECK-ENCODING: [0x80,0x19,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0021980 <unknown>

mov     z17.b, p2/m, za0h.b[w14, 1]
// CHECK-INST: mov     z17.b, p2/m, za0h.b[w14, 1]
// CHECK-ENCODING: [0x31,0x48,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0024831 <unknown>

mov     z29.b, p2/m, za0h.b[w12, 6]
// CHECK-INST: mov     z29.b, p2/m, za0h.b[w12, 6]
// CHECK-ENCODING: [0xdd,0x08,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c00208dd <unknown>

mov     z2.b, p5/m, za0h.b[w15, 9]
// CHECK-INST: mov     z2.b, p5/m, za0h.b[w15, 9]
// CHECK-ENCODING: [0x22,0x75,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0027522 <unknown>

mov     z7.b, p2/m, za0h.b[w13, 12]
// CHECK-INST: mov     z7.b, p2/m, za0h.b[w13, 12]
// CHECK-ENCODING: [0x87,0x29,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0022987 <unknown>

// --------------------------------------------------------------------------//
// Extract, tile to vector, vertical, 8-bit

mova    z0.b, p0/m, za0v.b[w12, 0]
// CHECK-INST: mov     z0.b, p0/m, za0v.b[w12, 0]
// CHECK-ENCODING: [0x00,0x80,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0028000 <unknown>

mova    z21.b, p5/m, za0v.b[w14, 10]
// CHECK-INST: mov     z21.b, p5/m, za0v.b[w14, 10]
// CHECK-ENCODING: [0x55,0xd5,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c002d555 <unknown>

mova    z23.b, p3/m, za0v.b[w15, 13]
// CHECK-INST: mov     z23.b, p3/m, za0v.b[w15, 13]
// CHECK-ENCODING: [0xb7,0xed,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c002edb7 <unknown>

mova    z31.b, p7/m, za0v.b[w15, 15]
// CHECK-INST: mov     z31.b, p7/m, za0v.b[w15, 15]
// CHECK-ENCODING: [0xff,0xfd,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c002fdff <unknown>

mova    z5.b, p3/m, za0v.b[w12, 1]
// CHECK-INST: mov     z5.b, p3/m, za0v.b[w12, 1]
// CHECK-ENCODING: [0x25,0x8c,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0028c25 <unknown>

mova    z1.b, p1/m, za0v.b[w12, 1]
// CHECK-INST: mov     z1.b, p1/m, za0v.b[w12, 1]
// CHECK-ENCODING: [0x21,0x84,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0028421 <unknown>

mova    z24.b, p5/m, za0v.b[w14, 3]
// CHECK-INST: mov     z24.b, p5/m, za0v.b[w14, 3]
// CHECK-ENCODING: [0x78,0xd4,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c002d478 <unknown>

mova    z0.b, p6/m, za0v.b[w12, 12]
// CHECK-INST: mov     z0.b, p6/m, za0v.b[w12, 12]
// CHECK-ENCODING: [0x80,0x99,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0029980 <unknown>

mova    z17.b, p2/m, za0v.b[w14, 1]
// CHECK-INST: mov     z17.b, p2/m, za0v.b[w14, 1]
// CHECK-ENCODING: [0x31,0xc8,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c002c831 <unknown>

mova    z29.b, p2/m, za0v.b[w12, 6]
// CHECK-INST: mov     z29.b, p2/m, za0v.b[w12, 6]
// CHECK-ENCODING: [0xdd,0x88,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c00288dd <unknown>

mova    z2.b, p5/m, za0v.b[w15, 9]
// CHECK-INST: mov     z2.b, p5/m, za0v.b[w15, 9]
// CHECK-ENCODING: [0x22,0xf5,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c002f522 <unknown>

mova    z7.b, p2/m, za0v.b[w13, 12]
// CHECK-INST: mov     z7.b, p2/m, za0v.b[w13, 12]
// CHECK-ENCODING: [0x87,0xa9,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c002a987 <unknown>

// Aliases

mov     z0.b, p0/m, za0v.b[w12, 0]
// CHECK-INST: mov     z0.b, p0/m, za0v.b[w12, 0]
// CHECK-ENCODING: [0x00,0x80,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0028000 <unknown>

mov     z21.b, p5/m, za0v.b[w14, 10]
// CHECK-INST: mov     z21.b, p5/m, za0v.b[w14, 10]
// CHECK-ENCODING: [0x55,0xd5,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c002d555 <unknown>

mov     z23.b, p3/m, za0v.b[w15, 13]
// CHECK-INST: mov     z23.b, p3/m, za0v.b[w15, 13]
// CHECK-ENCODING: [0xb7,0xed,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c002edb7 <unknown>

mov     z31.b, p7/m, za0v.b[w15, 15]
// CHECK-INST: mov     z31.b, p7/m, za0v.b[w15, 15]
// CHECK-ENCODING: [0xff,0xfd,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c002fdff <unknown>

mov     z5.b, p3/m, za0v.b[w12, 1]
// CHECK-INST: mov     z5.b, p3/m, za0v.b[w12, 1]
// CHECK-ENCODING: [0x25,0x8c,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0028c25 <unknown>

mov     z1.b, p1/m, za0v.b[w12, 1]
// CHECK-INST: mov     z1.b, p1/m, za0v.b[w12, 1]
// CHECK-ENCODING: [0x21,0x84,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0028421 <unknown>

mov     z24.b, p5/m, za0v.b[w14, 3]
// CHECK-INST: mov     z24.b, p5/m, za0v.b[w14, 3]
// CHECK-ENCODING: [0x78,0xd4,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c002d478 <unknown>

mov     z0.b, p6/m, za0v.b[w12, 12]
// CHECK-INST: mov     z0.b, p6/m, za0v.b[w12, 12]
// CHECK-ENCODING: [0x80,0x99,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0029980 <unknown>

mov     z17.b, p2/m, za0v.b[w14, 1]
// CHECK-INST: mov     z17.b, p2/m, za0v.b[w14, 1]
// CHECK-ENCODING: [0x31,0xc8,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c002c831 <unknown>

mov     z29.b, p2/m, za0v.b[w12, 6]
// CHECK-INST: mov     z29.b, p2/m, za0v.b[w12, 6]
// CHECK-ENCODING: [0xdd,0x88,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c00288dd <unknown>

mov     z2.b, p5/m, za0v.b[w15, 9]
// CHECK-INST: mov     z2.b, p5/m, za0v.b[w15, 9]
// CHECK-ENCODING: [0x22,0xf5,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c002f522 <unknown>

mov     z7.b, p2/m, za0v.b[w13, 12]
// CHECK-INST: mov     z7.b, p2/m, za0v.b[w13, 12]
// CHECK-ENCODING: [0x87,0xa9,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c002a987 <unknown>

// --------------------------------------------------------------------------//
// Extract, tile to vector, horizontal, 16-bit

mova    z0.h, p0/m, za0h.h[w12, 0]
// CHECK-INST: mov     z0.h, p0/m, za0h.h[w12, 0]
// CHECK-ENCODING: [0x00,0x00,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0420000 <unknown>

mova    z21.h, p5/m, za1h.h[w14, 2]
// CHECK-INST: mov     z21.h, p5/m, za1h.h[w14, 2]
// CHECK-ENCODING: [0x55,0x55,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0425555 <unknown>

mova    z23.h, p3/m, za1h.h[w15, 5]
// CHECK-INST: mov     z23.h, p3/m, za1h.h[w15, 5]
// CHECK-ENCODING: [0xb7,0x6d,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0426db7 <unknown>

mova    z31.h, p7/m, za1h.h[w15, 7]
// CHECK-INST: mov     z31.h, p7/m, za1h.h[w15, 7]
// CHECK-ENCODING: [0xff,0x7d,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0427dff <unknown>

mova    z5.h, p3/m, za0h.h[w12, 1]
// CHECK-INST: mov     z5.h, p3/m, za0h.h[w12, 1]
// CHECK-ENCODING: [0x25,0x0c,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0420c25 <unknown>

mova    z1.h, p1/m, za0h.h[w12, 1]
// CHECK-INST: mov     z1.h, p1/m, za0h.h[w12, 1]
// CHECK-ENCODING: [0x21,0x04,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0420421 <unknown>

mova    z24.h, p5/m, za0h.h[w14, 3]
// CHECK-INST: mov     z24.h, p5/m, za0h.h[w14, 3]
// CHECK-ENCODING: [0x78,0x54,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0425478 <unknown>

mova    z0.h, p6/m, za1h.h[w12, 4]
// CHECK-INST: mov     z0.h, p6/m, za1h.h[w12, 4]
// CHECK-ENCODING: [0x80,0x19,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0421980 <unknown>

mova    z17.h, p2/m, za0h.h[w14, 1]
// CHECK-INST: mov     z17.h, p2/m, za0h.h[w14, 1]
// CHECK-ENCODING: [0x31,0x48,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0424831 <unknown>

mova    z29.h, p2/m, za0h.h[w12, 6]
// CHECK-INST: mov     z29.h, p2/m, za0h.h[w12, 6]
// CHECK-ENCODING: [0xdd,0x08,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c04208dd <unknown>

mova    z2.h, p5/m, za1h.h[w15, 1]
// CHECK-INST: mov     z2.h, p5/m, za1h.h[w15, 1]
// CHECK-ENCODING: [0x22,0x75,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0427522 <unknown>

mova    z7.h, p2/m, za1h.h[w13, 4]
// CHECK-INST: mov     z7.h, p2/m, za1h.h[w13, 4]
// CHECK-ENCODING: [0x87,0x29,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0422987 <unknown>

// Aliases

mov     z0.h, p0/m, za0h.h[w12, 0]
// CHECK-INST: mov     z0.h, p0/m, za0h.h[w12, 0]
// CHECK-ENCODING: [0x00,0x00,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0420000 <unknown>

mov     z21.h, p5/m, za1h.h[w14, 2]
// CHECK-INST: mov     z21.h, p5/m, za1h.h[w14, 2]
// CHECK-ENCODING: [0x55,0x55,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0425555 <unknown>

mov     z23.h, p3/m, za1h.h[w15, 5]
// CHECK-INST: mov     z23.h, p3/m, za1h.h[w15, 5]
// CHECK-ENCODING: [0xb7,0x6d,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0426db7 <unknown>

mov     z31.h, p7/m, za1h.h[w15, 7]
// CHECK-INST: mov     z31.h, p7/m, za1h.h[w15, 7]
// CHECK-ENCODING: [0xff,0x7d,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0427dff <unknown>

mov     z5.h, p3/m, za0h.h[w12, 1]
// CHECK-INST: mov     z5.h, p3/m, za0h.h[w12, 1]
// CHECK-ENCODING: [0x25,0x0c,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0420c25 <unknown>

mov     z1.h, p1/m, za0h.h[w12, 1]
// CHECK-INST: mov     z1.h, p1/m, za0h.h[w12, 1]
// CHECK-ENCODING: [0x21,0x04,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0420421 <unknown>

mov     z24.h, p5/m, za0h.h[w14, 3]
// CHECK-INST: mov     z24.h, p5/m, za0h.h[w14, 3]
// CHECK-ENCODING: [0x78,0x54,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0425478 <unknown>

mov     z0.h, p6/m, za1h.h[w12, 4]
// CHECK-INST: mov     z0.h, p6/m, za1h.h[w12, 4]
// CHECK-ENCODING: [0x80,0x19,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0421980 <unknown>

mov     z17.h, p2/m, za0h.h[w14, 1]
// CHECK-INST: mov     z17.h, p2/m, za0h.h[w14, 1]
// CHECK-ENCODING: [0x31,0x48,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0424831 <unknown>

mov     z29.h, p2/m, za0h.h[w12, 6]
// CHECK-INST: mov     z29.h, p2/m, za0h.h[w12, 6]
// CHECK-ENCODING: [0xdd,0x08,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c04208dd <unknown>

mov     z2.h, p5/m, za1h.h[w15, 1]
// CHECK-INST: mov     z2.h, p5/m, za1h.h[w15, 1]
// CHECK-ENCODING: [0x22,0x75,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0427522 <unknown>

mov     z7.h, p2/m, za1h.h[w13, 4]
// CHECK-INST: mov     z7.h, p2/m, za1h.h[w13, 4]
// CHECK-ENCODING: [0x87,0x29,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0422987 <unknown>

// --------------------------------------------------------------------------//
// Extract, tile to vector, vertical, 16-bit

mova    z0.h, p0/m, za0v.h[w12, 0]
// CHECK-INST: mov     z0.h, p0/m, za0v.h[w12, 0]
// CHECK-ENCODING: [0x00,0x80,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0428000 <unknown>

mova    z21.h, p5/m, za1v.h[w14, 2]
// CHECK-INST: mov     z21.h, p5/m, za1v.h[w14, 2]
// CHECK-ENCODING: [0x55,0xd5,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c042d555 <unknown>

mova    z23.h, p3/m, za1v.h[w15, 5]
// CHECK-INST: mov     z23.h, p3/m, za1v.h[w15, 5]
// CHECK-ENCODING: [0xb7,0xed,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c042edb7 <unknown>

mova    z31.h, p7/m, za1v.h[w15, 7]
// CHECK-INST: mov     z31.h, p7/m, za1v.h[w15, 7]
// CHECK-ENCODING: [0xff,0xfd,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c042fdff <unknown>

mova    z5.h, p3/m, za0v.h[w12, 1]
// CHECK-INST: mov     z5.h, p3/m, za0v.h[w12, 1]
// CHECK-ENCODING: [0x25,0x8c,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0428c25 <unknown>

mova    z1.h, p1/m, za0v.h[w12, 1]
// CHECK-INST: mov     z1.h, p1/m, za0v.h[w12, 1]
// CHECK-ENCODING: [0x21,0x84,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0428421 <unknown>

mova    z24.h, p5/m, za0v.h[w14, 3]
// CHECK-INST: mov     z24.h, p5/m, za0v.h[w14, 3]
// CHECK-ENCODING: [0x78,0xd4,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c042d478 <unknown>

mova    z0.h, p6/m, za1v.h[w12, 4]
// CHECK-INST: mov     z0.h, p6/m, za1v.h[w12, 4]
// CHECK-ENCODING: [0x80,0x99,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0429980 <unknown>

mova    z17.h, p2/m, za0v.h[w14, 1]
// CHECK-INST: mov     z17.h, p2/m, za0v.h[w14, 1]
// CHECK-ENCODING: [0x31,0xc8,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c042c831 <unknown>

mova    z29.h, p2/m, za0v.h[w12, 6]
// CHECK-INST: mov     z29.h, p2/m, za0v.h[w12, 6]
// CHECK-ENCODING: [0xdd,0x88,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c04288dd <unknown>

mova    z2.h, p5/m, za1v.h[w15, 1]
// CHECK-INST: mov     z2.h, p5/m, za1v.h[w15, 1]
// CHECK-ENCODING: [0x22,0xf5,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c042f522 <unknown>

mova    z7.h, p2/m, za1v.h[w13, 4]
// CHECK-INST: mov     z7.h, p2/m, za1v.h[w13, 4]
// CHECK-ENCODING: [0x87,0xa9,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c042a987 <unknown>

// Aliases

mov     z0.h, p0/m, za0v.h[w12, 0]
// CHECK-INST: mov     z0.h, p0/m, za0v.h[w12, 0]
// CHECK-ENCODING: [0x00,0x80,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0428000 <unknown>

mov     z21.h, p5/m, za1v.h[w14, 2]
// CHECK-INST: mov     z21.h, p5/m, za1v.h[w14, 2]
// CHECK-ENCODING: [0x55,0xd5,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c042d555 <unknown>

mov     z23.h, p3/m, za1v.h[w15, 5]
// CHECK-INST: mov     z23.h, p3/m, za1v.h[w15, 5]
// CHECK-ENCODING: [0xb7,0xed,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c042edb7 <unknown>

mov     z31.h, p7/m, za1v.h[w15, 7]
// CHECK-INST: mov     z31.h, p7/m, za1v.h[w15, 7]
// CHECK-ENCODING: [0xff,0xfd,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c042fdff <unknown>

mov     z5.h, p3/m, za0v.h[w12, 1]
// CHECK-INST: mov     z5.h, p3/m, za0v.h[w12, 1]
// CHECK-ENCODING: [0x25,0x8c,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0428c25 <unknown>

mov     z1.h, p1/m, za0v.h[w12, 1]
// CHECK-INST: mov     z1.h, p1/m, za0v.h[w12, 1]
// CHECK-ENCODING: [0x21,0x84,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0428421 <unknown>

mov     z24.h, p5/m, za0v.h[w14, 3]
// CHECK-INST: mov     z24.h, p5/m, za0v.h[w14, 3]
// CHECK-ENCODING: [0x78,0xd4,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c042d478 <unknown>

mov     z0.h, p6/m, za1v.h[w12, 4]
// CHECK-INST: mov     z0.h, p6/m, za1v.h[w12, 4]
// CHECK-ENCODING: [0x80,0x99,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0429980 <unknown>

mov     z17.h, p2/m, za0v.h[w14, 1]
// CHECK-INST: mov     z17.h, p2/m, za0v.h[w14, 1]
// CHECK-ENCODING: [0x31,0xc8,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c042c831 <unknown>

mov     z29.h, p2/m, za0v.h[w12, 6]
// CHECK-INST: mov     z29.h, p2/m, za0v.h[w12, 6]
// CHECK-ENCODING: [0xdd,0x88,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c04288dd <unknown>

mov     z2.h, p5/m, za1v.h[w15, 1]
// CHECK-INST: mov     z2.h, p5/m, za1v.h[w15, 1]
// CHECK-ENCODING: [0x22,0xf5,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c042f522 <unknown>

mov     z7.h, p2/m, za1v.h[w13, 4]
// CHECK-INST: mov     z7.h, p2/m, za1v.h[w13, 4]
// CHECK-ENCODING: [0x87,0xa9,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c042a987 <unknown>

// --------------------------------------------------------------------------//
// Extract, tile to vector, horizontal, 32-bit

mova    z0.s, p0/m, za0h.s[w12, 0]
// CHECK-INST: mov     z0.s, p0/m, za0h.s[w12, 0]
// CHECK-ENCODING: [0x00,0x00,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0820000 <unknown>

mova    z21.s, p5/m, za2h.s[w14, 2]
// CHECK-INST: mov     z21.s, p5/m, za2h.s[w14, 2]
// CHECK-ENCODING: [0x55,0x55,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0825555 <unknown>

mova    z23.s, p3/m, za3h.s[w15, 1]
// CHECK-INST: mov     z23.s, p3/m, za3h.s[w15, 1]
// CHECK-ENCODING: [0xb7,0x6d,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0826db7 <unknown>

mova    z31.s, p7/m, za3h.s[w15, 3]
// CHECK-INST: mov     z31.s, p7/m, za3h.s[w15, 3]
// CHECK-ENCODING: [0xff,0x7d,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0827dff <unknown>

mova    z5.s, p3/m, za0h.s[w12, 1]
// CHECK-INST: mov     z5.s, p3/m, za0h.s[w12, 1]
// CHECK-ENCODING: [0x25,0x0c,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0820c25 <unknown>

mova    z1.s, p1/m, za0h.s[w12, 1]
// CHECK-INST: mov     z1.s, p1/m, za0h.s[w12, 1]
// CHECK-ENCODING: [0x21,0x04,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0820421 <unknown>

mova    z24.s, p5/m, za0h.s[w14, 3]
// CHECK-INST: mov     z24.s, p5/m, za0h.s[w14, 3]
// CHECK-ENCODING: [0x78,0x54,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0825478 <unknown>

mova    z0.s, p6/m, za3h.s[w12, 0]
// CHECK-INST: mov     z0.s, p6/m, za3h.s[w12, 0]
// CHECK-ENCODING: [0x80,0x19,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0821980 <unknown>

mova    z17.s, p2/m, za0h.s[w14, 1]
// CHECK-INST: mov     z17.s, p2/m, za0h.s[w14, 1]
// CHECK-ENCODING: [0x31,0x48,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0824831 <unknown>

mova    z29.s, p2/m, za1h.s[w12, 2]
// CHECK-INST: mov     z29.s, p2/m, za1h.s[w12, 2]
// CHECK-ENCODING: [0xdd,0x08,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c08208dd <unknown>

mova    z2.s, p5/m, za2h.s[w15, 1]
// CHECK-INST: mov     z2.s, p5/m, za2h.s[w15, 1]
// CHECK-ENCODING: [0x22,0x75,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0827522 <unknown>

mova    z7.s, p2/m, za3h.s[w13, 0]
// CHECK-INST: mov     z7.s, p2/m, za3h.s[w13, 0]
// CHECK-ENCODING: [0x87,0x29,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0822987 <unknown>

// Aliases

mov     z0.s, p0/m, za0h.s[w12, 0]
// CHECK-INST: mov     z0.s, p0/m, za0h.s[w12, 0]
// CHECK-ENCODING: [0x00,0x00,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0820000 <unknown>

mov     z21.s, p5/m, za2h.s[w14, 2]
// CHECK-INST: mov     z21.s, p5/m, za2h.s[w14, 2]
// CHECK-ENCODING: [0x55,0x55,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0825555 <unknown>

mov     z23.s, p3/m, za3h.s[w15, 1]
// CHECK-INST: mov     z23.s, p3/m, za3h.s[w15, 1]
// CHECK-ENCODING: [0xb7,0x6d,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0826db7 <unknown>

mov     z31.s, p7/m, za3h.s[w15, 3]
// CHECK-INST: mov     z31.s, p7/m, za3h.s[w15, 3]
// CHECK-ENCODING: [0xff,0x7d,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0827dff <unknown>

mov     z5.s, p3/m, za0h.s[w12, 1]
// CHECK-INST: mov     z5.s, p3/m, za0h.s[w12, 1]
// CHECK-ENCODING: [0x25,0x0c,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0820c25 <unknown>

mov     z1.s, p1/m, za0h.s[w12, 1]
// CHECK-INST: mov     z1.s, p1/m, za0h.s[w12, 1]
// CHECK-ENCODING: [0x21,0x04,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0820421 <unknown>

mov     z24.s, p5/m, za0h.s[w14, 3]
// CHECK-INST: mov     z24.s, p5/m, za0h.s[w14, 3]
// CHECK-ENCODING: [0x78,0x54,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0825478 <unknown>

mov     z0.s, p6/m, za3h.s[w12, 0]
// CHECK-INST: mov     z0.s, p6/m, za3h.s[w12, 0]
// CHECK-ENCODING: [0x80,0x19,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0821980 <unknown>

mov     z17.s, p2/m, za0h.s[w14, 1]
// CHECK-INST: mov     z17.s, p2/m, za0h.s[w14, 1]
// CHECK-ENCODING: [0x31,0x48,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0824831 <unknown>

mov     z29.s, p2/m, za1h.s[w12, 2]
// CHECK-INST: mov     z29.s, p2/m, za1h.s[w12, 2]
// CHECK-ENCODING: [0xdd,0x08,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c08208dd <unknown>

mov     z2.s, p5/m, za2h.s[w15, 1]
// CHECK-INST: mov     z2.s, p5/m, za2h.s[w15, 1]
// CHECK-ENCODING: [0x22,0x75,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0827522 <unknown>

mov     z7.s, p2/m, za3h.s[w13, 0]
// CHECK-INST: mov     z7.s, p2/m, za3h.s[w13, 0]
// CHECK-ENCODING: [0x87,0x29,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0822987 <unknown>

// --------------------------------------------------------------------------//
// Extract, tile to vector, vertical, 32-bit

mova    z0.s, p0/m, za0v.s[w12, 0]
// CHECK-INST: mov     z0.s, p0/m, za0v.s[w12, 0]
// CHECK-ENCODING: [0x00,0x80,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0828000 <unknown>

mova    z21.s, p5/m, za2v.s[w14, 2]
// CHECK-INST: mov     z21.s, p5/m, za2v.s[w14, 2]
// CHECK-ENCODING: [0x55,0xd5,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c082d555 <unknown>

mova    z23.s, p3/m, za3v.s[w15, 1]
// CHECK-INST: mov     z23.s, p3/m, za3v.s[w15, 1]
// CHECK-ENCODING: [0xb7,0xed,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c082edb7 <unknown>

mova    z31.s, p7/m, za3v.s[w15, 3]
// CHECK-INST: mov     z31.s, p7/m, za3v.s[w15, 3]
// CHECK-ENCODING: [0xff,0xfd,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c082fdff <unknown>

mova    z5.s, p3/m, za0v.s[w12, 1]
// CHECK-INST: mov     z5.s, p3/m, za0v.s[w12, 1]
// CHECK-ENCODING: [0x25,0x8c,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0828c25 <unknown>

mova    z1.s, p1/m, za0v.s[w12, 1]
// CHECK-INST: mov     z1.s, p1/m, za0v.s[w12, 1]
// CHECK-ENCODING: [0x21,0x84,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0828421 <unknown>

mova    z24.s, p5/m, za0v.s[w14, 3]
// CHECK-INST: mov     z24.s, p5/m, za0v.s[w14, 3]
// CHECK-ENCODING: [0x78,0xd4,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c082d478 <unknown>

mova    z0.s, p6/m, za3v.s[w12, 0]
// CHECK-INST: mov     z0.s, p6/m, za3v.s[w12, 0]
// CHECK-ENCODING: [0x80,0x99,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0829980 <unknown>

mova    z17.s, p2/m, za0v.s[w14, 1]
// CHECK-INST: mov     z17.s, p2/m, za0v.s[w14, 1]
// CHECK-ENCODING: [0x31,0xc8,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c082c831 <unknown>

mova    z29.s, p2/m, za1v.s[w12, 2]
// CHECK-INST: mov     z29.s, p2/m, za1v.s[w12, 2]
// CHECK-ENCODING: [0xdd,0x88,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c08288dd <unknown>

mova    z2.s, p5/m, za2v.s[w15, 1]
// CHECK-INST: mov     z2.s, p5/m, za2v.s[w15, 1]
// CHECK-ENCODING: [0x22,0xf5,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c082f522 <unknown>

mova    z7.s, p2/m, za3v.s[w13, 0]
// CHECK-INST: mov     z7.s, p2/m, za3v.s[w13, 0]
// CHECK-ENCODING: [0x87,0xa9,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c082a987 <unknown>

// Aliases

mov     z0.s, p0/m, za0v.s[w12, 0]
// CHECK-INST: mov     z0.s, p0/m, za0v.s[w12, 0]
// CHECK-ENCODING: [0x00,0x80,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0828000 <unknown>

mov     z21.s, p5/m, za2v.s[w14, 2]
// CHECK-INST: mov     z21.s, p5/m, za2v.s[w14, 2]
// CHECK-ENCODING: [0x55,0xd5,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c082d555 <unknown>

mov     z23.s, p3/m, za3v.s[w15, 1]
// CHECK-INST: mov     z23.s, p3/m, za3v.s[w15, 1]
// CHECK-ENCODING: [0xb7,0xed,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c082edb7 <unknown>

mov     z31.s, p7/m, za3v.s[w15, 3]
// CHECK-INST: mov     z31.s, p7/m, za3v.s[w15, 3]
// CHECK-ENCODING: [0xff,0xfd,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c082fdff <unknown>

mov     z5.s, p3/m, za0v.s[w12, 1]
// CHECK-INST: mov     z5.s, p3/m, za0v.s[w12, 1]
// CHECK-ENCODING: [0x25,0x8c,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0828c25 <unknown>

mov     z1.s, p1/m, za0v.s[w12, 1]
// CHECK-INST: mov     z1.s, p1/m, za0v.s[w12, 1]
// CHECK-ENCODING: [0x21,0x84,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0828421 <unknown>

mov     z24.s, p5/m, za0v.s[w14, 3]
// CHECK-INST: mov     z24.s, p5/m, za0v.s[w14, 3]
// CHECK-ENCODING: [0x78,0xd4,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c082d478 <unknown>

mov     z0.s, p6/m, za3v.s[w12, 0]
// CHECK-INST: mov     z0.s, p6/m, za3v.s[w12, 0]
// CHECK-ENCODING: [0x80,0x99,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0829980 <unknown>

mov     z17.s, p2/m, za0v.s[w14, 1]
// CHECK-INST: mov     z17.s, p2/m, za0v.s[w14, 1]
// CHECK-ENCODING: [0x31,0xc8,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c082c831 <unknown>

mov     z29.s, p2/m, za1v.s[w12, 2]
// CHECK-INST: mov     z29.s, p2/m, za1v.s[w12, 2]
// CHECK-ENCODING: [0xdd,0x88,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c08288dd <unknown>

mov     z2.s, p5/m, za2v.s[w15, 1]
// CHECK-INST: mov     z2.s, p5/m, za2v.s[w15, 1]
// CHECK-ENCODING: [0x22,0xf5,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c082f522 <unknown>

mov     z7.s, p2/m, za3v.s[w13, 0]
// CHECK-INST: mov     z7.s, p2/m, za3v.s[w13, 0]
// CHECK-ENCODING: [0x87,0xa9,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c082a987 <unknown>

// --------------------------------------------------------------------------//
// Extract, tile to vector, horizontal, 64-bit

mova    z0.d, p0/m, za0h.d[w12, 0]
// CHECK-INST: mov     z0.d, p0/m, za0h.d[w12, 0]
// CHECK-ENCODING: [0x00,0x00,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c20000 <unknown>

mova    z21.d, p5/m, za5h.d[w14, 0]
// CHECK-INST: mov     z21.d, p5/m, za5h.d[w14, 0]
// CHECK-ENCODING: [0x55,0x55,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c25555 <unknown>

mova    z23.d, p3/m, za6h.d[w15, 1]
// CHECK-INST: mov     z23.d, p3/m, za6h.d[w15, 1]
// CHECK-ENCODING: [0xb7,0x6d,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c26db7 <unknown>

mova    z31.d, p7/m, za7h.d[w15, 1]
// CHECK-INST: mov     z31.d, p7/m, za7h.d[w15, 1]
// CHECK-ENCODING: [0xff,0x7d,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c27dff <unknown>

mova    z5.d, p3/m, za0h.d[w12, 1]
// CHECK-INST: mov     z5.d, p3/m, za0h.d[w12, 1]
// CHECK-ENCODING: [0x25,0x0c,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c20c25 <unknown>

mova    z1.d, p1/m, za0h.d[w12, 1]
// CHECK-INST: mov     z1.d, p1/m, za0h.d[w12, 1]
// CHECK-ENCODING: [0x21,0x04,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c20421 <unknown>

mova    z24.d, p5/m, za1h.d[w14, 1]
// CHECK-INST: mov     z24.d, p5/m, za1h.d[w14, 1]
// CHECK-ENCODING: [0x78,0x54,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c25478 <unknown>

mova    z0.d, p6/m, za6h.d[w12, 0]
// CHECK-INST: mov     z0.d, p6/m, za6h.d[w12, 0]
// CHECK-ENCODING: [0x80,0x19,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c21980 <unknown>

mova    z17.d, p2/m, za0h.d[w14, 1]
// CHECK-INST: mov     z17.d, p2/m, za0h.d[w14, 1]
// CHECK-ENCODING: [0x31,0x48,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c24831 <unknown>

mova    z29.d, p2/m, za3h.d[w12, 0]
// CHECK-INST: mov     z29.d, p2/m, za3h.d[w12, 0]
// CHECK-ENCODING: [0xdd,0x08,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c208dd <unknown>

mova    z2.d, p5/m, za4h.d[w15, 1]
// CHECK-INST: mov     z2.d, p5/m, za4h.d[w15, 1]
// CHECK-ENCODING: [0x22,0x75,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c27522 <unknown>

mova    z7.d, p2/m, za6h.d[w13, 0]
// CHECK-INST: mov     z7.d, p2/m, za6h.d[w13, 0]
// CHECK-ENCODING: [0x87,0x29,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c22987 <unknown>

// Aliases

mov     z0.d, p0/m, za0h.d[w12, 0]
// CHECK-INST: mov     z0.d, p0/m, za0h.d[w12, 0]
// CHECK-ENCODING: [0x00,0x00,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c20000 <unknown>

mov     z21.d, p5/m, za5h.d[w14, 0]
// CHECK-INST: mov     z21.d, p5/m, za5h.d[w14, 0]
// CHECK-ENCODING: [0x55,0x55,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c25555 <unknown>

mov     z23.d, p3/m, za6h.d[w15, 1]
// CHECK-INST: mov     z23.d, p3/m, za6h.d[w15, 1]
// CHECK-ENCODING: [0xb7,0x6d,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c26db7 <unknown>

mov     z31.d, p7/m, za7h.d[w15, 1]
// CHECK-INST: mov     z31.d, p7/m, za7h.d[w15, 1]
// CHECK-ENCODING: [0xff,0x7d,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c27dff <unknown>

mov     z5.d, p3/m, za0h.d[w12, 1]
// CHECK-INST: mov     z5.d, p3/m, za0h.d[w12, 1]
// CHECK-ENCODING: [0x25,0x0c,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c20c25 <unknown>

mov     z1.d, p1/m, za0h.d[w12, 1]
// CHECK-INST: mov     z1.d, p1/m, za0h.d[w12, 1]
// CHECK-ENCODING: [0x21,0x04,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c20421 <unknown>

mov     z24.d, p5/m, za1h.d[w14, 1]
// CHECK-INST: mov     z24.d, p5/m, za1h.d[w14, 1]
// CHECK-ENCODING: [0x78,0x54,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c25478 <unknown>

mov     z0.d, p6/m, za6h.d[w12, 0]
// CHECK-INST: mov     z0.d, p6/m, za6h.d[w12, 0]
// CHECK-ENCODING: [0x80,0x19,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c21980 <unknown>

mov     z17.d, p2/m, za0h.d[w14, 1]
// CHECK-INST: mov     z17.d, p2/m, za0h.d[w14, 1]
// CHECK-ENCODING: [0x31,0x48,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c24831 <unknown>

mov     z29.d, p2/m, za3h.d[w12, 0]
// CHECK-INST: mov     z29.d, p2/m, za3h.d[w12, 0]
// CHECK-ENCODING: [0xdd,0x08,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c208dd <unknown>

mov     z2.d, p5/m, za4h.d[w15, 1]
// CHECK-INST: mov     z2.d, p5/m, za4h.d[w15, 1]
// CHECK-ENCODING: [0x22,0x75,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c27522 <unknown>

mov     z7.d, p2/m, za6h.d[w13, 0]
// CHECK-INST: mov     z7.d, p2/m, za6h.d[w13, 0]
// CHECK-ENCODING: [0x87,0x29,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c22987 <unknown>

// --------------------------------------------------------------------------//
// Extract, tile to vector, vertical, 64-bit

mova    z0.d, p0/m, za0v.d[w12, 0]
// CHECK-INST: mov     z0.d, p0/m, za0v.d[w12, 0]
// CHECK-ENCODING: [0x00,0x80,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c28000 <unknown>

mova    z21.d, p5/m, za5v.d[w14, 0]
// CHECK-INST: mov     z21.d, p5/m, za5v.d[w14, 0]
// CHECK-ENCODING: [0x55,0xd5,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c2d555 <unknown>

mova    z23.d, p3/m, za6v.d[w15, 1]
// CHECK-INST: mov     z23.d, p3/m, za6v.d[w15, 1]
// CHECK-ENCODING: [0xb7,0xed,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c2edb7 <unknown>

mova    z31.d, p7/m, za7v.d[w15, 1]
// CHECK-INST: mov     z31.d, p7/m, za7v.d[w15, 1]
// CHECK-ENCODING: [0xff,0xfd,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c2fdff <unknown>

mova    z5.d, p3/m, za0v.d[w12, 1]
// CHECK-INST: mov     z5.d, p3/m, za0v.d[w12, 1]
// CHECK-ENCODING: [0x25,0x8c,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c28c25 <unknown>

mova    z1.d, p1/m, za0v.d[w12, 1]
// CHECK-INST: mov     z1.d, p1/m, za0v.d[w12, 1]
// CHECK-ENCODING: [0x21,0x84,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c28421 <unknown>

mova    z24.d, p5/m, za1v.d[w14, 1]
// CHECK-INST: mov     z24.d, p5/m, za1v.d[w14, 1]
// CHECK-ENCODING: [0x78,0xd4,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c2d478 <unknown>

mova    z0.d, p6/m, za6v.d[w12, 0]
// CHECK-INST: mov     z0.d, p6/m, za6v.d[w12, 0]
// CHECK-ENCODING: [0x80,0x99,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c29980 <unknown>

mova    z17.d, p2/m, za0v.d[w14, 1]
// CHECK-INST: mov     z17.d, p2/m, za0v.d[w14, 1]
// CHECK-ENCODING: [0x31,0xc8,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c2c831 <unknown>

mova    z29.d, p2/m, za3v.d[w12, 0]
// CHECK-INST: mov     z29.d, p2/m, za3v.d[w12, 0]
// CHECK-ENCODING: [0xdd,0x88,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c288dd <unknown>

mova    z2.d, p5/m, za4v.d[w15, 1]
// CHECK-INST: mov     z2.d, p5/m, za4v.d[w15, 1]
// CHECK-ENCODING: [0x22,0xf5,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c2f522 <unknown>

mova    z7.d, p2/m, za6v.d[w13, 0]
// CHECK-INST: mov     z7.d, p2/m, za6v.d[w13, 0]
// CHECK-ENCODING: [0x87,0xa9,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c2a987 <unknown>

// Aliases

mov     z0.d, p0/m, za0v.d[w12, 0]
// CHECK-INST: mov     z0.d, p0/m, za0v.d[w12, 0]
// CHECK-ENCODING: [0x00,0x80,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c28000 <unknown>

mov     z21.d, p5/m, za5v.d[w14, 0]
// CHECK-INST: mov     z21.d, p5/m, za5v.d[w14, 0]
// CHECK-ENCODING: [0x55,0xd5,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c2d555 <unknown>

mov     z23.d, p3/m, za6v.d[w15, 1]
// CHECK-INST: mov     z23.d, p3/m, za6v.d[w15, 1]
// CHECK-ENCODING: [0xb7,0xed,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c2edb7 <unknown>

mov     z31.d, p7/m, za7v.d[w15, 1]
// CHECK-INST: mov     z31.d, p7/m, za7v.d[w15, 1]
// CHECK-ENCODING: [0xff,0xfd,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c2fdff <unknown>

mov     z5.d, p3/m, za0v.d[w12, 1]
// CHECK-INST: mov     z5.d, p3/m, za0v.d[w12, 1]
// CHECK-ENCODING: [0x25,0x8c,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c28c25 <unknown>

mov     z1.d, p1/m, za0v.d[w12, 1]
// CHECK-INST: mov     z1.d, p1/m, za0v.d[w12, 1]
// CHECK-ENCODING: [0x21,0x84,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c28421 <unknown>

mov     z24.d, p5/m, za1v.d[w14, 1]
// CHECK-INST: mov     z24.d, p5/m, za1v.d[w14, 1]
// CHECK-ENCODING: [0x78,0xd4,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c2d478 <unknown>

mov     z0.d, p6/m, za6v.d[w12, 0]
// CHECK-INST: mov     z0.d, p6/m, za6v.d[w12, 0]
// CHECK-ENCODING: [0x80,0x99,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c29980 <unknown>

mov     z17.d, p2/m, za0v.d[w14, 1]
// CHECK-INST: mov     z17.d, p2/m, za0v.d[w14, 1]
// CHECK-ENCODING: [0x31,0xc8,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c2c831 <unknown>

mov     z29.d, p2/m, za3v.d[w12, 0]
// CHECK-INST: mov     z29.d, p2/m, za3v.d[w12, 0]
// CHECK-ENCODING: [0xdd,0x88,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c288dd <unknown>

mov     z2.d, p5/m, za4v.d[w15, 1]
// CHECK-INST: mov     z2.d, p5/m, za4v.d[w15, 1]
// CHECK-ENCODING: [0x22,0xf5,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c2f522 <unknown>

mov     z7.d, p2/m, za6v.d[w13, 0]
// CHECK-INST: mov     z7.d, p2/m, za6v.d[w13, 0]
// CHECK-ENCODING: [0x87,0xa9,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c2a987 <unknown>

// --------------------------------------------------------------------------//
// Extract, tile to vector, horizontal, 128-bit

mova    z0.q, p0/m, za0h.q[w12, 0]
// CHECK-INST: mov     z0.q, p0/m, za0h.q[w12, 0]
// CHECK-ENCODING: [0x00,0x00,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c30000 <unknown>

mova    z21.q, p5/m, za10h.q[w14, 0]
// CHECK-INST: mov     z21.q, p5/m, za10h.q[w14, 0]
// CHECK-ENCODING: [0x55,0x55,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c35555 <unknown>

mova    z23.q, p3/m, za13h.q[w15, 0]
// CHECK-INST: mov     z23.q, p3/m, za13h.q[w15, 0]
// CHECK-ENCODING: [0xb7,0x6d,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c36db7 <unknown>

mova    z31.q, p7/m, za15h.q[w15, 0]
// CHECK-INST: mov     z31.q, p7/m, za15h.q[w15, 0]
// CHECK-ENCODING: [0xff,0x7d,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c37dff <unknown>

mova    z5.q, p3/m, za1h.q[w12, 0]
// CHECK-INST: mov     z5.q, p3/m, za1h.q[w12, 0]
// CHECK-ENCODING: [0x25,0x0c,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c30c25 <unknown>

mova    z1.q, p1/m, za1h.q[w12, 0]
// CHECK-INST: mov     z1.q, p1/m, za1h.q[w12, 0]
// CHECK-ENCODING: [0x21,0x04,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c30421 <unknown>

mova    z24.q, p5/m, za3h.q[w14, 0]
// CHECK-INST: mov     z24.q, p5/m, za3h.q[w14, 0]
// CHECK-ENCODING: [0x78,0x54,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c35478 <unknown>

mova    z0.q, p6/m, za12h.q[w12, 0]
// CHECK-INST: mov     z0.q, p6/m, za12h.q[w12, 0]
// CHECK-ENCODING: [0x80,0x19,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c31980 <unknown>

mova    z17.q, p2/m, za1h.q[w14, 0]
// CHECK-INST: mov     z17.q, p2/m, za1h.q[w14, 0]
// CHECK-ENCODING: [0x31,0x48,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c34831 <unknown>

mova    z29.q, p2/m, za6h.q[w12, 0]
// CHECK-INST: mov     z29.q, p2/m, za6h.q[w12, 0]
// CHECK-ENCODING: [0xdd,0x08,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c308dd <unknown>

mova    z2.q, p5/m, za9h.q[w15, 0]
// CHECK-INST: mov     z2.q, p5/m, za9h.q[w15, 0]
// CHECK-ENCODING: [0x22,0x75,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c37522 <unknown>

mova    z7.q, p2/m, za12h.q[w13, 0]
// CHECK-INST: mov     z7.q, p2/m, za12h.q[w13, 0]
// CHECK-ENCODING: [0x87,0x29,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c32987 <unknown>

// Aliases

mov     z0.q, p0/m, za0h.q[w12, 0]
// CHECK-INST: mov     z0.q, p0/m, za0h.q[w12, 0]
// CHECK-ENCODING: [0x00,0x00,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c30000 <unknown>

mov     z21.q, p5/m, za10h.q[w14, 0]
// CHECK-INST: mov     z21.q, p5/m, za10h.q[w14, 0]
// CHECK-ENCODING: [0x55,0x55,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c35555 <unknown>

mov     z23.q, p3/m, za13h.q[w15, 0]
// CHECK-INST: mov     z23.q, p3/m, za13h.q[w15, 0]
// CHECK-ENCODING: [0xb7,0x6d,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c36db7 <unknown>

mov     z31.q, p7/m, za15h.q[w15, 0]
// CHECK-INST: mov     z31.q, p7/m, za15h.q[w15, 0]
// CHECK-ENCODING: [0xff,0x7d,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c37dff <unknown>

mov     z5.q, p3/m, za1h.q[w12, 0]
// CHECK-INST: mov     z5.q, p3/m, za1h.q[w12, 0]
// CHECK-ENCODING: [0x25,0x0c,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c30c25 <unknown>

mov     z1.q, p1/m, za1h.q[w12, 0]
// CHECK-INST: mov     z1.q, p1/m, za1h.q[w12, 0]
// CHECK-ENCODING: [0x21,0x04,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c30421 <unknown>

mov     z24.q, p5/m, za3h.q[w14, 0]
// CHECK-INST: mov     z24.q, p5/m, za3h.q[w14, 0]
// CHECK-ENCODING: [0x78,0x54,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c35478 <unknown>

mov     z0.q, p6/m, za12h.q[w12, 0]
// CHECK-INST: mov     z0.q, p6/m, za12h.q[w12, 0]
// CHECK-ENCODING: [0x80,0x19,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c31980 <unknown>

mov     z17.q, p2/m, za1h.q[w14, 0]
// CHECK-INST: mov     z17.q, p2/m, za1h.q[w14, 0]
// CHECK-ENCODING: [0x31,0x48,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c34831 <unknown>

mov     z29.q, p2/m, za6h.q[w12, 0]
// CHECK-INST: mov     z29.q, p2/m, za6h.q[w12, 0]
// CHECK-ENCODING: [0xdd,0x08,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c308dd <unknown>

mov     z2.q, p5/m, za9h.q[w15, 0]
// CHECK-INST: mov     z2.q, p5/m, za9h.q[w15, 0]
// CHECK-ENCODING: [0x22,0x75,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c37522 <unknown>

mov     z7.q, p2/m, za12h.q[w13, 0]
// CHECK-INST: mov     z7.q, p2/m, za12h.q[w13, 0]
// CHECK-ENCODING: [0x87,0x29,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c32987 <unknown>

// --------------------------------------------------------------------------//
// Extract, tile to vector, vertical, 128-bit

mova    z0.q, p0/m, za0v.q[w12, 0]
// CHECK-INST: mov     z0.q, p0/m, za0v.q[w12, 0]
// CHECK-ENCODING: [0x00,0x80,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c38000 <unknown>

mova    z21.q, p5/m, za10v.q[w14, 0]
// CHECK-INST: mov     z21.q, p5/m, za10v.q[w14, 0]
// CHECK-ENCODING: [0x55,0xd5,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c3d555 <unknown>

mova    z23.q, p3/m, za13v.q[w15, 0]
// CHECK-INST: mov     z23.q, p3/m, za13v.q[w15, 0]
// CHECK-ENCODING: [0xb7,0xed,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c3edb7 <unknown>

mova    z31.q, p7/m, za15v.q[w15, 0]
// CHECK-INST: mov     z31.q, p7/m, za15v.q[w15, 0]
// CHECK-ENCODING: [0xff,0xfd,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c3fdff <unknown>

mova    z5.q, p3/m, za1v.q[w12, 0]
// CHECK-INST: mov     z5.q, p3/m, za1v.q[w12, 0]
// CHECK-ENCODING: [0x25,0x8c,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c38c25 <unknown>

mova    z1.q, p1/m, za1v.q[w12, 0]
// CHECK-INST: mov     z1.q, p1/m, za1v.q[w12, 0]
// CHECK-ENCODING: [0x21,0x84,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c38421 <unknown>

mova    z24.q, p5/m, za3v.q[w14, 0]
// CHECK-INST: mov     z24.q, p5/m, za3v.q[w14, 0]
// CHECK-ENCODING: [0x78,0xd4,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c3d478 <unknown>

mova    z0.q, p6/m, za12v.q[w12, 0]
// CHECK-INST: mov     z0.q, p6/m, za12v.q[w12, 0]
// CHECK-ENCODING: [0x80,0x99,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c39980 <unknown>

mova    z17.q, p2/m, za1v.q[w14, 0]
// CHECK-INST: mov     z17.q, p2/m, za1v.q[w14, 0]
// CHECK-ENCODING: [0x31,0xc8,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c3c831 <unknown>

mova    z29.q, p2/m, za6v.q[w12, 0]
// CHECK-INST: mov     z29.q, p2/m, za6v.q[w12, 0]
// CHECK-ENCODING: [0xdd,0x88,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c388dd <unknown>

mova    z2.q, p5/m, za9v.q[w15, 0]
// CHECK-INST: mov     z2.q, p5/m, za9v.q[w15, 0]
// CHECK-ENCODING: [0x22,0xf5,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c3f522 <unknown>

mova    z7.q, p2/m, za12v.q[w13, 0]
// CHECK-INST: mov     z7.q, p2/m, za12v.q[w13, 0]
// CHECK-ENCODING: [0x87,0xa9,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c3a987 <unknown>

// Aliases

mov     z0.q, p0/m, za0v.q[w12, 0]
// CHECK-INST: mov     z0.q, p0/m, za0v.q[w12, 0]
// CHECK-ENCODING: [0x00,0x80,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c38000 <unknown>

mov     z21.q, p5/m, za10v.q[w14, 0]
// CHECK-INST: mov     z21.q, p5/m, za10v.q[w14, 0]
// CHECK-ENCODING: [0x55,0xd5,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c3d555 <unknown>

mov     z23.q, p3/m, za13v.q[w15, 0]
// CHECK-INST: mov     z23.q, p3/m, za13v.q[w15, 0]
// CHECK-ENCODING: [0xb7,0xed,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c3edb7 <unknown>

mov     z31.q, p7/m, za15v.q[w15, 0]
// CHECK-INST: mov     z31.q, p7/m, za15v.q[w15, 0]
// CHECK-ENCODING: [0xff,0xfd,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c3fdff <unknown>

mov     z5.q, p3/m, za1v.q[w12, 0]
// CHECK-INST: mov     z5.q, p3/m, za1v.q[w12, 0]
// CHECK-ENCODING: [0x25,0x8c,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c38c25 <unknown>

mov     z1.q, p1/m, za1v.q[w12, 0]
// CHECK-INST: mov     z1.q, p1/m, za1v.q[w12, 0]
// CHECK-ENCODING: [0x21,0x84,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c38421 <unknown>

mov     z24.q, p5/m, za3v.q[w14, 0]
// CHECK-INST: mov     z24.q, p5/m, za3v.q[w14, 0]
// CHECK-ENCODING: [0x78,0xd4,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c3d478 <unknown>

mov     z0.q, p6/m, za12v.q[w12, 0]
// CHECK-INST: mov     z0.q, p6/m, za12v.q[w12, 0]
// CHECK-ENCODING: [0x80,0x99,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c39980 <unknown>

mov     z17.q, p2/m, za1v.q[w14, 0]
// CHECK-INST: mov     z17.q, p2/m, za1v.q[w14, 0]
// CHECK-ENCODING: [0x31,0xc8,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c3c831 <unknown>

mov     z29.q, p2/m, za6v.q[w12, 0]
// CHECK-INST: mov     z29.q, p2/m, za6v.q[w12, 0]
// CHECK-ENCODING: [0xdd,0x88,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c388dd <unknown>

mov     z2.q, p5/m, za9v.q[w15, 0]
// CHECK-INST: mov     z2.q, p5/m, za9v.q[w15, 0]
// CHECK-ENCODING: [0x22,0xf5,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c3f522 <unknown>

mov     z7.q, p2/m, za12v.q[w13, 0]
// CHECK-INST: mov     z7.q, p2/m, za12v.q[w13, 0]
// CHECK-ENCODING: [0x87,0xa9,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c3a987 <unknown>

// --------------------------------------------------------------------------//
// Insert, vector to tile, horizontal, 8-bit

mova    za0h.b[w12, 0], p0/m, z0.b
// CHECK-INST: mov     za0h.b[w12, 0], p0/m, z0.b
// CHECK-ENCODING: [0x00,0x00,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0000000 <unknown>

mova    za0h.b[w14, 5], p5/m, z10.b
// CHECK-INST: mov     za0h.b[w14, 5], p5/m, z10.b
// CHECK-ENCODING: [0x45,0x55,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0005545 <unknown>

mova    za0h.b[w15, 7], p3/m, z13.b
// CHECK-INST: mov     za0h.b[w15, 7], p3/m, z13.b
// CHECK-ENCODING: [0xa7,0x6d,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0006da7 <unknown>

mova    za0h.b[w15, 15], p7/m, z31.b
// CHECK-INST: mov     za0h.b[w15, 15], p7/m, z31.b
// CHECK-ENCODING: [0xef,0x7f,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0007fef <unknown>

mova    za0h.b[w12, 5], p3/m, z17.b
// CHECK-INST: mov     za0h.b[w12, 5], p3/m, z17.b
// CHECK-ENCODING: [0x25,0x0e,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0000e25 <unknown>

mova    za0h.b[w12, 1], p1/m, z1.b
// CHECK-INST: mov     za0h.b[w12, 1], p1/m, z1.b
// CHECK-ENCODING: [0x21,0x04,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0000421 <unknown>

mova    za0h.b[w14, 8], p5/m, z19.b
// CHECK-INST: mov     za0h.b[w14, 8], p5/m, z19.b
// CHECK-ENCODING: [0x68,0x56,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0005668 <unknown>

mova    za0h.b[w12, 0], p6/m, z12.b
// CHECK-INST: mov     za0h.b[w12, 0], p6/m, z12.b
// CHECK-ENCODING: [0x80,0x19,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0001980 <unknown>

mova    za0h.b[w14, 1], p2/m, z1.b
// CHECK-INST: mov     za0h.b[w14, 1], p2/m, z1.b
// CHECK-ENCODING: [0x21,0x48,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0004821 <unknown>

mova    za0h.b[w12, 13], p2/m, z22.b
// CHECK-INST: mov     za0h.b[w12, 13], p2/m, z22.b
// CHECK-ENCODING: [0xcd,0x0a,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0000acd <unknown>

mova    za0h.b[w15, 2], p5/m, z9.b
// CHECK-INST: mov     za0h.b[w15, 2], p5/m, z9.b
// CHECK-ENCODING: [0x22,0x75,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0007522 <unknown>

mova    za0h.b[w13, 7], p2/m, z12.b
// CHECK-INST: mov     za0h.b[w13, 7], p2/m, z12.b
// CHECK-ENCODING: [0x87,0x29,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0002987 <unknown>

// Aliases

mov     za0h.b[w12, 0], p0/m, z0.b
// CHECK-INST: mov     za0h.b[w12, 0], p0/m, z0.b
// CHECK-ENCODING: [0x00,0x00,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0000000 <unknown>

mov     za0h.b[w14, 5], p5/m, z10.b
// CHECK-INST: mov     za0h.b[w14, 5], p5/m, z10.b
// CHECK-ENCODING: [0x45,0x55,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0005545 <unknown>

mov     za0h.b[w15, 7], p3/m, z13.b
// CHECK-INST: mov     za0h.b[w15, 7], p3/m, z13.b
// CHECK-ENCODING: [0xa7,0x6d,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0006da7 <unknown>

mov     za0h.b[w15, 15], p7/m, z31.b
// CHECK-INST: mov     za0h.b[w15, 15], p7/m, z31.b
// CHECK-ENCODING: [0xef,0x7f,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0007fef <unknown>

mov     za0h.b[w12, 5], p3/m, z17.b
// CHECK-INST: mov     za0h.b[w12, 5], p3/m, z17.b
// CHECK-ENCODING: [0x25,0x0e,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0000e25 <unknown>

mov     za0h.b[w12, 1], p1/m, z1.b
// CHECK-INST: mov     za0h.b[w12, 1], p1/m, z1.b
// CHECK-ENCODING: [0x21,0x04,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0000421 <unknown>

mov     za0h.b[w14, 8], p5/m, z19.b
// CHECK-INST: mov     za0h.b[w14, 8], p5/m, z19.b
// CHECK-ENCODING: [0x68,0x56,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0005668 <unknown>

mov     za0h.b[w12, 0], p6/m, z12.b
// CHECK-INST: mov     za0h.b[w12, 0], p6/m, z12.b
// CHECK-ENCODING: [0x80,0x19,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0001980 <unknown>

mov     za0h.b[w14, 1], p2/m, z1.b
// CHECK-INST: mov     za0h.b[w14, 1], p2/m, z1.b
// CHECK-ENCODING: [0x21,0x48,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0004821 <unknown>

mov     za0h.b[w12, 13], p2/m, z22.b
// CHECK-INST: mov     za0h.b[w12, 13], p2/m, z22.b
// CHECK-ENCODING: [0xcd,0x0a,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0000acd <unknown>

mov     za0h.b[w15, 2], p5/m, z9.b
// CHECK-INST: mov     za0h.b[w15, 2], p5/m, z9.b
// CHECK-ENCODING: [0x22,0x75,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0007522 <unknown>

mov     za0h.b[w13, 7], p2/m, z12.b
// CHECK-INST: mov     za0h.b[w13, 7], p2/m, z12.b
// CHECK-ENCODING: [0x87,0x29,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0002987 <unknown>

// --------------------------------------------------------------------------//
// Insert, vector to tile, vertical, 8-bit

mova    za0v.b[w12, 0], p0/m, z0.b
// CHECK-INST: mov     za0v.b[w12, 0], p0/m, z0.b
// CHECK-ENCODING: [0x00,0x80,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0008000 <unknown>

mova    za0v.b[w14, 5], p5/m, z10.b
// CHECK-INST: mov     za0v.b[w14, 5], p5/m, z10.b
// CHECK-ENCODING: [0x45,0xd5,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c000d545 <unknown>

mova    za0v.b[w15, 7], p3/m, z13.b
// CHECK-INST: mov     za0v.b[w15, 7], p3/m, z13.b
// CHECK-ENCODING: [0xa7,0xed,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c000eda7 <unknown>

mova    za0v.b[w15, 15], p7/m, z31.b
// CHECK-INST: mov     za0v.b[w15, 15], p7/m, z31.b
// CHECK-ENCODING: [0xef,0xff,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c000ffef <unknown>

mova    za0v.b[w12, 5], p3/m, z17.b
// CHECK-INST: mov     za0v.b[w12, 5], p3/m, z17.b
// CHECK-ENCODING: [0x25,0x8e,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0008e25 <unknown>

mova    za0v.b[w12, 1], p1/m, z1.b
// CHECK-INST: mov     za0v.b[w12, 1], p1/m, z1.b
// CHECK-ENCODING: [0x21,0x84,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0008421 <unknown>

mova    za0v.b[w14, 8], p5/m, z19.b
// CHECK-INST: mov     za0v.b[w14, 8], p5/m, z19.b
// CHECK-ENCODING: [0x68,0xd6,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c000d668 <unknown>

mova    za0v.b[w12, 0], p6/m, z12.b
// CHECK-INST: mov     za0v.b[w12, 0], p6/m, z12.b
// CHECK-ENCODING: [0x80,0x99,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0009980 <unknown>

mova    za0v.b[w14, 1], p2/m, z1.b
// CHECK-INST: mov     za0v.b[w14, 1], p2/m, z1.b
// CHECK-ENCODING: [0x21,0xc8,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c000c821 <unknown>

mova    za0v.b[w12, 13], p2/m, z22.b
// CHECK-INST: mov     za0v.b[w12, 13], p2/m, z22.b
// CHECK-ENCODING: [0xcd,0x8a,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0008acd <unknown>

mova    za0v.b[w15, 2], p5/m, z9.b
// CHECK-INST: mov     za0v.b[w15, 2], p5/m, z9.b
// CHECK-ENCODING: [0x22,0xf5,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c000f522 <unknown>

mova    za0v.b[w13, 7], p2/m, z12.b
// CHECK-INST: mov     za0v.b[w13, 7], p2/m, z12.b
// CHECK-ENCODING: [0x87,0xa9,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c000a987 <unknown>

// Aliases

mov     za0v.b[w12, 0], p0/m, z0.b
// CHECK-INST: mov     za0v.b[w12, 0], p0/m, z0.b
// CHECK-ENCODING: [0x00,0x80,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0008000 <unknown>

mov     za0v.b[w14, 5], p5/m, z10.b
// CHECK-INST: mov     za0v.b[w14, 5], p5/m, z10.b
// CHECK-ENCODING: [0x45,0xd5,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c000d545 <unknown>

mov     za0v.b[w15, 7], p3/m, z13.b
// CHECK-INST: mov     za0v.b[w15, 7], p3/m, z13.b
// CHECK-ENCODING: [0xa7,0xed,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c000eda7 <unknown>

mov     za0v.b[w15, 15], p7/m, z31.b
// CHECK-INST: mov     za0v.b[w15, 15], p7/m, z31.b
// CHECK-ENCODING: [0xef,0xff,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c000ffef <unknown>

mov     za0v.b[w12, 5], p3/m, z17.b
// CHECK-INST: mov     za0v.b[w12, 5], p3/m, z17.b
// CHECK-ENCODING: [0x25,0x8e,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0008e25 <unknown>

mov     za0v.b[w12, 1], p1/m, z1.b
// CHECK-INST: mov     za0v.b[w12, 1], p1/m, z1.b
// CHECK-ENCODING: [0x21,0x84,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0008421 <unknown>

mov     za0v.b[w14, 8], p5/m, z19.b
// CHECK-INST: mov     za0v.b[w14, 8], p5/m, z19.b
// CHECK-ENCODING: [0x68,0xd6,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c000d668 <unknown>

mov     za0v.b[w12, 0], p6/m, z12.b
// CHECK-INST: mov     za0v.b[w12, 0], p6/m, z12.b
// CHECK-ENCODING: [0x80,0x99,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0009980 <unknown>

mov     za0v.b[w14, 1], p2/m, z1.b
// CHECK-INST: mov     za0v.b[w14, 1], p2/m, z1.b
// CHECK-ENCODING: [0x21,0xc8,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c000c821 <unknown>

mov     za0v.b[w12, 13], p2/m, z22.b
// CHECK-INST: mov     za0v.b[w12, 13], p2/m, z22.b
// CHECK-ENCODING: [0xcd,0x8a,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0008acd <unknown>

mov     za0v.b[w15, 2], p5/m, z9.b
// CHECK-INST: mov     za0v.b[w15, 2], p5/m, z9.b
// CHECK-ENCODING: [0x22,0xf5,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c000f522 <unknown>

mov     za0v.b[w13, 7], p2/m, z12.b
// CHECK-INST: mov     za0v.b[w13, 7], p2/m, z12.b
// CHECK-ENCODING: [0x87,0xa9,0x00,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c000a987 <unknown>

// --------------------------------------------------------------------------//
// Insert, vector to tile, horizontal, 16-bit

mova    za0h.h[w12, 0], p0/m, z0.h
// CHECK-INST: mov     za0h.h[w12, 0], p0/m, z0.h
// CHECK-ENCODING: [0x00,0x00,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0400000 <unknown>

mova    za0h.h[w14, 5], p5/m, z10.h
// CHECK-INST: mov     za0h.h[w14, 5], p5/m, z10.h
// CHECK-ENCODING: [0x45,0x55,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0405545 <unknown>

mova    za0h.h[w15, 7], p3/m, z13.h
// CHECK-INST: mov     za0h.h[w15, 7], p3/m, z13.h
// CHECK-ENCODING: [0xa7,0x6d,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0406da7 <unknown>

mova    za1h.h[w15, 7], p7/m, z31.h
// CHECK-INST: mov     za1h.h[w15, 7], p7/m, z31.h
// CHECK-ENCODING: [0xef,0x7f,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0407fef <unknown>

mova    za0h.h[w12, 5], p3/m, z17.h
// CHECK-INST: mov     za0h.h[w12, 5], p3/m, z17.h
// CHECK-ENCODING: [0x25,0x0e,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0400e25 <unknown>

mova    za0h.h[w12, 1], p1/m, z1.h
// CHECK-INST: mov     za0h.h[w12, 1], p1/m, z1.h
// CHECK-ENCODING: [0x21,0x04,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0400421 <unknown>

mova    za1h.h[w14, 0], p5/m, z19.h
// CHECK-INST: mov     za1h.h[w14, 0], p5/m, z19.h
// CHECK-ENCODING: [0x68,0x56,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0405668 <unknown>

mova    za0h.h[w12, 0], p6/m, z12.h
// CHECK-INST: mov     za0h.h[w12, 0], p6/m, z12.h
// CHECK-ENCODING: [0x80,0x19,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0401980 <unknown>

mova    za0h.h[w14, 1], p2/m, z1.h
// CHECK-INST: mov     za0h.h[w14, 1], p2/m, z1.h
// CHECK-ENCODING: [0x21,0x48,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0404821 <unknown>

mova    za1h.h[w12, 5], p2/m, z22.h
// CHECK-INST: mov     za1h.h[w12, 5], p2/m, z22.h
// CHECK-ENCODING: [0xcd,0x0a,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0400acd <unknown>

mova    za0h.h[w15, 2], p5/m, z9.h
// CHECK-INST: mov     za0h.h[w15, 2], p5/m, z9.h
// CHECK-ENCODING: [0x22,0x75,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0407522 <unknown>

mova    za0h.h[w13, 7], p2/m, z12.h
// CHECK-INST: mov     za0h.h[w13, 7], p2/m, z12.h
// CHECK-ENCODING: [0x87,0x29,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0402987 <unknown>

// Aliases

mov     za0h.h[w12, 0], p0/m, z0.h
// CHECK-INST: mov     za0h.h[w12, 0], p0/m, z0.h
// CHECK-ENCODING: [0x00,0x00,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0400000 <unknown>

mov     za0h.h[w14, 5], p5/m, z10.h
// CHECK-INST: mov     za0h.h[w14, 5], p5/m, z10.h
// CHECK-ENCODING: [0x45,0x55,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0405545 <unknown>

mov     za0h.h[w15, 7], p3/m, z13.h
// CHECK-INST: mov     za0h.h[w15, 7], p3/m, z13.h
// CHECK-ENCODING: [0xa7,0x6d,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0406da7 <unknown>

mov     za1h.h[w15, 7], p7/m, z31.h
// CHECK-INST: mov     za1h.h[w15, 7], p7/m, z31.h
// CHECK-ENCODING: [0xef,0x7f,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0407fef <unknown>

mov     za0h.h[w12, 5], p3/m, z17.h
// CHECK-INST: mov     za0h.h[w12, 5], p3/m, z17.h
// CHECK-ENCODING: [0x25,0x0e,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0400e25 <unknown>

mov     za0h.h[w12, 1], p1/m, z1.h
// CHECK-INST: mov     za0h.h[w12, 1], p1/m, z1.h
// CHECK-ENCODING: [0x21,0x04,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0400421 <unknown>

mov     za1h.h[w14, 0], p5/m, z19.h
// CHECK-INST: mov     za1h.h[w14, 0], p5/m, z19.h
// CHECK-ENCODING: [0x68,0x56,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0405668 <unknown>

mov     za0h.h[w12, 0], p6/m, z12.h
// CHECK-INST: mov     za0h.h[w12, 0], p6/m, z12.h
// CHECK-ENCODING: [0x80,0x19,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0401980 <unknown>

mov     za0h.h[w14, 1], p2/m, z1.h
// CHECK-INST: mov     za0h.h[w14, 1], p2/m, z1.h
// CHECK-ENCODING: [0x21,0x48,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0404821 <unknown>

mov     za1h.h[w12, 5], p2/m, z22.h
// CHECK-INST: mov     za1h.h[w12, 5], p2/m, z22.h
// CHECK-ENCODING: [0xcd,0x0a,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0400acd <unknown>

mov     za0h.h[w15, 2], p5/m, z9.h
// CHECK-INST: mov     za0h.h[w15, 2], p5/m, z9.h
// CHECK-ENCODING: [0x22,0x75,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0407522 <unknown>

mov     za0h.h[w13, 7], p2/m, z12.h
// CHECK-INST: mov     za0h.h[w13, 7], p2/m, z12.h
// CHECK-ENCODING: [0x87,0x29,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0402987 <unknown>

// --------------------------------------------------------------------------//
// Insert, vector to tile, vertical, 16-bit

mova    za0v.h[w12, 0], p0/m, z0.h
// CHECK-INST: mov     za0v.h[w12, 0], p0/m, z0.h
// CHECK-ENCODING: [0x00,0x80,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0408000 <unknown>

mova    za0v.h[w14, 5], p5/m, z10.h
// CHECK-INST: mov     za0v.h[w14, 5], p5/m, z10.h
// CHECK-ENCODING: [0x45,0xd5,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c040d545 <unknown>

mova    za0v.h[w15, 7], p3/m, z13.h
// CHECK-INST: mov     za0v.h[w15, 7], p3/m, z13.h
// CHECK-ENCODING: [0xa7,0xed,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c040eda7 <unknown>

mova    za1v.h[w15, 7], p7/m, z31.h
// CHECK-INST: mov     za1v.h[w15, 7], p7/m, z31.h
// CHECK-ENCODING: [0xef,0xff,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c040ffef <unknown>

mova    za0v.h[w12, 5], p3/m, z17.h
// CHECK-INST: mov     za0v.h[w12, 5], p3/m, z17.h
// CHECK-ENCODING: [0x25,0x8e,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0408e25 <unknown>

mova    za0v.h[w12, 1], p1/m, z1.h
// CHECK-INST: mov     za0v.h[w12, 1], p1/m, z1.h
// CHECK-ENCODING: [0x21,0x84,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0408421 <unknown>

mova    za1v.h[w14, 0], p5/m, z19.h
// CHECK-INST: mov     za1v.h[w14, 0], p5/m, z19.h
// CHECK-ENCODING: [0x68,0xd6,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c040d668 <unknown>

mova    za0v.h[w12, 0], p6/m, z12.h
// CHECK-INST: mov     za0v.h[w12, 0], p6/m, z12.h
// CHECK-ENCODING: [0x80,0x99,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0409980 <unknown>

mova    za0v.h[w14, 1], p2/m, z1.h
// CHECK-INST: mov     za0v.h[w14, 1], p2/m, z1.h
// CHECK-ENCODING: [0x21,0xc8,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c040c821 <unknown>

mova    za1v.h[w12, 5], p2/m, z22.h
// CHECK-INST: mov     za1v.h[w12, 5], p2/m, z22.h
// CHECK-ENCODING: [0xcd,0x8a,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0408acd <unknown>

mova    za0v.h[w15, 2], p5/m, z9.h
// CHECK-INST: mov     za0v.h[w15, 2], p5/m, z9.h
// CHECK-ENCODING: [0x22,0xf5,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c040f522 <unknown>

mova    za0v.h[w13, 7], p2/m, z12.h
// CHECK-INST: mov     za0v.h[w13, 7], p2/m, z12.h
// CHECK-ENCODING: [0x87,0xa9,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c040a987 <unknown>

// Aliases

mov     za0v.h[w12, 0], p0/m, z0.h
// CHECK-INST: mov     za0v.h[w12, 0], p0/m, z0.h
// CHECK-ENCODING: [0x00,0x80,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0408000 <unknown>

mov     za0v.h[w14, 5], p5/m, z10.h
// CHECK-INST: mov     za0v.h[w14, 5], p5/m, z10.h
// CHECK-ENCODING: [0x45,0xd5,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c040d545 <unknown>

mov     za0v.h[w15, 7], p3/m, z13.h
// CHECK-INST: mov     za0v.h[w15, 7], p3/m, z13.h
// CHECK-ENCODING: [0xa7,0xed,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c040eda7 <unknown>

mov     za1v.h[w15, 7], p7/m, z31.h
// CHECK-INST: mov     za1v.h[w15, 7], p7/m, z31.h
// CHECK-ENCODING: [0xef,0xff,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c040ffef <unknown>

mov     za0v.h[w12, 5], p3/m, z17.h
// CHECK-INST: mov     za0v.h[w12, 5], p3/m, z17.h
// CHECK-ENCODING: [0x25,0x8e,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0408e25 <unknown>

mov     za0v.h[w12, 1], p1/m, z1.h
// CHECK-INST: mov     za0v.h[w12, 1], p1/m, z1.h
// CHECK-ENCODING: [0x21,0x84,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0408421 <unknown>

mov     za1v.h[w14, 0], p5/m, z19.h
// CHECK-INST: mov     za1v.h[w14, 0], p5/m, z19.h
// CHECK-ENCODING: [0x68,0xd6,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c040d668 <unknown>

mov     za0v.h[w12, 0], p6/m, z12.h
// CHECK-INST: mov     za0v.h[w12, 0], p6/m, z12.h
// CHECK-ENCODING: [0x80,0x99,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0409980 <unknown>

mov     za0v.h[w14, 1], p2/m, z1.h
// CHECK-INST: mov     za0v.h[w14, 1], p2/m, z1.h
// CHECK-ENCODING: [0x21,0xc8,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c040c821 <unknown>

mov     za1v.h[w12, 5], p2/m, z22.h
// CHECK-INST: mov     za1v.h[w12, 5], p2/m, z22.h
// CHECK-ENCODING: [0xcd,0x8a,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0408acd <unknown>

mov     za0v.h[w15, 2], p5/m, z9.h
// CHECK-INST: mov     za0v.h[w15, 2], p5/m, z9.h
// CHECK-ENCODING: [0x22,0xf5,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c040f522 <unknown>

mov     za0v.h[w13, 7], p2/m, z12.h
// CHECK-INST: mov     za0v.h[w13, 7], p2/m, z12.h
// CHECK-ENCODING: [0x87,0xa9,0x40,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c040a987 <unknown>

// --------------------------------------------------------------------------//
// Insert, vector to tile, horizontal, 32-bit

mova    za0h.s[w12, 0], p0/m, z0.s
// CHECK-INST: mov     za0h.s[w12, 0], p0/m, z0.s
// CHECK-ENCODING: [0x00,0x00,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0800000 <unknown>

mova    za1h.s[w14, 1], p5/m, z10.s
// CHECK-INST: mov     za1h.s[w14, 1], p5/m, z10.s
// CHECK-ENCODING: [0x45,0x55,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0805545 <unknown>

mova    za1h.s[w15, 3], p3/m, z13.s
// CHECK-INST: mov     za1h.s[w15, 3], p3/m, z13.s
// CHECK-ENCODING: [0xa7,0x6d,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0806da7 <unknown>

mova    za3h.s[w15, 3], p7/m, z31.s
// CHECK-INST: mov     za3h.s[w15, 3], p7/m, z31.s
// CHECK-ENCODING: [0xef,0x7f,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0807fef <unknown>

mova    za1h.s[w12, 1], p3/m, z17.s
// CHECK-INST: mov     za1h.s[w12, 1], p3/m, z17.s
// CHECK-ENCODING: [0x25,0x0e,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0800e25 <unknown>

mova    za0h.s[w12, 1], p1/m, z1.s
// CHECK-INST: mov     za0h.s[w12, 1], p1/m, z1.s
// CHECK-ENCODING: [0x21,0x04,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0800421 <unknown>

mova    za2h.s[w14, 0], p5/m, z19.s
// CHECK-INST: mov     za2h.s[w14, 0], p5/m, z19.s
// CHECK-ENCODING: [0x68,0x56,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0805668 <unknown>

mova    za0h.s[w12, 0], p6/m, z12.s
// CHECK-INST: mov     za0h.s[w12, 0], p6/m, z12.s
// CHECK-ENCODING: [0x80,0x19,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0801980 <unknown>

mova    za0h.s[w14, 1], p2/m, z1.s
// CHECK-INST: mov     za0h.s[w14, 1], p2/m, z1.s
// CHECK-ENCODING: [0x21,0x48,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0804821 <unknown>

mova    za3h.s[w12, 1], p2/m, z22.s
// CHECK-INST: mov     za3h.s[w12, 1], p2/m, z22.s
// CHECK-ENCODING: [0xcd,0x0a,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0800acd <unknown>

mova    za0h.s[w15, 2], p5/m, z9.s
// CHECK-INST: mov     za0h.s[w15, 2], p5/m, z9.s
// CHECK-ENCODING: [0x22,0x75,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0807522 <unknown>

mova    za1h.s[w13, 3], p2/m, z12.s
// CHECK-INST: mov     za1h.s[w13, 3], p2/m, z12.s
// CHECK-ENCODING: [0x87,0x29,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0802987 <unknown>

// Aliases

mov     za0h.s[w12, 0], p0/m, z0.s
// CHECK-INST: mov     za0h.s[w12, 0], p0/m, z0.s
// CHECK-ENCODING: [0x00,0x00,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0800000 <unknown>

mov     za1h.s[w14, 1], p5/m, z10.s
// CHECK-INST: mov     za1h.s[w14, 1], p5/m, z10.s
// CHECK-ENCODING: [0x45,0x55,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0805545 <unknown>

mov     za1h.s[w15, 3], p3/m, z13.s
// CHECK-INST: mov     za1h.s[w15, 3], p3/m, z13.s
// CHECK-ENCODING: [0xa7,0x6d,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0806da7 <unknown>

mov     za3h.s[w15, 3], p7/m, z31.s
// CHECK-INST: mov     za3h.s[w15, 3], p7/m, z31.s
// CHECK-ENCODING: [0xef,0x7f,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0807fef <unknown>

mov     za1h.s[w12, 1], p3/m, z17.s
// CHECK-INST: mov     za1h.s[w12, 1], p3/m, z17.s
// CHECK-ENCODING: [0x25,0x0e,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0800e25 <unknown>

mov     za0h.s[w12, 1], p1/m, z1.s
// CHECK-INST: mov     za0h.s[w12, 1], p1/m, z1.s
// CHECK-ENCODING: [0x21,0x04,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0800421 <unknown>

mov     za2h.s[w14, 0], p5/m, z19.s
// CHECK-INST: mov     za2h.s[w14, 0], p5/m, z19.s
// CHECK-ENCODING: [0x68,0x56,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0805668 <unknown>

mov     za0h.s[w12, 0], p6/m, z12.s
// CHECK-INST: mov     za0h.s[w12, 0], p6/m, z12.s
// CHECK-ENCODING: [0x80,0x19,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0801980 <unknown>

mov     za0h.s[w14, 1], p2/m, z1.s
// CHECK-INST: mov     za0h.s[w14, 1], p2/m, z1.s
// CHECK-ENCODING: [0x21,0x48,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0804821 <unknown>

mov     za3h.s[w12, 1], p2/m, z22.s
// CHECK-INST: mov     za3h.s[w12, 1], p2/m, z22.s
// CHECK-ENCODING: [0xcd,0x0a,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0800acd <unknown>

mov     za0h.s[w15, 2], p5/m, z9.s
// CHECK-INST: mov     za0h.s[w15, 2], p5/m, z9.s
// CHECK-ENCODING: [0x22,0x75,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0807522 <unknown>

mov     za1h.s[w13, 3], p2/m, z12.s
// CHECK-INST: mov     za1h.s[w13, 3], p2/m, z12.s
// CHECK-ENCODING: [0x87,0x29,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0802987 <unknown>

// --------------------------------------------------------------------------//
// Insert, vector to tile, vertical, 32-bit

mova    za0v.s[w12, 0], p0/m, z0.s
// CHECK-INST: mov     za0v.s[w12, 0], p0/m, z0.s
// CHECK-ENCODING: [0x00,0x80,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0808000 <unknown>

mova    za1v.s[w14, 1], p5/m, z10.s
// CHECK-INST: mov     za1v.s[w14, 1], p5/m, z10.s
// CHECK-ENCODING: [0x45,0xd5,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c080d545 <unknown>

mova    za1v.s[w15, 3], p3/m, z13.s
// CHECK-INST: mov     za1v.s[w15, 3], p3/m, z13.s
// CHECK-ENCODING: [0xa7,0xed,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c080eda7 <unknown>

mova    za3v.s[w15, 3], p7/m, z31.s
// CHECK-INST: mov     za3v.s[w15, 3], p7/m, z31.s
// CHECK-ENCODING: [0xef,0xff,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c080ffef <unknown>

mova    za1v.s[w12, 1], p3/m, z17.s
// CHECK-INST: mov     za1v.s[w12, 1], p3/m, z17.s
// CHECK-ENCODING: [0x25,0x8e,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0808e25 <unknown>

mova    za0v.s[w12, 1], p1/m, z1.s
// CHECK-INST: mov     za0v.s[w12, 1], p1/m, z1.s
// CHECK-ENCODING: [0x21,0x84,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0808421 <unknown>

mova    za2v.s[w14, 0], p5/m, z19.s
// CHECK-INST: mov     za2v.s[w14, 0], p5/m, z19.s
// CHECK-ENCODING: [0x68,0xd6,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c080d668 <unknown>

mova    za0v.s[w12, 0], p6/m, z12.s
// CHECK-INST: mov     za0v.s[w12, 0], p6/m, z12.s
// CHECK-ENCODING: [0x80,0x99,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0809980 <unknown>

mova    za0v.s[w14, 1], p2/m, z1.s
// CHECK-INST: mov     za0v.s[w14, 1], p2/m, z1.s
// CHECK-ENCODING: [0x21,0xc8,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c080c821 <unknown>

mova    za3v.s[w12, 1], p2/m, z22.s
// CHECK-INST: mov     za3v.s[w12, 1], p2/m, z22.s
// CHECK-ENCODING: [0xcd,0x8a,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0808acd <unknown>

mova    za0v.s[w15, 2], p5/m, z9.s
// CHECK-INST: mov     za0v.s[w15, 2], p5/m, z9.s
// CHECK-ENCODING: [0x22,0xf5,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c080f522 <unknown>

mova    za1v.s[w13, 3], p2/m, z12.s
// CHECK-INST: mov     za1v.s[w13, 3], p2/m, z12.s
// CHECK-ENCODING: [0x87,0xa9,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c080a987 <unknown>

// Aliases

mov     za0v.s[w12, 0], p0/m, z0.s
// CHECK-INST: mov     za0v.s[w12, 0], p0/m, z0.s
// CHECK-ENCODING: [0x00,0x80,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0808000 <unknown>

mov     za1v.s[w14, 1], p5/m, z10.s
// CHECK-INST: mov     za1v.s[w14, 1], p5/m, z10.s
// CHECK-ENCODING: [0x45,0xd5,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c080d545 <unknown>

mov     za1v.s[w15, 3], p3/m, z13.s
// CHECK-INST: mov     za1v.s[w15, 3], p3/m, z13.s
// CHECK-ENCODING: [0xa7,0xed,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c080eda7 <unknown>

mov     za3v.s[w15, 3], p7/m, z31.s
// CHECK-INST: mov     za3v.s[w15, 3], p7/m, z31.s
// CHECK-ENCODING: [0xef,0xff,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c080ffef <unknown>

mov     za1v.s[w12, 1], p3/m, z17.s
// CHECK-INST: mov     za1v.s[w12, 1], p3/m, z17.s
// CHECK-ENCODING: [0x25,0x8e,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0808e25 <unknown>

mov     za0v.s[w12, 1], p1/m, z1.s
// CHECK-INST: mov     za0v.s[w12, 1], p1/m, z1.s
// CHECK-ENCODING: [0x21,0x84,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0808421 <unknown>

mov     za2v.s[w14, 0], p5/m, z19.s
// CHECK-INST: mov     za2v.s[w14, 0], p5/m, z19.s
// CHECK-ENCODING: [0x68,0xd6,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c080d668 <unknown>

mov     za0v.s[w12, 0], p6/m, z12.s
// CHECK-INST: mov     za0v.s[w12, 0], p6/m, z12.s
// CHECK-ENCODING: [0x80,0x99,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0809980 <unknown>

mov     za0v.s[w14, 1], p2/m, z1.s
// CHECK-INST: mov     za0v.s[w14, 1], p2/m, z1.s
// CHECK-ENCODING: [0x21,0xc8,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c080c821 <unknown>

mov     za3v.s[w12, 1], p2/m, z22.s
// CHECK-INST: mov     za3v.s[w12, 1], p2/m, z22.s
// CHECK-ENCODING: [0xcd,0x8a,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0808acd <unknown>

mov     za0v.s[w15, 2], p5/m, z9.s
// CHECK-INST: mov     za0v.s[w15, 2], p5/m, z9.s
// CHECK-ENCODING: [0x22,0xf5,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c080f522 <unknown>

mov     za1v.s[w13, 3], p2/m, z12.s
// CHECK-INST: mov     za1v.s[w13, 3], p2/m, z12.s
// CHECK-ENCODING: [0x87,0xa9,0x80,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c080a987 <unknown>

// --------------------------------------------------------------------------//
// Insert, vector to tile, horizontal, 64-bit

mova    za0h.d[w12, 0], p0/m, z0.d
// CHECK-INST: mov     za0h.d[w12, 0], p0/m, z0.d
// CHECK-ENCODING: [0x00,0x00,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c00000 <unknown>

mova    za2h.d[w14, 1], p5/m, z10.d
// CHECK-INST: mov     za2h.d[w14, 1], p5/m, z10.d
// CHECK-ENCODING: [0x45,0x55,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c05545 <unknown>

mova    za3h.d[w15, 1], p3/m, z13.d
// CHECK-INST: mov     za3h.d[w15, 1], p3/m, z13.d
// CHECK-ENCODING: [0xa7,0x6d,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c06da7 <unknown>

mova    za7h.d[w15, 1], p7/m, z31.d
// CHECK-INST: mov     za7h.d[w15, 1], p7/m, z31.d
// CHECK-ENCODING: [0xef,0x7f,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c07fef <unknown>

mova    za2h.d[w12, 1], p3/m, z17.d
// CHECK-INST: mov     za2h.d[w12, 1], p3/m, z17.d
// CHECK-ENCODING: [0x25,0x0e,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c00e25 <unknown>

mova    za0h.d[w12, 1], p1/m, z1.d
// CHECK-INST: mov     za0h.d[w12, 1], p1/m, z1.d
// CHECK-ENCODING: [0x21,0x04,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c00421 <unknown>

mova    za4h.d[w14, 0], p5/m, z19.d
// CHECK-INST: mov     za4h.d[w14, 0], p5/m, z19.d
// CHECK-ENCODING: [0x68,0x56,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c05668 <unknown>

mova    za0h.d[w12, 0], p6/m, z12.d
// CHECK-INST: mov     za0h.d[w12, 0], p6/m, z12.d
// CHECK-ENCODING: [0x80,0x19,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c01980 <unknown>

mova    za0h.d[w14, 1], p2/m, z1.d
// CHECK-INST: mov     za0h.d[w14, 1], p2/m, z1.d
// CHECK-ENCODING: [0x21,0x48,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c04821 <unknown>

mova    za6h.d[w12, 1], p2/m, z22.d
// CHECK-INST: mov     za6h.d[w12, 1], p2/m, z22.d
// CHECK-ENCODING: [0xcd,0x0a,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c00acd <unknown>

mova    za1h.d[w15, 0], p5/m, z9.d
// CHECK-INST: mov     za1h.d[w15, 0], p5/m, z9.d
// CHECK-ENCODING: [0x22,0x75,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c07522 <unknown>

mova    za3h.d[w13, 1], p2/m, z12.d
// CHECK-INST: mov     za3h.d[w13, 1], p2/m, z12.d
// CHECK-ENCODING: [0x87,0x29,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c02987 <unknown>

// Aliases

mov     za0h.d[w12, 0], p0/m, z0.d
// CHECK-INST: mov     za0h.d[w12, 0], p0/m, z0.d
// CHECK-ENCODING: [0x00,0x00,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c00000 <unknown>

mov     za2h.d[w14, 1], p5/m, z10.d
// CHECK-INST: mov     za2h.d[w14, 1], p5/m, z10.d
// CHECK-ENCODING: [0x45,0x55,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c05545 <unknown>

mov     za3h.d[w15, 1], p3/m, z13.d
// CHECK-INST: mov     za3h.d[w15, 1], p3/m, z13.d
// CHECK-ENCODING: [0xa7,0x6d,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c06da7 <unknown>

mov     za7h.d[w15, 1], p7/m, z31.d
// CHECK-INST: mov     za7h.d[w15, 1], p7/m, z31.d
// CHECK-ENCODING: [0xef,0x7f,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c07fef <unknown>

mov     za2h.d[w12, 1], p3/m, z17.d
// CHECK-INST: mov     za2h.d[w12, 1], p3/m, z17.d
// CHECK-ENCODING: [0x25,0x0e,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c00e25 <unknown>

mov     za0h.d[w12, 1], p1/m, z1.d
// CHECK-INST: mov     za0h.d[w12, 1], p1/m, z1.d
// CHECK-ENCODING: [0x21,0x04,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c00421 <unknown>

mov     za4h.d[w14, 0], p5/m, z19.d
// CHECK-INST: mov     za4h.d[w14, 0], p5/m, z19.d
// CHECK-ENCODING: [0x68,0x56,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c05668 <unknown>

mov     za0h.d[w12, 0], p6/m, z12.d
// CHECK-INST: mov     za0h.d[w12, 0], p6/m, z12.d
// CHECK-ENCODING: [0x80,0x19,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c01980 <unknown>

mov     za0h.d[w14, 1], p2/m, z1.d
// CHECK-INST: mov     za0h.d[w14, 1], p2/m, z1.d
// CHECK-ENCODING: [0x21,0x48,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c04821 <unknown>

mov     za6h.d[w12, 1], p2/m, z22.d
// CHECK-INST: mov     za6h.d[w12, 1], p2/m, z22.d
// CHECK-ENCODING: [0xcd,0x0a,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c00acd <unknown>

mov     za1h.d[w15, 0], p5/m, z9.d
// CHECK-INST: mov     za1h.d[w15, 0], p5/m, z9.d
// CHECK-ENCODING: [0x22,0x75,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c07522 <unknown>

mov     za3h.d[w13, 1], p2/m, z12.d
// CHECK-INST: mov     za3h.d[w13, 1], p2/m, z12.d
// CHECK-ENCODING: [0x87,0x29,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c02987 <unknown>

// --------------------------------------------------------------------------//
// Insert, vector to tile, vertical, 64-bit

mova    za0v.d[w12, 0], p0/m, z0.d
// CHECK-INST: mov     za0v.d[w12, 0], p0/m, z0.d
// CHECK-ENCODING: [0x00,0x80,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c08000 <unknown>

mova    za2v.d[w14, 1], p5/m, z10.d
// CHECK-INST: mov     za2v.d[w14, 1], p5/m, z10.d
// CHECK-ENCODING: [0x45,0xd5,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c0d545 <unknown>

mova    za3v.d[w15, 1], p3/m, z13.d
// CHECK-INST: mov     za3v.d[w15, 1], p3/m, z13.d
// CHECK-ENCODING: [0xa7,0xed,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c0eda7 <unknown>

mova    za7v.d[w15, 1], p7/m, z31.d
// CHECK-INST: mov     za7v.d[w15, 1], p7/m, z31.d
// CHECK-ENCODING: [0xef,0xff,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c0ffef <unknown>

mova    za2v.d[w12, 1], p3/m, z17.d
// CHECK-INST: mov     za2v.d[w12, 1], p3/m, z17.d
// CHECK-ENCODING: [0x25,0x8e,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c08e25 <unknown>

mova    za0v.d[w12, 1], p1/m, z1.d
// CHECK-INST: mov     za0v.d[w12, 1], p1/m, z1.d
// CHECK-ENCODING: [0x21,0x84,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c08421 <unknown>

mova    za4v.d[w14, 0], p5/m, z19.d
// CHECK-INST: mov     za4v.d[w14, 0], p5/m, z19.d
// CHECK-ENCODING: [0x68,0xd6,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c0d668 <unknown>

mova    za0v.d[w12, 0], p6/m, z12.d
// CHECK-INST: mov     za0v.d[w12, 0], p6/m, z12.d
// CHECK-ENCODING: [0x80,0x99,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c09980 <unknown>

mova    za0v.d[w14, 1], p2/m, z1.d
// CHECK-INST: mov     za0v.d[w14, 1], p2/m, z1.d
// CHECK-ENCODING: [0x21,0xc8,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c0c821 <unknown>

mova    za6v.d[w12, 1], p2/m, z22.d
// CHECK-INST: mov     za6v.d[w12, 1], p2/m, z22.d
// CHECK-ENCODING: [0xcd,0x8a,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c08acd <unknown>

mova    za1v.d[w15, 0], p5/m, z9.d
// CHECK-INST: mov     za1v.d[w15, 0], p5/m, z9.d
// CHECK-ENCODING: [0x22,0xf5,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c0f522 <unknown>

mova    za3v.d[w13, 1], p2/m, z12.d
// CHECK-INST: mov     za3v.d[w13, 1], p2/m, z12.d
// CHECK-ENCODING: [0x87,0xa9,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c0a987 <unknown>

// Aliases

mov     za0v.d[w12, 0], p0/m, z0.d
// CHECK-INST: mov     za0v.d[w12, 0], p0/m, z0.d
// CHECK-ENCODING: [0x00,0x80,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c08000 <unknown>

mov     za2v.d[w14, 1], p5/m, z10.d
// CHECK-INST: mov     za2v.d[w14, 1], p5/m, z10.d
// CHECK-ENCODING: [0x45,0xd5,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c0d545 <unknown>

mov     za3v.d[w15, 1], p3/m, z13.d
// CHECK-INST: mov     za3v.d[w15, 1], p3/m, z13.d
// CHECK-ENCODING: [0xa7,0xed,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c0eda7 <unknown>

mov     za7v.d[w15, 1], p7/m, z31.d
// CHECK-INST: mov     za7v.d[w15, 1], p7/m, z31.d
// CHECK-ENCODING: [0xef,0xff,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c0ffef <unknown>

mov     za2v.d[w12, 1], p3/m, z17.d
// CHECK-INST: mov     za2v.d[w12, 1], p3/m, z17.d
// CHECK-ENCODING: [0x25,0x8e,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c08e25 <unknown>

mov     za0v.d[w12, 1], p1/m, z1.d
// CHECK-INST: mov     za0v.d[w12, 1], p1/m, z1.d
// CHECK-ENCODING: [0x21,0x84,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c08421 <unknown>

mov     za4v.d[w14, 0], p5/m, z19.d
// CHECK-INST: mov     za4v.d[w14, 0], p5/m, z19.d
// CHECK-ENCODING: [0x68,0xd6,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c0d668 <unknown>

mov     za0v.d[w12, 0], p6/m, z12.d
// CHECK-INST: mov     za0v.d[w12, 0], p6/m, z12.d
// CHECK-ENCODING: [0x80,0x99,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c09980 <unknown>

mov     za0v.d[w14, 1], p2/m, z1.d
// CHECK-INST: mov     za0v.d[w14, 1], p2/m, z1.d
// CHECK-ENCODING: [0x21,0xc8,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c0c821 <unknown>

mov     za6v.d[w12, 1], p2/m, z22.d
// CHECK-INST: mov     za6v.d[w12, 1], p2/m, z22.d
// CHECK-ENCODING: [0xcd,0x8a,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c08acd <unknown>

mov     za1v.d[w15, 0], p5/m, z9.d
// CHECK-INST: mov     za1v.d[w15, 0], p5/m, z9.d
// CHECK-ENCODING: [0x22,0xf5,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c0f522 <unknown>

mov     za3v.d[w13, 1], p2/m, z12.d
// CHECK-INST: mov     za3v.d[w13, 1], p2/m, z12.d
// CHECK-ENCODING: [0x87,0xa9,0xc0,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c0a987 <unknown>

// --------------------------------------------------------------------------//
// Insert, vector to tile, horizontal, 128-bit

mova    za0h.q[w12, 0], p0/m, z0.q
// CHECK-INST: mov     za0h.q[w12, 0], p0/m, z0.q
// CHECK-ENCODING: [0x00,0x00,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c10000 <unknown>

mova    za5h.q[w14, 0], p5/m, z10.q
// CHECK-INST: mov     za5h.q[w14, 0], p5/m, z10.q
// CHECK-ENCODING: [0x45,0x55,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c15545 <unknown>

mova    za7h.q[w15, 0], p3/m, z13.q
// CHECK-INST: mov     za7h.q[w15, 0], p3/m, z13.q
// CHECK-ENCODING: [0xa7,0x6d,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c16da7 <unknown>

mova    za15h.q[w15, 0], p7/m, z31.q
// CHECK-INST: mov     za15h.q[w15, 0], p7/m, z31.q
// CHECK-ENCODING: [0xef,0x7f,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c17fef <unknown>

mova    za5h.q[w12, 0], p3/m, z17.q
// CHECK-INST: mov     za5h.q[w12, 0], p3/m, z17.q
// CHECK-ENCODING: [0x25,0x0e,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c10e25 <unknown>

mova    za1h.q[w12, 0], p1/m, z1.q
// CHECK-INST: mov     za1h.q[w12, 0], p1/m, z1.q
// CHECK-ENCODING: [0x21,0x04,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c10421 <unknown>

mova    za8h.q[w14, 0], p5/m, z19.q
// CHECK-INST: mov     za8h.q[w14, 0], p5/m, z19.q
// CHECK-ENCODING: [0x68,0x56,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c15668 <unknown>

mova    za0h.q[w12, 0], p6/m, z12.q
// CHECK-INST: mov     za0h.q[w12, 0], p6/m, z12.q
// CHECK-ENCODING: [0x80,0x19,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c11980 <unknown>

mova    za1h.q[w14, 0], p2/m, z1.q
// CHECK-INST: mov     za1h.q[w14, 0], p2/m, z1.q
// CHECK-ENCODING: [0x21,0x48,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c14821 <unknown>

mova    za13h.q[w12, 0], p2/m, z22.q
// CHECK-INST: mov     za13h.q[w12, 0], p2/m, z22.q
// CHECK-ENCODING: [0xcd,0x0a,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c10acd <unknown>

mova    za2h.q[w15, 0], p5/m, z9.q
// CHECK-INST: mov     za2h.q[w15, 0], p5/m, z9.q
// CHECK-ENCODING: [0x22,0x75,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c17522 <unknown>

mova    za7h.q[w13, 0], p2/m, z12.q
// CHECK-INST: mov     za7h.q[w13, 0], p2/m, z12.q
// CHECK-ENCODING: [0x87,0x29,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c12987 <unknown>

// Aliases

mov     za0h.q[w12, 0], p0/m, z0.q
// CHECK-INST: mov     za0h.q[w12, 0], p0/m, z0.q
// CHECK-ENCODING: [0x00,0x00,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c10000 <unknown>

mov     za5h.q[w14, 0], p5/m, z10.q
// CHECK-INST: mov     za5h.q[w14, 0], p5/m, z10.q
// CHECK-ENCODING: [0x45,0x55,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c15545 <unknown>

mov     za7h.q[w15, 0], p3/m, z13.q
// CHECK-INST: mov     za7h.q[w15, 0], p3/m, z13.q
// CHECK-ENCODING: [0xa7,0x6d,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c16da7 <unknown>

mov     za15h.q[w15, 0], p7/m, z31.q
// CHECK-INST: mov     za15h.q[w15, 0], p7/m, z31.q
// CHECK-ENCODING: [0xef,0x7f,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c17fef <unknown>

mov     za5h.q[w12, 0], p3/m, z17.q
// CHECK-INST: mov     za5h.q[w12, 0], p3/m, z17.q
// CHECK-ENCODING: [0x25,0x0e,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c10e25 <unknown>

mov     za1h.q[w12, 0], p1/m, z1.q
// CHECK-INST: mov     za1h.q[w12, 0], p1/m, z1.q
// CHECK-ENCODING: [0x21,0x04,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c10421 <unknown>

mov     za8h.q[w14, 0], p5/m, z19.q
// CHECK-INST: mov     za8h.q[w14, 0], p5/m, z19.q
// CHECK-ENCODING: [0x68,0x56,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c15668 <unknown>

mov     za0h.q[w12, 0], p6/m, z12.q
// CHECK-INST: mov     za0h.q[w12, 0], p6/m, z12.q
// CHECK-ENCODING: [0x80,0x19,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c11980 <unknown>

mov     za1h.q[w14, 0], p2/m, z1.q
// CHECK-INST: mov     za1h.q[w14, 0], p2/m, z1.q
// CHECK-ENCODING: [0x21,0x48,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c14821 <unknown>

mov     za13h.q[w12, 0], p2/m, z22.q
// CHECK-INST: mov     za13h.q[w12, 0], p2/m, z22.q
// CHECK-ENCODING: [0xcd,0x0a,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c10acd <unknown>

mov     za2h.q[w15, 0], p5/m, z9.q
// CHECK-INST: mov     za2h.q[w15, 0], p5/m, z9.q
// CHECK-ENCODING: [0x22,0x75,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c17522 <unknown>

mov     za7h.q[w13, 0], p2/m, z12.q
// CHECK-INST: mov     za7h.q[w13, 0], p2/m, z12.q
// CHECK-ENCODING: [0x87,0x29,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c12987 <unknown>

// --------------------------------------------------------------------------//
// Insert, vector to tile, vertical, 128-bit

mova    za0v.q[w12, 0], p0/m, z0.q
// CHECK-INST: mov     za0v.q[w12, 0], p0/m, z0.q
// CHECK-ENCODING: [0x00,0x80,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c18000 <unknown>

mova    za5v.q[w14, 0], p5/m, z10.q
// CHECK-INST: mov     za5v.q[w14, 0], p5/m, z10.q
// CHECK-ENCODING: [0x45,0xd5,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c1d545 <unknown>

mova    za7v.q[w15, 0], p3/m, z13.q
// CHECK-INST: mov     za7v.q[w15, 0], p3/m, z13.q
// CHECK-ENCODING: [0xa7,0xed,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c1eda7 <unknown>

mova    za15v.q[w15, 0], p7/m, z31.q
// CHECK-INST: mov     za15v.q[w15, 0], p7/m, z31.q
// CHECK-ENCODING: [0xef,0xff,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c1ffef <unknown>

mova    za5v.q[w12, 0], p3/m, z17.q
// CHECK-INST: mov     za5v.q[w12, 0], p3/m, z17.q
// CHECK-ENCODING: [0x25,0x8e,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c18e25 <unknown>

mova    za1v.q[w12, 0], p1/m, z1.q
// CHECK-INST: mov     za1v.q[w12, 0], p1/m, z1.q
// CHECK-ENCODING: [0x21,0x84,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c18421 <unknown>

mova    za8v.q[w14, 0], p5/m, z19.q
// CHECK-INST: mov     za8v.q[w14, 0], p5/m, z19.q
// CHECK-ENCODING: [0x68,0xd6,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c1d668 <unknown>

mova    za0v.q[w12, 0], p6/m, z12.q
// CHECK-INST: mov     za0v.q[w12, 0], p6/m, z12.q
// CHECK-ENCODING: [0x80,0x99,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c19980 <unknown>

mova    za1v.q[w14, 0], p2/m, z1.q
// CHECK-INST: mov     za1v.q[w14, 0], p2/m, z1.q
// CHECK-ENCODING: [0x21,0xc8,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c1c821 <unknown>

mova    za13v.q[w12, 0], p2/m, z22.q
// CHECK-INST: mov     za13v.q[w12, 0], p2/m, z22.q
// CHECK-ENCODING: [0xcd,0x8a,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c18acd <unknown>

mova    za2v.q[w15, 0], p5/m, z9.q
// CHECK-INST: mov     za2v.q[w15, 0], p5/m, z9.q
// CHECK-ENCODING: [0x22,0xf5,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c1f522 <unknown>

mova    za7v.q[w13, 0], p2/m, z12.q
// CHECK-INST: mov     za7v.q[w13, 0], p2/m, z12.q
// CHECK-ENCODING: [0x87,0xa9,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c1a987 <unknown>

// Aliases

mov     za0v.q[w12, 0], p0/m, z0.q
// CHECK-INST: mov     za0v.q[w12, 0], p0/m, z0.q
// CHECK-ENCODING: [0x00,0x80,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c18000 <unknown>

mov     za5v.q[w14, 0], p5/m, z10.q
// CHECK-INST: mov     za5v.q[w14, 0], p5/m, z10.q
// CHECK-ENCODING: [0x45,0xd5,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c1d545 <unknown>

mov     za7v.q[w15, 0], p3/m, z13.q
// CHECK-INST: mov     za7v.q[w15, 0], p3/m, z13.q
// CHECK-ENCODING: [0xa7,0xed,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c1eda7 <unknown>

mov     za15v.q[w15, 0], p7/m, z31.q
// CHECK-INST: mov     za15v.q[w15, 0], p7/m, z31.q
// CHECK-ENCODING: [0xef,0xff,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c1ffef <unknown>

mov     za5v.q[w12, 0], p3/m, z17.q
// CHECK-INST: mov     za5v.q[w12, 0], p3/m, z17.q
// CHECK-ENCODING: [0x25,0x8e,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c18e25 <unknown>

mov     za1v.q[w12, 0], p1/m, z1.q
// CHECK-INST: mov     za1v.q[w12, 0], p1/m, z1.q
// CHECK-ENCODING: [0x21,0x84,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c18421 <unknown>

mov     za8v.q[w14, 0], p5/m, z19.q
// CHECK-INST: mov     za8v.q[w14, 0], p5/m, z19.q
// CHECK-ENCODING: [0x68,0xd6,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c1d668 <unknown>

mov     za0v.q[w12, 0], p6/m, z12.q
// CHECK-INST: mov     za0v.q[w12, 0], p6/m, z12.q
// CHECK-ENCODING: [0x80,0x99,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c19980 <unknown>

mov     za1v.q[w14, 0], p2/m, z1.q
// CHECK-INST: mov     za1v.q[w14, 0], p2/m, z1.q
// CHECK-ENCODING: [0x21,0xc8,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c1c821 <unknown>

mov     za13v.q[w12, 0], p2/m, z22.q
// CHECK-INST: mov     za13v.q[w12, 0], p2/m, z22.q
// CHECK-ENCODING: [0xcd,0x8a,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c18acd <unknown>

mov     za2v.q[w15, 0], p5/m, z9.q
// CHECK-INST: mov     za2v.q[w15, 0], p5/m, z9.q
// CHECK-ENCODING: [0x22,0xf5,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c1f522 <unknown>

mov     za7v.q[w13, 0], p2/m, z12.q
// CHECK-INST: mov     za7v.q[w13, 0], p2/m, z12.q
// CHECK-ENCODING: [0x87,0xa9,0xc1,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c0c1a987 <unknown>
