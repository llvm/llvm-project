// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+sme-i16i64 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2,+sme-i16i64 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2,+sme-i16i64 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2,+sme-i16i64 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+sme-i16i64 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2,+sme-i16i64 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


udot    za.s[w8, 0, vgx2], {z0.h, z1.h}, z0.h[0]  // 11000001-01010000-00010000-00010000
// CHECK-INST: udot    za.s[w8, 0, vgx2], { z0.h, z1.h }, z0.h[0]
// CHECK-ENCODING: [0x10,0x10,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1501010 <unknown>

udot    za.s[w8, 0], {z0.h, z1.h}, z0.h[0]  // 11000001-01010000-00010000-00010000
// CHECK-INST: udot    za.s[w8, 0, vgx2], { z0.h, z1.h }, z0.h[0]
// CHECK-ENCODING: [0x10,0x10,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1501010 <unknown>

udot    za.s[w10, 5, vgx2], {z10.h, z11.h}, z5.h[1]  // 11000001-01010101-01010101-01010101
// CHECK-INST: udot    za.s[w10, 5, vgx2], { z10.h, z11.h }, z5.h[1]
// CHECK-ENCODING: [0x55,0x55,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1555555 <unknown>

udot    za.s[w10, 5], {z10.h, z11.h}, z5.h[1]  // 11000001-01010101-01010101-01010101
// CHECK-INST: udot    za.s[w10, 5, vgx2], { z10.h, z11.h }, z5.h[1]
// CHECK-ENCODING: [0x55,0x55,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1555555 <unknown>

udot    za.s[w11, 7, vgx2], {z12.h, z13.h}, z8.h[3]  // 11000001-01011000-01111101-10010111
// CHECK-INST: udot    za.s[w11, 7, vgx2], { z12.h, z13.h }, z8.h[3]
// CHECK-ENCODING: [0x97,0x7d,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1587d97 <unknown>

udot    za.s[w11, 7], {z12.h, z13.h}, z8.h[3]  // 11000001-01011000-01111101-10010111
// CHECK-INST: udot    za.s[w11, 7, vgx2], { z12.h, z13.h }, z8.h[3]
// CHECK-ENCODING: [0x97,0x7d,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1587d97 <unknown>

udot    za.s[w11, 7, vgx2], {z30.h, z31.h}, z15.h[3]  // 11000001-01011111-01111111-11010111
// CHECK-INST: udot    za.s[w11, 7, vgx2], { z30.h, z31.h }, z15.h[3]
// CHECK-ENCODING: [0xd7,0x7f,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15f7fd7 <unknown>

udot    za.s[w11, 7], {z30.h, z31.h}, z15.h[3]  // 11000001-01011111-01111111-11010111
// CHECK-INST: udot    za.s[w11, 7, vgx2], { z30.h, z31.h }, z15.h[3]
// CHECK-ENCODING: [0xd7,0x7f,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15f7fd7 <unknown>

udot    za.s[w8, 5, vgx2], {z16.h, z17.h}, z0.h[3]  // 11000001-01010000-00011110-00010101
// CHECK-INST: udot    za.s[w8, 5, vgx2], { z16.h, z17.h }, z0.h[3]
// CHECK-ENCODING: [0x15,0x1e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1501e15 <unknown>

udot    za.s[w8, 5], {z16.h, z17.h}, z0.h[3]  // 11000001-01010000-00011110-00010101
// CHECK-INST: udot    za.s[w8, 5, vgx2], { z16.h, z17.h }, z0.h[3]
// CHECK-ENCODING: [0x15,0x1e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1501e15 <unknown>

udot    za.s[w8, 1, vgx2], {z0.h, z1.h}, z14.h[1]  // 11000001-01011110-00010100-00010001
// CHECK-INST: udot    za.s[w8, 1, vgx2], { z0.h, z1.h }, z14.h[1]
// CHECK-ENCODING: [0x11,0x14,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e1411 <unknown>

udot    za.s[w8, 1], {z0.h, z1.h}, z14.h[1]  // 11000001-01011110-00010100-00010001
// CHECK-INST: udot    za.s[w8, 1, vgx2], { z0.h, z1.h }, z14.h[1]
// CHECK-ENCODING: [0x11,0x14,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e1411 <unknown>

udot    za.s[w10, 0, vgx2], {z18.h, z19.h}, z4.h[1]  // 11000001-01010100-01010110-01010000
// CHECK-INST: udot    za.s[w10, 0, vgx2], { z18.h, z19.h }, z4.h[1]
// CHECK-ENCODING: [0x50,0x56,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1545650 <unknown>

udot    za.s[w10, 0], {z18.h, z19.h}, z4.h[1]  // 11000001-01010100-01010110-01010000
// CHECK-INST: udot    za.s[w10, 0, vgx2], { z18.h, z19.h }, z4.h[1]
// CHECK-ENCODING: [0x50,0x56,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1545650 <unknown>

udot    za.s[w8, 0, vgx2], {z12.h, z13.h}, z2.h[2]  // 11000001-01010010-00011001-10010000
// CHECK-INST: udot    za.s[w8, 0, vgx2], { z12.h, z13.h }, z2.h[2]
// CHECK-ENCODING: [0x90,0x19,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1521990 <unknown>

udot    za.s[w8, 0], {z12.h, z13.h}, z2.h[2]  // 11000001-01010010-00011001-10010000
// CHECK-INST: udot    za.s[w8, 0, vgx2], { z12.h, z13.h }, z2.h[2]
// CHECK-ENCODING: [0x90,0x19,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1521990 <unknown>

udot    za.s[w10, 1, vgx2], {z0.h, z1.h}, z10.h[2]  // 11000001-01011010-01011000-00010001
// CHECK-INST: udot    za.s[w10, 1, vgx2], { z0.h, z1.h }, z10.h[2]
// CHECK-ENCODING: [0x11,0x58,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15a5811 <unknown>

udot    za.s[w10, 1], {z0.h, z1.h}, z10.h[2]  // 11000001-01011010-01011000-00010001
// CHECK-INST: udot    za.s[w10, 1, vgx2], { z0.h, z1.h }, z10.h[2]
// CHECK-ENCODING: [0x11,0x58,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15a5811 <unknown>

udot    za.s[w8, 5, vgx2], {z22.h, z23.h}, z14.h[2]  // 11000001-01011110-00011010-11010101
// CHECK-INST: udot    za.s[w8, 5, vgx2], { z22.h, z23.h }, z14.h[2]
// CHECK-ENCODING: [0xd5,0x1a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e1ad5 <unknown>

udot    za.s[w8, 5], {z22.h, z23.h}, z14.h[2]  // 11000001-01011110-00011010-11010101
// CHECK-INST: udot    za.s[w8, 5, vgx2], { z22.h, z23.h }, z14.h[2]
// CHECK-ENCODING: [0xd5,0x1a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e1ad5 <unknown>

udot    za.s[w11, 2, vgx2], {z8.h, z9.h}, z1.h[1]  // 11000001-01010001-01110101-00010010
// CHECK-INST: udot    za.s[w11, 2, vgx2], { z8.h, z9.h }, z1.h[1]
// CHECK-ENCODING: [0x12,0x75,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1517512 <unknown>

udot    za.s[w11, 2], {z8.h, z9.h}, z1.h[1]  // 11000001-01010001-01110101-00010010
// CHECK-INST: udot    za.s[w11, 2, vgx2], { z8.h, z9.h }, z1.h[1]
// CHECK-ENCODING: [0x12,0x75,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1517512 <unknown>

udot    za.s[w9, 7, vgx2], {z12.h, z13.h}, z11.h[2]  // 11000001-01011011-00111001-10010111
// CHECK-INST: udot    za.s[w9, 7, vgx2], { z12.h, z13.h }, z11.h[2]
// CHECK-ENCODING: [0x97,0x39,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15b3997 <unknown>

udot    za.s[w9, 7], {z12.h, z13.h}, z11.h[2]  // 11000001-01011011-00111001-10010111
// CHECK-INST: udot    za.s[w9, 7, vgx2], { z12.h, z13.h }, z11.h[2]
// CHECK-ENCODING: [0x97,0x39,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15b3997 <unknown>


udot    za.s[w8, 0, vgx2], {z0.b, z1.b}, z0.b[0]  // 11000001-01010000-00010000-00110000
// CHECK-INST: udot    za.s[w8, 0, vgx2], { z0.b, z1.b }, z0.b[0]
// CHECK-ENCODING: [0x30,0x10,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1501030 <unknown>

udot    za.s[w8, 0], {z0.b, z1.b}, z0.b[0]  // 11000001-01010000-00010000-00110000
// CHECK-INST: udot    za.s[w8, 0, vgx2], { z0.b, z1.b }, z0.b[0]
// CHECK-ENCODING: [0x30,0x10,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1501030 <unknown>

udot    za.s[w10, 5, vgx2], {z10.b, z11.b}, z5.b[1]  // 11000001-01010101-01010101-01110101
// CHECK-INST: udot    za.s[w10, 5, vgx2], { z10.b, z11.b }, z5.b[1]
// CHECK-ENCODING: [0x75,0x55,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1555575 <unknown>

udot    za.s[w10, 5], {z10.b, z11.b}, z5.b[1]  // 11000001-01010101-01010101-01110101
// CHECK-INST: udot    za.s[w10, 5, vgx2], { z10.b, z11.b }, z5.b[1]
// CHECK-ENCODING: [0x75,0x55,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1555575 <unknown>

udot    za.s[w11, 7, vgx2], {z12.b, z13.b}, z8.b[3]  // 11000001-01011000-01111101-10110111
// CHECK-INST: udot    za.s[w11, 7, vgx2], { z12.b, z13.b }, z8.b[3]
// CHECK-ENCODING: [0xb7,0x7d,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1587db7 <unknown>

udot    za.s[w11, 7], {z12.b, z13.b}, z8.b[3]  // 11000001-01011000-01111101-10110111
// CHECK-INST: udot    za.s[w11, 7, vgx2], { z12.b, z13.b }, z8.b[3]
// CHECK-ENCODING: [0xb7,0x7d,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1587db7 <unknown>

udot    za.s[w11, 7, vgx2], {z30.b, z31.b}, z15.b[3]  // 11000001-01011111-01111111-11110111
// CHECK-INST: udot    za.s[w11, 7, vgx2], { z30.b, z31.b }, z15.b[3]
// CHECK-ENCODING: [0xf7,0x7f,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15f7ff7 <unknown>

udot    za.s[w11, 7], {z30.b, z31.b}, z15.b[3]  // 11000001-01011111-01111111-11110111
// CHECK-INST: udot    za.s[w11, 7, vgx2], { z30.b, z31.b }, z15.b[3]
// CHECK-ENCODING: [0xf7,0x7f,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15f7ff7 <unknown>

udot    za.s[w8, 5, vgx2], {z16.b, z17.b}, z0.b[3]  // 11000001-01010000-00011110-00110101
// CHECK-INST: udot    za.s[w8, 5, vgx2], { z16.b, z17.b }, z0.b[3]
// CHECK-ENCODING: [0x35,0x1e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1501e35 <unknown>

udot    za.s[w8, 5], {z16.b, z17.b}, z0.b[3]  // 11000001-01010000-00011110-00110101
// CHECK-INST: udot    za.s[w8, 5, vgx2], { z16.b, z17.b }, z0.b[3]
// CHECK-ENCODING: [0x35,0x1e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1501e35 <unknown>

udot    za.s[w8, 1, vgx2], {z0.b, z1.b}, z14.b[1]  // 11000001-01011110-00010100-00110001
// CHECK-INST: udot    za.s[w8, 1, vgx2], { z0.b, z1.b }, z14.b[1]
// CHECK-ENCODING: [0x31,0x14,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e1431 <unknown>

udot    za.s[w8, 1], {z0.b, z1.b}, z14.b[1]  // 11000001-01011110-00010100-00110001
// CHECK-INST: udot    za.s[w8, 1, vgx2], { z0.b, z1.b }, z14.b[1]
// CHECK-ENCODING: [0x31,0x14,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e1431 <unknown>

udot    za.s[w10, 0, vgx2], {z18.b, z19.b}, z4.b[1]  // 11000001-01010100-01010110-01110000
// CHECK-INST: udot    za.s[w10, 0, vgx2], { z18.b, z19.b }, z4.b[1]
// CHECK-ENCODING: [0x70,0x56,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1545670 <unknown>

udot    za.s[w10, 0], {z18.b, z19.b}, z4.b[1]  // 11000001-01010100-01010110-01110000
// CHECK-INST: udot    za.s[w10, 0, vgx2], { z18.b, z19.b }, z4.b[1]
// CHECK-ENCODING: [0x70,0x56,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1545670 <unknown>

udot    za.s[w8, 0, vgx2], {z12.b, z13.b}, z2.b[2]  // 11000001-01010010-00011001-10110000
// CHECK-INST: udot    za.s[w8, 0, vgx2], { z12.b, z13.b }, z2.b[2]
// CHECK-ENCODING: [0xb0,0x19,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15219b0 <unknown>

udot    za.s[w8, 0], {z12.b, z13.b}, z2.b[2]  // 11000001-01010010-00011001-10110000
// CHECK-INST: udot    za.s[w8, 0, vgx2], { z12.b, z13.b }, z2.b[2]
// CHECK-ENCODING: [0xb0,0x19,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15219b0 <unknown>

udot    za.s[w10, 1, vgx2], {z0.b, z1.b}, z10.b[2]  // 11000001-01011010-01011000-00110001
// CHECK-INST: udot    za.s[w10, 1, vgx2], { z0.b, z1.b }, z10.b[2]
// CHECK-ENCODING: [0x31,0x58,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15a5831 <unknown>

udot    za.s[w10, 1], {z0.b, z1.b}, z10.b[2]  // 11000001-01011010-01011000-00110001
// CHECK-INST: udot    za.s[w10, 1, vgx2], { z0.b, z1.b }, z10.b[2]
// CHECK-ENCODING: [0x31,0x58,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15a5831 <unknown>

udot    za.s[w8, 5, vgx2], {z22.b, z23.b}, z14.b[2]  // 11000001-01011110-00011010-11110101
// CHECK-INST: udot    za.s[w8, 5, vgx2], { z22.b, z23.b }, z14.b[2]
// CHECK-ENCODING: [0xf5,0x1a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e1af5 <unknown>

udot    za.s[w8, 5], {z22.b, z23.b}, z14.b[2]  // 11000001-01011110-00011010-11110101
// CHECK-INST: udot    za.s[w8, 5, vgx2], { z22.b, z23.b }, z14.b[2]
// CHECK-ENCODING: [0xf5,0x1a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e1af5 <unknown>

udot    za.s[w11, 2, vgx2], {z8.b, z9.b}, z1.b[1]  // 11000001-01010001-01110101-00110010
// CHECK-INST: udot    za.s[w11, 2, vgx2], { z8.b, z9.b }, z1.b[1]
// CHECK-ENCODING: [0x32,0x75,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1517532 <unknown>

udot    za.s[w11, 2], {z8.b, z9.b}, z1.b[1]  // 11000001-01010001-01110101-00110010
// CHECK-INST: udot    za.s[w11, 2, vgx2], { z8.b, z9.b }, z1.b[1]
// CHECK-ENCODING: [0x32,0x75,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1517532 <unknown>

udot    za.s[w9, 7, vgx2], {z12.b, z13.b}, z11.b[2]  // 11000001-01011011-00111001-10110111
// CHECK-INST: udot    za.s[w9, 7, vgx2], { z12.b, z13.b }, z11.b[2]
// CHECK-ENCODING: [0xb7,0x39,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15b39b7 <unknown>

udot    za.s[w9, 7], {z12.b, z13.b}, z11.b[2]  // 11000001-01011011-00111001-10110111
// CHECK-INST: udot    za.s[w9, 7, vgx2], { z12.b, z13.b }, z11.b[2]
// CHECK-ENCODING: [0xb7,0x39,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15b39b7 <unknown>


udot    za.d[w8, 0, vgx2], {z0.h, z1.h}, z0.h[0]  // 11000001-11010000-00000000-00011000
// CHECK-INST: udot    za.d[w8, 0, vgx2], { z0.h, z1.h }, z0.h[0]
// CHECK-ENCODING: [0x18,0x00,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d00018 <unknown>

udot    za.d[w8, 0], {z0.h, z1.h}, z0.h[0]  // 11000001-11010000-00000000-00011000
// CHECK-INST: udot    za.d[w8, 0, vgx2], { z0.h, z1.h }, z0.h[0]
// CHECK-ENCODING: [0x18,0x00,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d00018 <unknown>

udot    za.d[w10, 5, vgx2], {z10.h, z11.h}, z5.h[1]  // 11000001-11010101-01000101-01011101
// CHECK-INST: udot    za.d[w10, 5, vgx2], { z10.h, z11.h }, z5.h[1]
// CHECK-ENCODING: [0x5d,0x45,0xd5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d5455d <unknown>

udot    za.d[w10, 5], {z10.h, z11.h}, z5.h[1]  // 11000001-11010101-01000101-01011101
// CHECK-INST: udot    za.d[w10, 5, vgx2], { z10.h, z11.h }, z5.h[1]
// CHECK-ENCODING: [0x5d,0x45,0xd5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d5455d <unknown>

udot    za.d[w11, 7, vgx2], {z12.h, z13.h}, z8.h[1]  // 11000001-11011000-01100101-10011111
// CHECK-INST: udot    za.d[w11, 7, vgx2], { z12.h, z13.h }, z8.h[1]
// CHECK-ENCODING: [0x9f,0x65,0xd8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d8659f <unknown>

udot    za.d[w11, 7], {z12.h, z13.h}, z8.h[1]  // 11000001-11011000-01100101-10011111
// CHECK-INST: udot    za.d[w11, 7, vgx2], { z12.h, z13.h }, z8.h[1]
// CHECK-ENCODING: [0x9f,0x65,0xd8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d8659f <unknown>

udot    za.d[w11, 7, vgx2], {z30.h, z31.h}, z15.h[1]  // 11000001-11011111-01100111-11011111
// CHECK-INST: udot    za.d[w11, 7, vgx2], { z30.h, z31.h }, z15.h[1]
// CHECK-ENCODING: [0xdf,0x67,0xdf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1df67df <unknown>

udot    za.d[w11, 7], {z30.h, z31.h}, z15.h[1]  // 11000001-11011111-01100111-11011111
// CHECK-INST: udot    za.d[w11, 7, vgx2], { z30.h, z31.h }, z15.h[1]
// CHECK-ENCODING: [0xdf,0x67,0xdf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1df67df <unknown>

udot    za.d[w8, 5, vgx2], {z16.h, z17.h}, z0.h[1]  // 11000001-11010000-00000110-00011101
// CHECK-INST: udot    za.d[w8, 5, vgx2], { z16.h, z17.h }, z0.h[1]
// CHECK-ENCODING: [0x1d,0x06,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d0061d <unknown>

udot    za.d[w8, 5], {z16.h, z17.h}, z0.h[1]  // 11000001-11010000-00000110-00011101
// CHECK-INST: udot    za.d[w8, 5, vgx2], { z16.h, z17.h }, z0.h[1]
// CHECK-ENCODING: [0x1d,0x06,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d0061d <unknown>

udot    za.d[w8, 1, vgx2], {z0.h, z1.h}, z14.h[1]  // 11000001-11011110-00000100-00011001
// CHECK-INST: udot    za.d[w8, 1, vgx2], { z0.h, z1.h }, z14.h[1]
// CHECK-ENCODING: [0x19,0x04,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de0419 <unknown>

udot    za.d[w8, 1], {z0.h, z1.h}, z14.h[1]  // 11000001-11011110-00000100-00011001
// CHECK-INST: udot    za.d[w8, 1, vgx2], { z0.h, z1.h }, z14.h[1]
// CHECK-ENCODING: [0x19,0x04,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de0419 <unknown>

udot    za.d[w10, 0, vgx2], {z18.h, z19.h}, z4.h[1]  // 11000001-11010100-01000110-01011000
// CHECK-INST: udot    za.d[w10, 0, vgx2], { z18.h, z19.h }, z4.h[1]
// CHECK-ENCODING: [0x58,0x46,0xd4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d44658 <unknown>

udot    za.d[w10, 0], {z18.h, z19.h}, z4.h[1]  // 11000001-11010100-01000110-01011000
// CHECK-INST: udot    za.d[w10, 0, vgx2], { z18.h, z19.h }, z4.h[1]
// CHECK-ENCODING: [0x58,0x46,0xd4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d44658 <unknown>

udot    za.d[w8, 0, vgx2], {z12.h, z13.h}, z2.h[0]  // 11000001-11010010-00000001-10011000
// CHECK-INST: udot    za.d[w8, 0, vgx2], { z12.h, z13.h }, z2.h[0]
// CHECK-ENCODING: [0x98,0x01,0xd2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d20198 <unknown>

udot    za.d[w8, 0], {z12.h, z13.h}, z2.h[0]  // 11000001-11010010-00000001-10011000
// CHECK-INST: udot    za.d[w8, 0, vgx2], { z12.h, z13.h }, z2.h[0]
// CHECK-ENCODING: [0x98,0x01,0xd2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d20198 <unknown>

udot    za.d[w10, 1, vgx2], {z0.h, z1.h}, z10.h[0]  // 11000001-11011010-01000000-00011001
// CHECK-INST: udot    za.d[w10, 1, vgx2], { z0.h, z1.h }, z10.h[0]
// CHECK-ENCODING: [0x19,0x40,0xda,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1da4019 <unknown>

udot    za.d[w10, 1], {z0.h, z1.h}, z10.h[0]  // 11000001-11011010-01000000-00011001
// CHECK-INST: udot    za.d[w10, 1, vgx2], { z0.h, z1.h }, z10.h[0]
// CHECK-ENCODING: [0x19,0x40,0xda,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1da4019 <unknown>

udot    za.d[w8, 5, vgx2], {z22.h, z23.h}, z14.h[0]  // 11000001-11011110-00000010-11011101
// CHECK-INST: udot    za.d[w8, 5, vgx2], { z22.h, z23.h }, z14.h[0]
// CHECK-ENCODING: [0xdd,0x02,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de02dd <unknown>

udot    za.d[w8, 5], {z22.h, z23.h}, z14.h[0]  // 11000001-11011110-00000010-11011101
// CHECK-INST: udot    za.d[w8, 5, vgx2], { z22.h, z23.h }, z14.h[0]
// CHECK-ENCODING: [0xdd,0x02,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de02dd <unknown>

udot    za.d[w11, 2, vgx2], {z8.h, z9.h}, z1.h[1]  // 11000001-11010001-01100101-00011010
// CHECK-INST: udot    za.d[w11, 2, vgx2], { z8.h, z9.h }, z1.h[1]
// CHECK-ENCODING: [0x1a,0x65,0xd1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d1651a <unknown>

udot    za.d[w11, 2], {z8.h, z9.h}, z1.h[1]  // 11000001-11010001-01100101-00011010
// CHECK-INST: udot    za.d[w11, 2, vgx2], { z8.h, z9.h }, z1.h[1]
// CHECK-ENCODING: [0x1a,0x65,0xd1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d1651a <unknown>

udot    za.d[w9, 7, vgx2], {z12.h, z13.h}, z11.h[0]  // 11000001-11011011-00100001-10011111
// CHECK-INST: udot    za.d[w9, 7, vgx2], { z12.h, z13.h }, z11.h[0]
// CHECK-ENCODING: [0x9f,0x21,0xdb,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1db219f <unknown>

udot    za.d[w9, 7], {z12.h, z13.h}, z11.h[0]  // 11000001-11011011-00100001-10011111
// CHECK-INST: udot    za.d[w9, 7, vgx2], { z12.h, z13.h }, z11.h[0]
// CHECK-ENCODING: [0x9f,0x21,0xdb,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1db219f <unknown>


udot    za.s[w8, 0, vgx4], {z0.h - z3.h}, z0.h[0]  // 11000001-01010000-10010000-00010000
// CHECK-INST: udot    za.s[w8, 0, vgx4], { z0.h - z3.h }, z0.h[0]
// CHECK-ENCODING: [0x10,0x90,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1509010 <unknown>

udot    za.s[w8, 0], {z0.h - z3.h}, z0.h[0]  // 11000001-01010000-10010000-00010000
// CHECK-INST: udot    za.s[w8, 0, vgx4], { z0.h - z3.h }, z0.h[0]
// CHECK-ENCODING: [0x10,0x90,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1509010 <unknown>

udot    za.s[w10, 5, vgx4], {z8.h - z11.h}, z5.h[1]  // 11000001-01010101-11010101-00010101
// CHECK-INST: udot    za.s[w10, 5, vgx4], { z8.h - z11.h }, z5.h[1]
// CHECK-ENCODING: [0x15,0xd5,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c155d515 <unknown>

udot    za.s[w10, 5], {z8.h - z11.h}, z5.h[1]  // 11000001-01010101-11010101-00010101
// CHECK-INST: udot    za.s[w10, 5, vgx4], { z8.h - z11.h }, z5.h[1]
// CHECK-ENCODING: [0x15,0xd5,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c155d515 <unknown>

udot    za.s[w11, 7, vgx4], {z12.h - z15.h}, z8.h[3]  // 11000001-01011000-11111101-10010111
// CHECK-INST: udot    za.s[w11, 7, vgx4], { z12.h - z15.h }, z8.h[3]
// CHECK-ENCODING: [0x97,0xfd,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c158fd97 <unknown>

udot    za.s[w11, 7], {z12.h - z15.h}, z8.h[3]  // 11000001-01011000-11111101-10010111
// CHECK-INST: udot    za.s[w11, 7, vgx4], { z12.h - z15.h }, z8.h[3]
// CHECK-ENCODING: [0x97,0xfd,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c158fd97 <unknown>

udot    za.s[w11, 7, vgx4], {z28.h - z31.h}, z15.h[3]  // 11000001-01011111-11111111-10010111
// CHECK-INST: udot    za.s[w11, 7, vgx4], { z28.h - z31.h }, z15.h[3]
// CHECK-ENCODING: [0x97,0xff,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15fff97 <unknown>

udot    za.s[w11, 7], {z28.h - z31.h}, z15.h[3]  // 11000001-01011111-11111111-10010111
// CHECK-INST: udot    za.s[w11, 7, vgx4], { z28.h - z31.h }, z15.h[3]
// CHECK-ENCODING: [0x97,0xff,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15fff97 <unknown>

udot    za.s[w8, 5, vgx4], {z16.h - z19.h}, z0.h[3]  // 11000001-01010000-10011110-00010101
// CHECK-INST: udot    za.s[w8, 5, vgx4], { z16.h - z19.h }, z0.h[3]
// CHECK-ENCODING: [0x15,0x9e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1509e15 <unknown>

udot    za.s[w8, 5], {z16.h - z19.h}, z0.h[3]  // 11000001-01010000-10011110-00010101
// CHECK-INST: udot    za.s[w8, 5, vgx4], { z16.h - z19.h }, z0.h[3]
// CHECK-ENCODING: [0x15,0x9e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1509e15 <unknown>

udot    za.s[w8, 1, vgx4], {z0.h - z3.h}, z14.h[1]  // 11000001-01011110-10010100-00010001
// CHECK-INST: udot    za.s[w8, 1, vgx4], { z0.h - z3.h }, z14.h[1]
// CHECK-ENCODING: [0x11,0x94,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e9411 <unknown>

udot    za.s[w8, 1], {z0.h - z3.h}, z14.h[1]  // 11000001-01011110-10010100-00010001
// CHECK-INST: udot    za.s[w8, 1, vgx4], { z0.h - z3.h }, z14.h[1]
// CHECK-ENCODING: [0x11,0x94,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e9411 <unknown>

udot    za.s[w10, 0, vgx4], {z16.h - z19.h}, z4.h[1]  // 11000001-01010100-11010110-00010000
// CHECK-INST: udot    za.s[w10, 0, vgx4], { z16.h - z19.h }, z4.h[1]
// CHECK-ENCODING: [0x10,0xd6,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c154d610 <unknown>

udot    za.s[w10, 0], {z16.h - z19.h}, z4.h[1]  // 11000001-01010100-11010110-00010000
// CHECK-INST: udot    za.s[w10, 0, vgx4], { z16.h - z19.h }, z4.h[1]
// CHECK-ENCODING: [0x10,0xd6,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c154d610 <unknown>

udot    za.s[w8, 0, vgx4], {z12.h - z15.h}, z2.h[2]  // 11000001-01010010-10011001-10010000
// CHECK-INST: udot    za.s[w8, 0, vgx4], { z12.h - z15.h }, z2.h[2]
// CHECK-ENCODING: [0x90,0x99,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1529990 <unknown>

udot    za.s[w8, 0], {z12.h - z15.h}, z2.h[2]  // 11000001-01010010-10011001-10010000
// CHECK-INST: udot    za.s[w8, 0, vgx4], { z12.h - z15.h }, z2.h[2]
// CHECK-ENCODING: [0x90,0x99,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1529990 <unknown>

udot    za.s[w10, 1, vgx4], {z0.h - z3.h}, z10.h[2]  // 11000001-01011010-11011000-00010001
// CHECK-INST: udot    za.s[w10, 1, vgx4], { z0.h - z3.h }, z10.h[2]
// CHECK-ENCODING: [0x11,0xd8,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15ad811 <unknown>

udot    za.s[w10, 1], {z0.h - z3.h}, z10.h[2]  // 11000001-01011010-11011000-00010001
// CHECK-INST: udot    za.s[w10, 1, vgx4], { z0.h - z3.h }, z10.h[2]
// CHECK-ENCODING: [0x11,0xd8,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15ad811 <unknown>

udot    za.s[w8, 5, vgx4], {z20.h - z23.h}, z14.h[2]  // 11000001-01011110-10011010-10010101
// CHECK-INST: udot    za.s[w8, 5, vgx4], { z20.h - z23.h }, z14.h[2]
// CHECK-ENCODING: [0x95,0x9a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e9a95 <unknown>

udot    za.s[w8, 5], {z20.h - z23.h}, z14.h[2]  // 11000001-01011110-10011010-10010101
// CHECK-INST: udot    za.s[w8, 5, vgx4], { z20.h - z23.h }, z14.h[2]
// CHECK-ENCODING: [0x95,0x9a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e9a95 <unknown>

udot    za.s[w11, 2, vgx4], {z8.h - z11.h}, z1.h[1]  // 11000001-01010001-11110101-00010010
// CHECK-INST: udot    za.s[w11, 2, vgx4], { z8.h - z11.h }, z1.h[1]
// CHECK-ENCODING: [0x12,0xf5,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c151f512 <unknown>

udot    za.s[w11, 2], {z8.h - z11.h}, z1.h[1]  // 11000001-01010001-11110101-00010010
// CHECK-INST: udot    za.s[w11, 2, vgx4], { z8.h - z11.h }, z1.h[1]
// CHECK-ENCODING: [0x12,0xf5,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c151f512 <unknown>

udot    za.s[w9, 7, vgx4], {z12.h - z15.h}, z11.h[2]  // 11000001-01011011-10111001-10010111
// CHECK-INST: udot    za.s[w9, 7, vgx4], { z12.h - z15.h }, z11.h[2]
// CHECK-ENCODING: [0x97,0xb9,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15bb997 <unknown>

udot    za.s[w9, 7], {z12.h - z15.h}, z11.h[2]  // 11000001-01011011-10111001-10010111
// CHECK-INST: udot    za.s[w9, 7, vgx4], { z12.h - z15.h }, z11.h[2]
// CHECK-ENCODING: [0x97,0xb9,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15bb997 <unknown>


udot    za.s[w8, 0, vgx4], {z0.b - z3.b}, z0.b[0]  // 11000001-01010000-10010000-00110000
// CHECK-INST: udot    za.s[w8, 0, vgx4], { z0.b - z3.b }, z0.b[0]
// CHECK-ENCODING: [0x30,0x90,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1509030 <unknown>

udot    za.s[w8, 0], {z0.b - z3.b}, z0.b[0]  // 11000001-01010000-10010000-00110000
// CHECK-INST: udot    za.s[w8, 0, vgx4], { z0.b - z3.b }, z0.b[0]
// CHECK-ENCODING: [0x30,0x90,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1509030 <unknown>

udot    za.s[w10, 5, vgx4], {z8.b - z11.b}, z5.b[1]  // 11000001-01010101-11010101-00110101
// CHECK-INST: udot    za.s[w10, 5, vgx4], { z8.b - z11.b }, z5.b[1]
// CHECK-ENCODING: [0x35,0xd5,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c155d535 <unknown>

udot    za.s[w10, 5], {z8.b - z11.b}, z5.b[1]  // 11000001-01010101-11010101-00110101
// CHECK-INST: udot    za.s[w10, 5, vgx4], { z8.b - z11.b }, z5.b[1]
// CHECK-ENCODING: [0x35,0xd5,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c155d535 <unknown>

udot    za.s[w11, 7, vgx4], {z12.b - z15.b}, z8.b[3]  // 11000001-01011000-11111101-10110111
// CHECK-INST: udot    za.s[w11, 7, vgx4], { z12.b - z15.b }, z8.b[3]
// CHECK-ENCODING: [0xb7,0xfd,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c158fdb7 <unknown>

udot    za.s[w11, 7], {z12.b - z15.b}, z8.b[3]  // 11000001-01011000-11111101-10110111
// CHECK-INST: udot    za.s[w11, 7, vgx4], { z12.b - z15.b }, z8.b[3]
// CHECK-ENCODING: [0xb7,0xfd,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c158fdb7 <unknown>

udot    za.s[w11, 7, vgx4], {z28.b - z31.b}, z15.b[3]  // 11000001-01011111-11111111-10110111
// CHECK-INST: udot    za.s[w11, 7, vgx4], { z28.b - z31.b }, z15.b[3]
// CHECK-ENCODING: [0xb7,0xff,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15fffb7 <unknown>

udot    za.s[w11, 7], {z28.b - z31.b}, z15.b[3]  // 11000001-01011111-11111111-10110111
// CHECK-INST: udot    za.s[w11, 7, vgx4], { z28.b - z31.b }, z15.b[3]
// CHECK-ENCODING: [0xb7,0xff,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15fffb7 <unknown>

udot    za.s[w8, 5, vgx4], {z16.b - z19.b}, z0.b[3]  // 11000001-01010000-10011110-00110101
// CHECK-INST: udot    za.s[w8, 5, vgx4], { z16.b - z19.b }, z0.b[3]
// CHECK-ENCODING: [0x35,0x9e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1509e35 <unknown>

udot    za.s[w8, 5], {z16.b - z19.b}, z0.b[3]  // 11000001-01010000-10011110-00110101
// CHECK-INST: udot    za.s[w8, 5, vgx4], { z16.b - z19.b }, z0.b[3]
// CHECK-ENCODING: [0x35,0x9e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1509e35 <unknown>

udot    za.s[w8, 1, vgx4], {z0.b - z3.b}, z14.b[1]  // 11000001-01011110-10010100-00110001
// CHECK-INST: udot    za.s[w8, 1, vgx4], { z0.b - z3.b }, z14.b[1]
// CHECK-ENCODING: [0x31,0x94,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e9431 <unknown>

udot    za.s[w8, 1], {z0.b - z3.b}, z14.b[1]  // 11000001-01011110-10010100-00110001
// CHECK-INST: udot    za.s[w8, 1, vgx4], { z0.b - z3.b }, z14.b[1]
// CHECK-ENCODING: [0x31,0x94,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e9431 <unknown>

udot    za.s[w10, 0, vgx4], {z16.b - z19.b}, z4.b[1]  // 11000001-01010100-11010110-00110000
// CHECK-INST: udot    za.s[w10, 0, vgx4], { z16.b - z19.b }, z4.b[1]
// CHECK-ENCODING: [0x30,0xd6,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c154d630 <unknown>

udot    za.s[w10, 0], {z16.b - z19.b}, z4.b[1]  // 11000001-01010100-11010110-00110000
// CHECK-INST: udot    za.s[w10, 0, vgx4], { z16.b - z19.b }, z4.b[1]
// CHECK-ENCODING: [0x30,0xd6,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c154d630 <unknown>

udot    za.s[w8, 0, vgx4], {z12.b - z15.b}, z2.b[2]  // 11000001-01010010-10011001-10110000
// CHECK-INST: udot    za.s[w8, 0, vgx4], { z12.b - z15.b }, z2.b[2]
// CHECK-ENCODING: [0xb0,0x99,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15299b0 <unknown>

udot    za.s[w8, 0], {z12.b - z15.b}, z2.b[2]  // 11000001-01010010-10011001-10110000
// CHECK-INST: udot    za.s[w8, 0, vgx4], { z12.b - z15.b }, z2.b[2]
// CHECK-ENCODING: [0xb0,0x99,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15299b0 <unknown>

udot    za.s[w10, 1, vgx4], {z0.b - z3.b}, z10.b[2]  // 11000001-01011010-11011000-00110001
// CHECK-INST: udot    za.s[w10, 1, vgx4], { z0.b - z3.b }, z10.b[2]
// CHECK-ENCODING: [0x31,0xd8,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15ad831 <unknown>

udot    za.s[w10, 1], {z0.b - z3.b}, z10.b[2]  // 11000001-01011010-11011000-00110001
// CHECK-INST: udot    za.s[w10, 1, vgx4], { z0.b - z3.b }, z10.b[2]
// CHECK-ENCODING: [0x31,0xd8,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15ad831 <unknown>

udot    za.s[w8, 5, vgx4], {z20.b - z23.b}, z14.b[2]  // 11000001-01011110-10011010-10110101
// CHECK-INST: udot    za.s[w8, 5, vgx4], { z20.b - z23.b }, z14.b[2]
// CHECK-ENCODING: [0xb5,0x9a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e9ab5 <unknown>

udot    za.s[w8, 5], {z20.b - z23.b}, z14.b[2]  // 11000001-01011110-10011010-10110101
// CHECK-INST: udot    za.s[w8, 5, vgx4], { z20.b - z23.b }, z14.b[2]
// CHECK-ENCODING: [0xb5,0x9a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e9ab5 <unknown>

udot    za.s[w11, 2, vgx4], {z8.b - z11.b}, z1.b[1]  // 11000001-01010001-11110101-00110010
// CHECK-INST: udot    za.s[w11, 2, vgx4], { z8.b - z11.b }, z1.b[1]
// CHECK-ENCODING: [0x32,0xf5,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c151f532 <unknown>

udot    za.s[w11, 2], {z8.b - z11.b}, z1.b[1]  // 11000001-01010001-11110101-00110010
// CHECK-INST: udot    za.s[w11, 2, vgx4], { z8.b - z11.b }, z1.b[1]
// CHECK-ENCODING: [0x32,0xf5,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c151f532 <unknown>

udot    za.s[w9, 7, vgx4], {z12.b - z15.b}, z11.b[2]  // 11000001-01011011-10111001-10110111
// CHECK-INST: udot    za.s[w9, 7, vgx4], { z12.b - z15.b }, z11.b[2]
// CHECK-ENCODING: [0xb7,0xb9,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15bb9b7 <unknown>

udot    za.s[w9, 7], {z12.b - z15.b}, z11.b[2]  // 11000001-01011011-10111001-10110111
// CHECK-INST: udot    za.s[w9, 7, vgx4], { z12.b - z15.b }, z11.b[2]
// CHECK-ENCODING: [0xb7,0xb9,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15bb9b7 <unknown>


udot    za.d[w8, 0, vgx4], {z0.h - z3.h}, z0.h[0]  // 11000001-11010000-10000000-00011000
// CHECK-INST: udot    za.d[w8, 0, vgx4], { z0.h - z3.h }, z0.h[0]
// CHECK-ENCODING: [0x18,0x80,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d08018 <unknown>

udot    za.d[w8, 0], {z0.h - z3.h}, z0.h[0]  // 11000001-11010000-10000000-00011000
// CHECK-INST: udot    za.d[w8, 0, vgx4], { z0.h - z3.h }, z0.h[0]
// CHECK-ENCODING: [0x18,0x80,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d08018 <unknown>

udot    za.d[w10, 5, vgx4], {z8.h - z11.h}, z5.h[1]  // 11000001-11010101-11000101-00011101
// CHECK-INST: udot    za.d[w10, 5, vgx4], { z8.h - z11.h }, z5.h[1]
// CHECK-ENCODING: [0x1d,0xc5,0xd5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d5c51d <unknown>

udot    za.d[w10, 5], {z8.h - z11.h}, z5.h[1]  // 11000001-11010101-11000101-00011101
// CHECK-INST: udot    za.d[w10, 5, vgx4], { z8.h - z11.h }, z5.h[1]
// CHECK-ENCODING: [0x1d,0xc5,0xd5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d5c51d <unknown>

udot    za.d[w11, 7, vgx4], {z12.h - z15.h}, z8.h[1]  // 11000001-11011000-11100101-10011111
// CHECK-INST: udot    za.d[w11, 7, vgx4], { z12.h - z15.h }, z8.h[1]
// CHECK-ENCODING: [0x9f,0xe5,0xd8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d8e59f <unknown>

udot    za.d[w11, 7], {z12.h - z15.h}, z8.h[1]  // 11000001-11011000-11100101-10011111
// CHECK-INST: udot    za.d[w11, 7, vgx4], { z12.h - z15.h }, z8.h[1]
// CHECK-ENCODING: [0x9f,0xe5,0xd8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d8e59f <unknown>

udot    za.d[w11, 7, vgx4], {z28.h - z31.h}, z15.h[1]  // 11000001-11011111-11100111-10011111
// CHECK-INST: udot    za.d[w11, 7, vgx4], { z28.h - z31.h }, z15.h[1]
// CHECK-ENCODING: [0x9f,0xe7,0xdf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dfe79f <unknown>

udot    za.d[w11, 7], {z28.h - z31.h}, z15.h[1]  // 11000001-11011111-11100111-10011111
// CHECK-INST: udot    za.d[w11, 7, vgx4], { z28.h - z31.h }, z15.h[1]
// CHECK-ENCODING: [0x9f,0xe7,0xdf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dfe79f <unknown>

udot    za.d[w8, 5, vgx4], {z16.h - z19.h}, z0.h[1]  // 11000001-11010000-10000110-00011101
// CHECK-INST: udot    za.d[w8, 5, vgx4], { z16.h - z19.h }, z0.h[1]
// CHECK-ENCODING: [0x1d,0x86,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d0861d <unknown>

udot    za.d[w8, 5], {z16.h - z19.h}, z0.h[1]  // 11000001-11010000-10000110-00011101
// CHECK-INST: udot    za.d[w8, 5, vgx4], { z16.h - z19.h }, z0.h[1]
// CHECK-ENCODING: [0x1d,0x86,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d0861d <unknown>

udot    za.d[w8, 1, vgx4], {z0.h - z3.h}, z14.h[1]  // 11000001-11011110-10000100-00011001
// CHECK-INST: udot    za.d[w8, 1, vgx4], { z0.h - z3.h }, z14.h[1]
// CHECK-ENCODING: [0x19,0x84,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de8419 <unknown>

udot    za.d[w8, 1], {z0.h - z3.h}, z14.h[1]  // 11000001-11011110-10000100-00011001
// CHECK-INST: udot    za.d[w8, 1, vgx4], { z0.h - z3.h }, z14.h[1]
// CHECK-ENCODING: [0x19,0x84,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de8419 <unknown>

udot    za.d[w10, 0, vgx4], {z16.h - z19.h}, z4.h[1]  // 11000001-11010100-11000110-00011000
// CHECK-INST: udot    za.d[w10, 0, vgx4], { z16.h - z19.h }, z4.h[1]
// CHECK-ENCODING: [0x18,0xc6,0xd4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d4c618 <unknown>

udot    za.d[w10, 0], {z16.h - z19.h}, z4.h[1]  // 11000001-11010100-11000110-00011000
// CHECK-INST: udot    za.d[w10, 0, vgx4], { z16.h - z19.h }, z4.h[1]
// CHECK-ENCODING: [0x18,0xc6,0xd4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d4c618 <unknown>

udot    za.d[w8, 0, vgx4], {z12.h - z15.h}, z2.h[0]  // 11000001-11010010-10000001-10011000
// CHECK-INST: udot    za.d[w8, 0, vgx4], { z12.h - z15.h }, z2.h[0]
// CHECK-ENCODING: [0x98,0x81,0xd2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d28198 <unknown>

udot    za.d[w8, 0], {z12.h - z15.h}, z2.h[0]  // 11000001-11010010-10000001-10011000
// CHECK-INST: udot    za.d[w8, 0, vgx4], { z12.h - z15.h }, z2.h[0]
// CHECK-ENCODING: [0x98,0x81,0xd2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d28198 <unknown>

udot    za.d[w10, 1, vgx4], {z0.h - z3.h}, z10.h[0]  // 11000001-11011010-11000000-00011001
// CHECK-INST: udot    za.d[w10, 1, vgx4], { z0.h - z3.h }, z10.h[0]
// CHECK-ENCODING: [0x19,0xc0,0xda,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dac019 <unknown>

udot    za.d[w10, 1], {z0.h - z3.h}, z10.h[0]  // 11000001-11011010-11000000-00011001
// CHECK-INST: udot    za.d[w10, 1, vgx4], { z0.h - z3.h }, z10.h[0]
// CHECK-ENCODING: [0x19,0xc0,0xda,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dac019 <unknown>

udot    za.d[w8, 5, vgx4], {z20.h - z23.h}, z14.h[0]  // 11000001-11011110-10000010-10011101
// CHECK-INST: udot    za.d[w8, 5, vgx4], { z20.h - z23.h }, z14.h[0]
// CHECK-ENCODING: [0x9d,0x82,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de829d <unknown>

udot    za.d[w8, 5], {z20.h - z23.h}, z14.h[0]  // 11000001-11011110-10000010-10011101
// CHECK-INST: udot    za.d[w8, 5, vgx4], { z20.h - z23.h }, z14.h[0]
// CHECK-ENCODING: [0x9d,0x82,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de829d <unknown>

udot    za.d[w11, 2, vgx4], {z8.h - z11.h}, z1.h[1]  // 11000001-11010001-11100101-00011010
// CHECK-INST: udot    za.d[w11, 2, vgx4], { z8.h - z11.h }, z1.h[1]
// CHECK-ENCODING: [0x1a,0xe5,0xd1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d1e51a <unknown>

udot    za.d[w11, 2], {z8.h - z11.h}, z1.h[1]  // 11000001-11010001-11100101-00011010
// CHECK-INST: udot    za.d[w11, 2, vgx4], { z8.h - z11.h }, z1.h[1]
// CHECK-ENCODING: [0x1a,0xe5,0xd1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d1e51a <unknown>

udot    za.d[w9, 7, vgx4], {z12.h - z15.h}, z11.h[0]  // 11000001-11011011-10100001-10011111
// CHECK-INST: udot    za.d[w9, 7, vgx4], { z12.h - z15.h }, z11.h[0]
// CHECK-ENCODING: [0x9f,0xa1,0xdb,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dba19f <unknown>

udot    za.d[w9, 7], {z12.h - z15.h}, z11.h[0]  // 11000001-11011011-10100001-10011111
// CHECK-INST: udot    za.d[w9, 7, vgx4], { z12.h - z15.h }, z11.h[0]
// CHECK-ENCODING: [0x9f,0xa1,0xdb,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dba19f <unknown>

