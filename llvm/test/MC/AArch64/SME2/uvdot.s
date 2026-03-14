// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+sme-i16i64 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2,+sme-i16i64 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2,+sme-i16i64 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2,+sme-i16i64 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+sme-i16i64 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2,+sme-i16i64 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


uvdot   za.s[w8, 0, vgx2], {z0.h, z1.h}, z0.h[0]  // 11000001-01010000-00000000-00110000
// CHECK-INST: uvdot   za.s[w8, 0, vgx2], { z0.h, z1.h }, z0.h[0]
// CHECK-ENCODING: [0x30,0x00,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1500030 <unknown>

uvdot   za.s[w8, 0], {z0.h, z1.h}, z0.h[0]  // 11000001-01010000-00000000-00110000
// CHECK-INST: uvdot   za.s[w8, 0, vgx2], { z0.h, z1.h }, z0.h[0]
// CHECK-ENCODING: [0x30,0x00,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1500030 <unknown>

uvdot   za.s[w10, 5, vgx2], {z10.h, z11.h}, z5.h[1]  // 11000001-01010101-01000101-01110101
// CHECK-INST: uvdot   za.s[w10, 5, vgx2], { z10.h, z11.h }, z5.h[1]
// CHECK-ENCODING: [0x75,0x45,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1554575 <unknown>

uvdot   za.s[w10, 5], {z10.h, z11.h}, z5.h[1]  // 11000001-01010101-01000101-01110101
// CHECK-INST: uvdot   za.s[w10, 5, vgx2], { z10.h, z11.h }, z5.h[1]
// CHECK-ENCODING: [0x75,0x45,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1554575 <unknown>

uvdot   za.s[w11, 7, vgx2], {z12.h, z13.h}, z8.h[3]  // 11000001-01011000-01101101-10110111
// CHECK-INST: uvdot   za.s[w11, 7, vgx2], { z12.h, z13.h }, z8.h[3]
// CHECK-ENCODING: [0xb7,0x6d,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1586db7 <unknown>

uvdot   za.s[w11, 7], {z12.h, z13.h}, z8.h[3]  // 11000001-01011000-01101101-10110111
// CHECK-INST: uvdot   za.s[w11, 7, vgx2], { z12.h, z13.h }, z8.h[3]
// CHECK-ENCODING: [0xb7,0x6d,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1586db7 <unknown>

uvdot   za.s[w11, 7, vgx2], {z30.h, z31.h}, z15.h[3]  // 11000001-01011111-01101111-11110111
// CHECK-INST: uvdot   za.s[w11, 7, vgx2], { z30.h, z31.h }, z15.h[3]
// CHECK-ENCODING: [0xf7,0x6f,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15f6ff7 <unknown>

uvdot   za.s[w11, 7], {z30.h, z31.h}, z15.h[3]  // 11000001-01011111-01101111-11110111
// CHECK-INST: uvdot   za.s[w11, 7, vgx2], { z30.h, z31.h }, z15.h[3]
// CHECK-ENCODING: [0xf7,0x6f,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15f6ff7 <unknown>

uvdot   za.s[w8, 5, vgx2], {z16.h, z17.h}, z0.h[3]  // 11000001-01010000-00001110-00110101
// CHECK-INST: uvdot   za.s[w8, 5, vgx2], { z16.h, z17.h }, z0.h[3]
// CHECK-ENCODING: [0x35,0x0e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1500e35 <unknown>

uvdot   za.s[w8, 5], {z16.h, z17.h}, z0.h[3]  // 11000001-01010000-00001110-00110101
// CHECK-INST: uvdot   za.s[w8, 5, vgx2], { z16.h, z17.h }, z0.h[3]
// CHECK-ENCODING: [0x35,0x0e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1500e35 <unknown>

uvdot   za.s[w8, 1, vgx2], {z0.h, z1.h}, z14.h[1]  // 11000001-01011110-00000100-00110001
// CHECK-INST: uvdot   za.s[w8, 1, vgx2], { z0.h, z1.h }, z14.h[1]
// CHECK-ENCODING: [0x31,0x04,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e0431 <unknown>

uvdot   za.s[w8, 1], {z0.h, z1.h}, z14.h[1]  // 11000001-01011110-00000100-00110001
// CHECK-INST: uvdot   za.s[w8, 1, vgx2], { z0.h, z1.h }, z14.h[1]
// CHECK-ENCODING: [0x31,0x04,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e0431 <unknown>

uvdot   za.s[w10, 0, vgx2], {z18.h, z19.h}, z4.h[1]  // 11000001-01010100-01000110-01110000
// CHECK-INST: uvdot   za.s[w10, 0, vgx2], { z18.h, z19.h }, z4.h[1]
// CHECK-ENCODING: [0x70,0x46,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1544670 <unknown>

uvdot   za.s[w10, 0], {z18.h, z19.h}, z4.h[1]  // 11000001-01010100-01000110-01110000
// CHECK-INST: uvdot   za.s[w10, 0, vgx2], { z18.h, z19.h }, z4.h[1]
// CHECK-ENCODING: [0x70,0x46,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1544670 <unknown>

uvdot   za.s[w8, 0, vgx2], {z12.h, z13.h}, z2.h[2]  // 11000001-01010010-00001001-10110000
// CHECK-INST: uvdot   za.s[w8, 0, vgx2], { z12.h, z13.h }, z2.h[2]
// CHECK-ENCODING: [0xb0,0x09,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15209b0 <unknown>

uvdot   za.s[w8, 0], {z12.h, z13.h}, z2.h[2]  // 11000001-01010010-00001001-10110000
// CHECK-INST: uvdot   za.s[w8, 0, vgx2], { z12.h, z13.h }, z2.h[2]
// CHECK-ENCODING: [0xb0,0x09,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15209b0 <unknown>

uvdot   za.s[w10, 1, vgx2], {z0.h, z1.h}, z10.h[2]  // 11000001-01011010-01001000-00110001
// CHECK-INST: uvdot   za.s[w10, 1, vgx2], { z0.h, z1.h }, z10.h[2]
// CHECK-ENCODING: [0x31,0x48,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15a4831 <unknown>

uvdot   za.s[w10, 1], {z0.h, z1.h}, z10.h[2]  // 11000001-01011010-01001000-00110001
// CHECK-INST: uvdot   za.s[w10, 1, vgx2], { z0.h, z1.h }, z10.h[2]
// CHECK-ENCODING: [0x31,0x48,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15a4831 <unknown>

uvdot   za.s[w8, 5, vgx2], {z22.h, z23.h}, z14.h[2]  // 11000001-01011110-00001010-11110101
// CHECK-INST: uvdot   za.s[w8, 5, vgx2], { z22.h, z23.h }, z14.h[2]
// CHECK-ENCODING: [0xf5,0x0a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e0af5 <unknown>

uvdot   za.s[w8, 5], {z22.h, z23.h}, z14.h[2]  // 11000001-01011110-00001010-11110101
// CHECK-INST: uvdot   za.s[w8, 5, vgx2], { z22.h, z23.h }, z14.h[2]
// CHECK-ENCODING: [0xf5,0x0a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e0af5 <unknown>

uvdot   za.s[w11, 2, vgx2], {z8.h, z9.h}, z1.h[1]  // 11000001-01010001-01100101-00110010
// CHECK-INST: uvdot   za.s[w11, 2, vgx2], { z8.h, z9.h }, z1.h[1]
// CHECK-ENCODING: [0x32,0x65,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1516532 <unknown>

uvdot   za.s[w11, 2], {z8.h, z9.h}, z1.h[1]  // 11000001-01010001-01100101-00110010
// CHECK-INST: uvdot   za.s[w11, 2, vgx2], { z8.h, z9.h }, z1.h[1]
// CHECK-ENCODING: [0x32,0x65,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1516532 <unknown>

uvdot   za.s[w9, 7, vgx2], {z12.h, z13.h}, z11.h[2]  // 11000001-01011011-00101001-10110111
// CHECK-INST: uvdot   za.s[w9, 7, vgx2], { z12.h, z13.h }, z11.h[2]
// CHECK-ENCODING: [0xb7,0x29,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15b29b7 <unknown>

uvdot   za.s[w9, 7], {z12.h, z13.h}, z11.h[2]  // 11000001-01011011-00101001-10110111
// CHECK-INST: uvdot   za.s[w9, 7, vgx2], { z12.h, z13.h }, z11.h[2]
// CHECK-ENCODING: [0xb7,0x29,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15b29b7 <unknown>


uvdot   za.s[w8, 0, vgx4], {z0.b - z3.b}, z0.b[0]  // 11000001-01010000-10000000-00110000
// CHECK-INST: uvdot   za.s[w8, 0, vgx4], { z0.b - z3.b }, z0.b[0]
// CHECK-ENCODING: [0x30,0x80,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1508030 <unknown>

uvdot   za.s[w8, 0], {z0.b - z3.b}, z0.b[0]  // 11000001-01010000-10000000-00110000
// CHECK-INST: uvdot   za.s[w8, 0, vgx4], { z0.b - z3.b }, z0.b[0]
// CHECK-ENCODING: [0x30,0x80,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1508030 <unknown>

uvdot   za.s[w10, 5, vgx4], {z8.b - z11.b}, z5.b[1]  // 11000001-01010101-11000101-00110101
// CHECK-INST: uvdot   za.s[w10, 5, vgx4], { z8.b - z11.b }, z5.b[1]
// CHECK-ENCODING: [0x35,0xc5,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c155c535 <unknown>

uvdot   za.s[w10, 5], {z8.b - z11.b}, z5.b[1]  // 11000001-01010101-11000101-00110101
// CHECK-INST: uvdot   za.s[w10, 5, vgx4], { z8.b - z11.b }, z5.b[1]
// CHECK-ENCODING: [0x35,0xc5,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c155c535 <unknown>

uvdot   za.s[w11, 7, vgx4], {z12.b - z15.b}, z8.b[3]  // 11000001-01011000-11101101-10110111
// CHECK-INST: uvdot   za.s[w11, 7, vgx4], { z12.b - z15.b }, z8.b[3]
// CHECK-ENCODING: [0xb7,0xed,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c158edb7 <unknown>

uvdot   za.s[w11, 7], {z12.b - z15.b}, z8.b[3]  // 11000001-01011000-11101101-10110111
// CHECK-INST: uvdot   za.s[w11, 7, vgx4], { z12.b - z15.b }, z8.b[3]
// CHECK-ENCODING: [0xb7,0xed,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c158edb7 <unknown>

uvdot   za.s[w11, 7, vgx4], {z28.b - z31.b}, z15.b[3]  // 11000001-01011111-11101111-10110111
// CHECK-INST: uvdot   za.s[w11, 7, vgx4], { z28.b - z31.b }, z15.b[3]
// CHECK-ENCODING: [0xb7,0xef,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15fefb7 <unknown>

uvdot   za.s[w11, 7], {z28.b - z31.b}, z15.b[3]  // 11000001-01011111-11101111-10110111
// CHECK-INST: uvdot   za.s[w11, 7, vgx4], { z28.b - z31.b }, z15.b[3]
// CHECK-ENCODING: [0xb7,0xef,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15fefb7 <unknown>

uvdot   za.s[w8, 5, vgx4], {z16.b - z19.b}, z0.b[3]  // 11000001-01010000-10001110-00110101
// CHECK-INST: uvdot   za.s[w8, 5, vgx4], { z16.b - z19.b }, z0.b[3]
// CHECK-ENCODING: [0x35,0x8e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1508e35 <unknown>

uvdot   za.s[w8, 5], {z16.b - z19.b}, z0.b[3]  // 11000001-01010000-10001110-00110101
// CHECK-INST: uvdot   za.s[w8, 5, vgx4], { z16.b - z19.b }, z0.b[3]
// CHECK-ENCODING: [0x35,0x8e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1508e35 <unknown>

uvdot   za.s[w8, 1, vgx4], {z0.b - z3.b}, z14.b[1]  // 11000001-01011110-10000100-00110001
// CHECK-INST: uvdot   za.s[w8, 1, vgx4], { z0.b - z3.b }, z14.b[1]
// CHECK-ENCODING: [0x31,0x84,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e8431 <unknown>

uvdot   za.s[w8, 1], {z0.b - z3.b}, z14.b[1]  // 11000001-01011110-10000100-00110001
// CHECK-INST: uvdot   za.s[w8, 1, vgx4], { z0.b - z3.b }, z14.b[1]
// CHECK-ENCODING: [0x31,0x84,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e8431 <unknown>

uvdot   za.s[w10, 0, vgx4], {z16.b - z19.b}, z4.b[1]  // 11000001-01010100-11000110-00110000
// CHECK-INST: uvdot   za.s[w10, 0, vgx4], { z16.b - z19.b }, z4.b[1]
// CHECK-ENCODING: [0x30,0xc6,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c154c630 <unknown>

uvdot   za.s[w10, 0], {z16.b - z19.b}, z4.b[1]  // 11000001-01010100-11000110-00110000
// CHECK-INST: uvdot   za.s[w10, 0, vgx4], { z16.b - z19.b }, z4.b[1]
// CHECK-ENCODING: [0x30,0xc6,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c154c630 <unknown>

uvdot   za.s[w8, 0, vgx4], {z12.b - z15.b}, z2.b[2]  // 11000001-01010010-10001001-10110000
// CHECK-INST: uvdot   za.s[w8, 0, vgx4], { z12.b - z15.b }, z2.b[2]
// CHECK-ENCODING: [0xb0,0x89,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15289b0 <unknown>

uvdot   za.s[w8, 0], {z12.b - z15.b}, z2.b[2]  // 11000001-01010010-10001001-10110000
// CHECK-INST: uvdot   za.s[w8, 0, vgx4], { z12.b - z15.b }, z2.b[2]
// CHECK-ENCODING: [0xb0,0x89,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15289b0 <unknown>

uvdot   za.s[w10, 1, vgx4], {z0.b - z3.b}, z10.b[2]  // 11000001-01011010-11001000-00110001
// CHECK-INST: uvdot   za.s[w10, 1, vgx4], { z0.b - z3.b }, z10.b[2]
// CHECK-ENCODING: [0x31,0xc8,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15ac831 <unknown>

uvdot   za.s[w10, 1], {z0.b - z3.b}, z10.b[2]  // 11000001-01011010-11001000-00110001
// CHECK-INST: uvdot   za.s[w10, 1, vgx4], { z0.b - z3.b }, z10.b[2]
// CHECK-ENCODING: [0x31,0xc8,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15ac831 <unknown>

uvdot   za.s[w8, 5, vgx4], {z20.b - z23.b}, z14.b[2]  // 11000001-01011110-10001010-10110101
// CHECK-INST: uvdot   za.s[w8, 5, vgx4], { z20.b - z23.b }, z14.b[2]
// CHECK-ENCODING: [0xb5,0x8a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e8ab5 <unknown>

uvdot   za.s[w8, 5], {z20.b - z23.b}, z14.b[2]  // 11000001-01011110-10001010-10110101
// CHECK-INST: uvdot   za.s[w8, 5, vgx4], { z20.b - z23.b }, z14.b[2]
// CHECK-ENCODING: [0xb5,0x8a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e8ab5 <unknown>

uvdot   za.s[w11, 2, vgx4], {z8.b - z11.b}, z1.b[1]  // 11000001-01010001-11100101-00110010
// CHECK-INST: uvdot   za.s[w11, 2, vgx4], { z8.b - z11.b }, z1.b[1]
// CHECK-ENCODING: [0x32,0xe5,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c151e532 <unknown>

uvdot   za.s[w11, 2], {z8.b - z11.b}, z1.b[1]  // 11000001-01010001-11100101-00110010
// CHECK-INST: uvdot   za.s[w11, 2, vgx4], { z8.b - z11.b }, z1.b[1]
// CHECK-ENCODING: [0x32,0xe5,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c151e532 <unknown>

uvdot   za.s[w9, 7, vgx4], {z12.b - z15.b}, z11.b[2]  // 11000001-01011011-10101001-10110111
// CHECK-INST: uvdot   za.s[w9, 7, vgx4], { z12.b - z15.b }, z11.b[2]
// CHECK-ENCODING: [0xb7,0xa9,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15ba9b7 <unknown>

uvdot   za.s[w9, 7], {z12.b - z15.b}, z11.b[2]  // 11000001-01011011-10101001-10110111
// CHECK-INST: uvdot   za.s[w9, 7, vgx4], { z12.b - z15.b }, z11.b[2]
// CHECK-ENCODING: [0xb7,0xa9,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15ba9b7 <unknown>


uvdot   za.d[w8, 0, vgx4], {z0.h - z3.h}, z0.h[0]  // 11000001-11010000-10001000-00011000
// CHECK-INST: uvdot   za.d[w8, 0, vgx4], { z0.h - z3.h }, z0.h[0]
// CHECK-ENCODING: [0x18,0x88,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d08818 <unknown>

uvdot   za.d[w8, 0], {z0.h - z3.h}, z0.h[0]  // 11000001-11010000-10001000-00011000
// CHECK-INST: uvdot   za.d[w8, 0, vgx4], { z0.h - z3.h }, z0.h[0]
// CHECK-ENCODING: [0x18,0x88,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d08818 <unknown>

uvdot   za.d[w10, 5, vgx4], {z8.h - z11.h}, z5.h[1]  // 11000001-11010101-11001101-00011101
// CHECK-INST: uvdot   za.d[w10, 5, vgx4], { z8.h - z11.h }, z5.h[1]
// CHECK-ENCODING: [0x1d,0xcd,0xd5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d5cd1d <unknown>

uvdot   za.d[w10, 5], {z8.h - z11.h}, z5.h[1]  // 11000001-11010101-11001101-00011101
// CHECK-INST: uvdot   za.d[w10, 5, vgx4], { z8.h - z11.h }, z5.h[1]
// CHECK-ENCODING: [0x1d,0xcd,0xd5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d5cd1d <unknown>

uvdot   za.d[w11, 7, vgx4], {z12.h - z15.h}, z8.h[1]  // 11000001-11011000-11101101-10011111
// CHECK-INST: uvdot   za.d[w11, 7, vgx4], { z12.h - z15.h }, z8.h[1]
// CHECK-ENCODING: [0x9f,0xed,0xd8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d8ed9f <unknown>

uvdot   za.d[w11, 7], {z12.h - z15.h}, z8.h[1]  // 11000001-11011000-11101101-10011111
// CHECK-INST: uvdot   za.d[w11, 7, vgx4], { z12.h - z15.h }, z8.h[1]
// CHECK-ENCODING: [0x9f,0xed,0xd8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d8ed9f <unknown>

uvdot   za.d[w11, 7, vgx4], {z28.h - z31.h}, z15.h[1]  // 11000001-11011111-11101111-10011111
// CHECK-INST: uvdot   za.d[w11, 7, vgx4], { z28.h - z31.h }, z15.h[1]
// CHECK-ENCODING: [0x9f,0xef,0xdf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dfef9f <unknown>

uvdot   za.d[w11, 7], {z28.h - z31.h}, z15.h[1]  // 11000001-11011111-11101111-10011111
// CHECK-INST: uvdot   za.d[w11, 7, vgx4], { z28.h - z31.h }, z15.h[1]
// CHECK-ENCODING: [0x9f,0xef,0xdf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dfef9f <unknown>

uvdot   za.d[w8, 5, vgx4], {z16.h - z19.h}, z0.h[1]  // 11000001-11010000-10001110-00011101
// CHECK-INST: uvdot   za.d[w8, 5, vgx4], { z16.h - z19.h }, z0.h[1]
// CHECK-ENCODING: [0x1d,0x8e,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d08e1d <unknown>

uvdot   za.d[w8, 5], {z16.h - z19.h}, z0.h[1]  // 11000001-11010000-10001110-00011101
// CHECK-INST: uvdot   za.d[w8, 5, vgx4], { z16.h - z19.h }, z0.h[1]
// CHECK-ENCODING: [0x1d,0x8e,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d08e1d <unknown>

uvdot   za.d[w8, 1, vgx4], {z0.h - z3.h}, z14.h[1]  // 11000001-11011110-10001100-00011001
// CHECK-INST: uvdot   za.d[w8, 1, vgx4], { z0.h - z3.h }, z14.h[1]
// CHECK-ENCODING: [0x19,0x8c,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de8c19 <unknown>

uvdot   za.d[w8, 1], {z0.h - z3.h}, z14.h[1]  // 11000001-11011110-10001100-00011001
// CHECK-INST: uvdot   za.d[w8, 1, vgx4], { z0.h - z3.h }, z14.h[1]
// CHECK-ENCODING: [0x19,0x8c,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de8c19 <unknown>

uvdot   za.d[w10, 0, vgx4], {z16.h - z19.h}, z4.h[1]  // 11000001-11010100-11001110-00011000
// CHECK-INST: uvdot   za.d[w10, 0, vgx4], { z16.h - z19.h }, z4.h[1]
// CHECK-ENCODING: [0x18,0xce,0xd4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d4ce18 <unknown>

uvdot   za.d[w10, 0], {z16.h - z19.h}, z4.h[1]  // 11000001-11010100-11001110-00011000
// CHECK-INST: uvdot   za.d[w10, 0, vgx4], { z16.h - z19.h }, z4.h[1]
// CHECK-ENCODING: [0x18,0xce,0xd4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d4ce18 <unknown>

uvdot   za.d[w8, 0, vgx4], {z12.h - z15.h}, z2.h[0]  // 11000001-11010010-10001001-10011000
// CHECK-INST: uvdot   za.d[w8, 0, vgx4], { z12.h - z15.h }, z2.h[0]
// CHECK-ENCODING: [0x98,0x89,0xd2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d28998 <unknown>

uvdot   za.d[w8, 0], {z12.h - z15.h}, z2.h[0]  // 11000001-11010010-10001001-10011000
// CHECK-INST: uvdot   za.d[w8, 0, vgx4], { z12.h - z15.h }, z2.h[0]
// CHECK-ENCODING: [0x98,0x89,0xd2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d28998 <unknown>

uvdot   za.d[w10, 1, vgx4], {z0.h - z3.h}, z10.h[0]  // 11000001-11011010-11001000-00011001
// CHECK-INST: uvdot   za.d[w10, 1, vgx4], { z0.h - z3.h }, z10.h[0]
// CHECK-ENCODING: [0x19,0xc8,0xda,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dac819 <unknown>

uvdot   za.d[w10, 1], {z0.h - z3.h}, z10.h[0]  // 11000001-11011010-11001000-00011001
// CHECK-INST: uvdot   za.d[w10, 1, vgx4], { z0.h - z3.h }, z10.h[0]
// CHECK-ENCODING: [0x19,0xc8,0xda,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dac819 <unknown>

uvdot   za.d[w8, 5, vgx4], {z20.h - z23.h}, z14.h[0]  // 11000001-11011110-10001010-10011101
// CHECK-INST: uvdot   za.d[w8, 5, vgx4], { z20.h - z23.h }, z14.h[0]
// CHECK-ENCODING: [0x9d,0x8a,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de8a9d <unknown>

uvdot   za.d[w8, 5], {z20.h - z23.h}, z14.h[0]  // 11000001-11011110-10001010-10011101
// CHECK-INST: uvdot   za.d[w8, 5, vgx4], { z20.h - z23.h }, z14.h[0]
// CHECK-ENCODING: [0x9d,0x8a,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de8a9d <unknown>

uvdot   za.d[w11, 2, vgx4], {z8.h - z11.h}, z1.h[1]  // 11000001-11010001-11101101-00011010
// CHECK-INST: uvdot   za.d[w11, 2, vgx4], { z8.h - z11.h }, z1.h[1]
// CHECK-ENCODING: [0x1a,0xed,0xd1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d1ed1a <unknown>

uvdot   za.d[w11, 2], {z8.h - z11.h}, z1.h[1]  // 11000001-11010001-11101101-00011010
// CHECK-INST: uvdot   za.d[w11, 2, vgx4], { z8.h - z11.h }, z1.h[1]
// CHECK-ENCODING: [0x1a,0xed,0xd1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d1ed1a <unknown>

uvdot   za.d[w9, 7, vgx4], {z12.h - z15.h}, z11.h[0]  // 11000001-11011011-10101001-10011111
// CHECK-INST: uvdot   za.d[w9, 7, vgx4], { z12.h - z15.h }, z11.h[0]
// CHECK-ENCODING: [0x9f,0xa9,0xdb,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dba99f <unknown>

uvdot   za.d[w9, 7], {z12.h - z15.h}, z11.h[0]  // 11000001-11011011-10101001-10011111
// CHECK-INST: uvdot   za.d[w9, 7, vgx4], { z12.h - z15.h }, z11.h[0]
// CHECK-ENCODING: [0x9f,0xa9,0xdb,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dba99f <unknown>

