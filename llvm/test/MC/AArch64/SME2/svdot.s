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


svdot   za.s[w8, 0, vgx2], {z0.h, z1.h}, z0.h[0]  // 11000001-01010000-00000000-00100000
// CHECK-INST: svdot   za.s[w8, 0, vgx2], { z0.h, z1.h }, z0.h[0]
// CHECK-ENCODING: [0x20,0x00,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1500020 <unknown>

svdot   za.s[w8, 0], {z0.h, z1.h}, z0.h[0]  // 11000001-01010000-00000000-00100000
// CHECK-INST: svdot   za.s[w8, 0, vgx2], { z0.h, z1.h }, z0.h[0]
// CHECK-ENCODING: [0x20,0x00,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1500020 <unknown>

svdot   za.s[w10, 5, vgx2], {z10.h, z11.h}, z5.h[1]  // 11000001-01010101-01000101-01100101
// CHECK-INST: svdot   za.s[w10, 5, vgx2], { z10.h, z11.h }, z5.h[1]
// CHECK-ENCODING: [0x65,0x45,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1554565 <unknown>

svdot   za.s[w10, 5], {z10.h, z11.h}, z5.h[1]  // 11000001-01010101-01000101-01100101
// CHECK-INST: svdot   za.s[w10, 5, vgx2], { z10.h, z11.h }, z5.h[1]
// CHECK-ENCODING: [0x65,0x45,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1554565 <unknown>

svdot   za.s[w11, 7, vgx2], {z12.h, z13.h}, z8.h[3]  // 11000001-01011000-01101101-10100111
// CHECK-INST: svdot   za.s[w11, 7, vgx2], { z12.h, z13.h }, z8.h[3]
// CHECK-ENCODING: [0xa7,0x6d,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1586da7 <unknown>

svdot   za.s[w11, 7], {z12.h, z13.h}, z8.h[3]  // 11000001-01011000-01101101-10100111
// CHECK-INST: svdot   za.s[w11, 7, vgx2], { z12.h, z13.h }, z8.h[3]
// CHECK-ENCODING: [0xa7,0x6d,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1586da7 <unknown>

svdot   za.s[w11, 7, vgx2], {z30.h, z31.h}, z15.h[3]  // 11000001-01011111-01101111-11100111
// CHECK-INST: svdot   za.s[w11, 7, vgx2], { z30.h, z31.h }, z15.h[3]
// CHECK-ENCODING: [0xe7,0x6f,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15f6fe7 <unknown>

svdot   za.s[w11, 7], {z30.h, z31.h}, z15.h[3]  // 11000001-01011111-01101111-11100111
// CHECK-INST: svdot   za.s[w11, 7, vgx2], { z30.h, z31.h }, z15.h[3]
// CHECK-ENCODING: [0xe7,0x6f,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15f6fe7 <unknown>

svdot   za.s[w8, 5, vgx2], {z16.h, z17.h}, z0.h[3]  // 11000001-01010000-00001110-00100101
// CHECK-INST: svdot   za.s[w8, 5, vgx2], { z16.h, z17.h }, z0.h[3]
// CHECK-ENCODING: [0x25,0x0e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1500e25 <unknown>

svdot   za.s[w8, 5], {z16.h, z17.h}, z0.h[3]  // 11000001-01010000-00001110-00100101
// CHECK-INST: svdot   za.s[w8, 5, vgx2], { z16.h, z17.h }, z0.h[3]
// CHECK-ENCODING: [0x25,0x0e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1500e25 <unknown>

svdot   za.s[w8, 1, vgx2], {z0.h, z1.h}, z14.h[1]  // 11000001-01011110-00000100-00100001
// CHECK-INST: svdot   za.s[w8, 1, vgx2], { z0.h, z1.h }, z14.h[1]
// CHECK-ENCODING: [0x21,0x04,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e0421 <unknown>

svdot   za.s[w8, 1], {z0.h, z1.h}, z14.h[1]  // 11000001-01011110-00000100-00100001
// CHECK-INST: svdot   za.s[w8, 1, vgx2], { z0.h, z1.h }, z14.h[1]
// CHECK-ENCODING: [0x21,0x04,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e0421 <unknown>

svdot   za.s[w10, 0, vgx2], {z18.h, z19.h}, z4.h[1]  // 11000001-01010100-01000110-01100000
// CHECK-INST: svdot   za.s[w10, 0, vgx2], { z18.h, z19.h }, z4.h[1]
// CHECK-ENCODING: [0x60,0x46,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1544660 <unknown>

svdot   za.s[w10, 0], {z18.h, z19.h}, z4.h[1]  // 11000001-01010100-01000110-01100000
// CHECK-INST: svdot   za.s[w10, 0, vgx2], { z18.h, z19.h }, z4.h[1]
// CHECK-ENCODING: [0x60,0x46,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1544660 <unknown>

svdot   za.s[w8, 0, vgx2], {z12.h, z13.h}, z2.h[2]  // 11000001-01010010-00001001-10100000
// CHECK-INST: svdot   za.s[w8, 0, vgx2], { z12.h, z13.h }, z2.h[2]
// CHECK-ENCODING: [0xa0,0x09,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15209a0 <unknown>

svdot   za.s[w8, 0], {z12.h, z13.h}, z2.h[2]  // 11000001-01010010-00001001-10100000
// CHECK-INST: svdot   za.s[w8, 0, vgx2], { z12.h, z13.h }, z2.h[2]
// CHECK-ENCODING: [0xa0,0x09,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15209a0 <unknown>

svdot   za.s[w10, 1, vgx2], {z0.h, z1.h}, z10.h[2]  // 11000001-01011010-01001000-00100001
// CHECK-INST: svdot   za.s[w10, 1, vgx2], { z0.h, z1.h }, z10.h[2]
// CHECK-ENCODING: [0x21,0x48,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15a4821 <unknown>

svdot   za.s[w10, 1], {z0.h, z1.h}, z10.h[2]  // 11000001-01011010-01001000-00100001
// CHECK-INST: svdot   za.s[w10, 1, vgx2], { z0.h, z1.h }, z10.h[2]
// CHECK-ENCODING: [0x21,0x48,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15a4821 <unknown>

svdot   za.s[w8, 5, vgx2], {z22.h, z23.h}, z14.h[2]  // 11000001-01011110-00001010-11100101
// CHECK-INST: svdot   za.s[w8, 5, vgx2], { z22.h, z23.h }, z14.h[2]
// CHECK-ENCODING: [0xe5,0x0a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e0ae5 <unknown>

svdot   za.s[w8, 5], {z22.h, z23.h}, z14.h[2]  // 11000001-01011110-00001010-11100101
// CHECK-INST: svdot   za.s[w8, 5, vgx2], { z22.h, z23.h }, z14.h[2]
// CHECK-ENCODING: [0xe5,0x0a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e0ae5 <unknown>

svdot   za.s[w11, 2, vgx2], {z8.h, z9.h}, z1.h[1]  // 11000001-01010001-01100101-00100010
// CHECK-INST: svdot   za.s[w11, 2, vgx2], { z8.h, z9.h }, z1.h[1]
// CHECK-ENCODING: [0x22,0x65,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1516522 <unknown>

svdot   za.s[w11, 2], {z8.h, z9.h}, z1.h[1]  // 11000001-01010001-01100101-00100010
// CHECK-INST: svdot   za.s[w11, 2, vgx2], { z8.h, z9.h }, z1.h[1]
// CHECK-ENCODING: [0x22,0x65,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1516522 <unknown>

svdot   za.s[w9, 7, vgx2], {z12.h, z13.h}, z11.h[2]  // 11000001-01011011-00101001-10100111
// CHECK-INST: svdot   za.s[w9, 7, vgx2], { z12.h, z13.h }, z11.h[2]
// CHECK-ENCODING: [0xa7,0x29,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15b29a7 <unknown>

svdot   za.s[w9, 7], {z12.h, z13.h}, z11.h[2]  // 11000001-01011011-00101001-10100111
// CHECK-INST: svdot   za.s[w9, 7, vgx2], { z12.h, z13.h }, z11.h[2]
// CHECK-ENCODING: [0xa7,0x29,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15b29a7 <unknown>


svdot   za.s[w8, 0, vgx4], {z0.b - z3.b}, z0.b[0]  // 11000001-01010000-10000000-00100000
// CHECK-INST: svdot   za.s[w8, 0, vgx4], { z0.b - z3.b }, z0.b[0]
// CHECK-ENCODING: [0x20,0x80,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1508020 <unknown>

svdot   za.s[w8, 0], {z0.b - z3.b}, z0.b[0]  // 11000001-01010000-10000000-00100000
// CHECK-INST: svdot   za.s[w8, 0, vgx4], { z0.b - z3.b }, z0.b[0]
// CHECK-ENCODING: [0x20,0x80,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1508020 <unknown>

svdot   za.s[w10, 5, vgx4], {z8.b - z11.b}, z5.b[1]  // 11000001-01010101-11000101-00100101
// CHECK-INST: svdot   za.s[w10, 5, vgx4], { z8.b - z11.b }, z5.b[1]
// CHECK-ENCODING: [0x25,0xc5,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c155c525 <unknown>

svdot   za.s[w10, 5], {z8.b - z11.b}, z5.b[1]  // 11000001-01010101-11000101-00100101
// CHECK-INST: svdot   za.s[w10, 5, vgx4], { z8.b - z11.b }, z5.b[1]
// CHECK-ENCODING: [0x25,0xc5,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c155c525 <unknown>

svdot   za.s[w11, 7, vgx4], {z12.b - z15.b}, z8.b[3]  // 11000001-01011000-11101101-10100111
// CHECK-INST: svdot   za.s[w11, 7, vgx4], { z12.b - z15.b }, z8.b[3]
// CHECK-ENCODING: [0xa7,0xed,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c158eda7 <unknown>

svdot   za.s[w11, 7], {z12.b - z15.b}, z8.b[3]  // 11000001-01011000-11101101-10100111
// CHECK-INST: svdot   za.s[w11, 7, vgx4], { z12.b - z15.b }, z8.b[3]
// CHECK-ENCODING: [0xa7,0xed,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c158eda7 <unknown>

svdot   za.s[w11, 7, vgx4], {z28.b - z31.b}, z15.b[3]  // 11000001-01011111-11101111-10100111
// CHECK-INST: svdot   za.s[w11, 7, vgx4], { z28.b - z31.b }, z15.b[3]
// CHECK-ENCODING: [0xa7,0xef,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15fefa7 <unknown>

svdot   za.s[w11, 7], {z28.b - z31.b}, z15.b[3]  // 11000001-01011111-11101111-10100111
// CHECK-INST: svdot   za.s[w11, 7, vgx4], { z28.b - z31.b }, z15.b[3]
// CHECK-ENCODING: [0xa7,0xef,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15fefa7 <unknown>

svdot   za.s[w8, 5, vgx4], {z16.b - z19.b}, z0.b[3]  // 11000001-01010000-10001110-00100101
// CHECK-INST: svdot   za.s[w8, 5, vgx4], { z16.b - z19.b }, z0.b[3]
// CHECK-ENCODING: [0x25,0x8e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1508e25 <unknown>

svdot   za.s[w8, 5], {z16.b - z19.b}, z0.b[3]  // 11000001-01010000-10001110-00100101
// CHECK-INST: svdot   za.s[w8, 5, vgx4], { z16.b - z19.b }, z0.b[3]
// CHECK-ENCODING: [0x25,0x8e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1508e25 <unknown>

svdot   za.s[w8, 1, vgx4], {z0.b - z3.b}, z14.b[1]  // 11000001-01011110-10000100-00100001
// CHECK-INST: svdot   za.s[w8, 1, vgx4], { z0.b - z3.b }, z14.b[1]
// CHECK-ENCODING: [0x21,0x84,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e8421 <unknown>

svdot   za.s[w8, 1], {z0.b - z3.b}, z14.b[1]  // 11000001-01011110-10000100-00100001
// CHECK-INST: svdot   za.s[w8, 1, vgx4], { z0.b - z3.b }, z14.b[1]
// CHECK-ENCODING: [0x21,0x84,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e8421 <unknown>

svdot   za.s[w10, 0, vgx4], {z16.b - z19.b}, z4.b[1]  // 11000001-01010100-11000110-00100000
// CHECK-INST: svdot   za.s[w10, 0, vgx4], { z16.b - z19.b }, z4.b[1]
// CHECK-ENCODING: [0x20,0xc6,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c154c620 <unknown>

svdot   za.s[w10, 0], {z16.b - z19.b}, z4.b[1]  // 11000001-01010100-11000110-00100000
// CHECK-INST: svdot   za.s[w10, 0, vgx4], { z16.b - z19.b }, z4.b[1]
// CHECK-ENCODING: [0x20,0xc6,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c154c620 <unknown>

svdot   za.s[w8, 0, vgx4], {z12.b - z15.b}, z2.b[2]  // 11000001-01010010-10001001-10100000
// CHECK-INST: svdot   za.s[w8, 0, vgx4], { z12.b - z15.b }, z2.b[2]
// CHECK-ENCODING: [0xa0,0x89,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15289a0 <unknown>

svdot   za.s[w8, 0], {z12.b - z15.b}, z2.b[2]  // 11000001-01010010-10001001-10100000
// CHECK-INST: svdot   za.s[w8, 0, vgx4], { z12.b - z15.b }, z2.b[2]
// CHECK-ENCODING: [0xa0,0x89,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15289a0 <unknown>

svdot   za.s[w10, 1, vgx4], {z0.b - z3.b}, z10.b[2]  // 11000001-01011010-11001000-00100001
// CHECK-INST: svdot   za.s[w10, 1, vgx4], { z0.b - z3.b }, z10.b[2]
// CHECK-ENCODING: [0x21,0xc8,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15ac821 <unknown>

svdot   za.s[w10, 1], {z0.b - z3.b}, z10.b[2]  // 11000001-01011010-11001000-00100001
// CHECK-INST: svdot   za.s[w10, 1, vgx4], { z0.b - z3.b }, z10.b[2]
// CHECK-ENCODING: [0x21,0xc8,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15ac821 <unknown>

svdot   za.s[w8, 5, vgx4], {z20.b - z23.b}, z14.b[2]  // 11000001-01011110-10001010-10100101
// CHECK-INST: svdot   za.s[w8, 5, vgx4], { z20.b - z23.b }, z14.b[2]
// CHECK-ENCODING: [0xa5,0x8a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e8aa5 <unknown>

svdot   za.s[w8, 5], {z20.b - z23.b}, z14.b[2]  // 11000001-01011110-10001010-10100101
// CHECK-INST: svdot   za.s[w8, 5, vgx4], { z20.b - z23.b }, z14.b[2]
// CHECK-ENCODING: [0xa5,0x8a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e8aa5 <unknown>

svdot   za.s[w11, 2, vgx4], {z8.b - z11.b}, z1.b[1]  // 11000001-01010001-11100101-00100010
// CHECK-INST: svdot   za.s[w11, 2, vgx4], { z8.b - z11.b }, z1.b[1]
// CHECK-ENCODING: [0x22,0xe5,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c151e522 <unknown>

svdot   za.s[w11, 2], {z8.b - z11.b}, z1.b[1]  // 11000001-01010001-11100101-00100010
// CHECK-INST: svdot   za.s[w11, 2, vgx4], { z8.b - z11.b }, z1.b[1]
// CHECK-ENCODING: [0x22,0xe5,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c151e522 <unknown>

svdot   za.s[w9, 7, vgx4], {z12.b - z15.b}, z11.b[2]  // 11000001-01011011-10101001-10100111
// CHECK-INST: svdot   za.s[w9, 7, vgx4], { z12.b - z15.b }, z11.b[2]
// CHECK-ENCODING: [0xa7,0xa9,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15ba9a7 <unknown>

svdot   za.s[w9, 7], {z12.b - z15.b}, z11.b[2]  // 11000001-01011011-10101001-10100111
// CHECK-INST: svdot   za.s[w9, 7, vgx4], { z12.b - z15.b }, z11.b[2]
// CHECK-ENCODING: [0xa7,0xa9,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15ba9a7 <unknown>


svdot   za.d[w8, 0, vgx4], {z0.h - z3.h}, z0.h[0]  // 11000001-11010000-10001000-00001000
// CHECK-INST: svdot   za.d[w8, 0, vgx4], { z0.h - z3.h }, z0.h[0]
// CHECK-ENCODING: [0x08,0x88,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d08808 <unknown>

svdot   za.d[w8, 0], {z0.h - z3.h}, z0.h[0]  // 11000001-11010000-10001000-00001000
// CHECK-INST: svdot   za.d[w8, 0, vgx4], { z0.h - z3.h }, z0.h[0]
// CHECK-ENCODING: [0x08,0x88,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d08808 <unknown>

svdot   za.d[w10, 5, vgx4], {z8.h - z11.h}, z5.h[1]  // 11000001-11010101-11001101-00001101
// CHECK-INST: svdot   za.d[w10, 5, vgx4], { z8.h - z11.h }, z5.h[1]
// CHECK-ENCODING: [0x0d,0xcd,0xd5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d5cd0d <unknown>

svdot   za.d[w10, 5], {z8.h - z11.h}, z5.h[1]  // 11000001-11010101-11001101-00001101
// CHECK-INST: svdot   za.d[w10, 5, vgx4], { z8.h - z11.h }, z5.h[1]
// CHECK-ENCODING: [0x0d,0xcd,0xd5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d5cd0d <unknown>

svdot   za.d[w11, 7, vgx4], {z12.h - z15.h}, z8.h[1]  // 11000001-11011000-11101101-10001111
// CHECK-INST: svdot   za.d[w11, 7, vgx4], { z12.h - z15.h }, z8.h[1]
// CHECK-ENCODING: [0x8f,0xed,0xd8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d8ed8f <unknown>

svdot   za.d[w11, 7], {z12.h - z15.h}, z8.h[1]  // 11000001-11011000-11101101-10001111
// CHECK-INST: svdot   za.d[w11, 7, vgx4], { z12.h - z15.h }, z8.h[1]
// CHECK-ENCODING: [0x8f,0xed,0xd8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d8ed8f <unknown>

svdot   za.d[w11, 7, vgx4], {z28.h - z31.h}, z15.h[1]  // 11000001-11011111-11101111-10001111
// CHECK-INST: svdot   za.d[w11, 7, vgx4], { z28.h - z31.h }, z15.h[1]
// CHECK-ENCODING: [0x8f,0xef,0xdf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dfef8f <unknown>

svdot   za.d[w11, 7], {z28.h - z31.h}, z15.h[1]  // 11000001-11011111-11101111-10001111
// CHECK-INST: svdot   za.d[w11, 7, vgx4], { z28.h - z31.h }, z15.h[1]
// CHECK-ENCODING: [0x8f,0xef,0xdf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dfef8f <unknown>

svdot   za.d[w8, 5, vgx4], {z16.h - z19.h}, z0.h[1]  // 11000001-11010000-10001110-00001101
// CHECK-INST: svdot   za.d[w8, 5, vgx4], { z16.h - z19.h }, z0.h[1]
// CHECK-ENCODING: [0x0d,0x8e,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d08e0d <unknown>

svdot   za.d[w8, 5], {z16.h - z19.h}, z0.h[1]  // 11000001-11010000-10001110-00001101
// CHECK-INST: svdot   za.d[w8, 5, vgx4], { z16.h - z19.h }, z0.h[1]
// CHECK-ENCODING: [0x0d,0x8e,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d08e0d <unknown>

svdot   za.d[w8, 1, vgx4], {z0.h - z3.h}, z14.h[1]  // 11000001-11011110-10001100-00001001
// CHECK-INST: svdot   za.d[w8, 1, vgx4], { z0.h - z3.h }, z14.h[1]
// CHECK-ENCODING: [0x09,0x8c,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de8c09 <unknown>

svdot   za.d[w8, 1], {z0.h - z3.h}, z14.h[1]  // 11000001-11011110-10001100-00001001
// CHECK-INST: svdot   za.d[w8, 1, vgx4], { z0.h - z3.h }, z14.h[1]
// CHECK-ENCODING: [0x09,0x8c,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de8c09 <unknown>

svdot   za.d[w10, 0, vgx4], {z16.h - z19.h}, z4.h[1]  // 11000001-11010100-11001110-00001000
// CHECK-INST: svdot   za.d[w10, 0, vgx4], { z16.h - z19.h }, z4.h[1]
// CHECK-ENCODING: [0x08,0xce,0xd4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d4ce08 <unknown>

svdot   za.d[w10, 0], {z16.h - z19.h}, z4.h[1]  // 11000001-11010100-11001110-00001000
// CHECK-INST: svdot   za.d[w10, 0, vgx4], { z16.h - z19.h }, z4.h[1]
// CHECK-ENCODING: [0x08,0xce,0xd4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d4ce08 <unknown>

svdot   za.d[w8, 0, vgx4], {z12.h - z15.h}, z2.h[0]  // 11000001-11010010-10001001-10001000
// CHECK-INST: svdot   za.d[w8, 0, vgx4], { z12.h - z15.h }, z2.h[0]
// CHECK-ENCODING: [0x88,0x89,0xd2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d28988 <unknown>

svdot   za.d[w8, 0], {z12.h - z15.h}, z2.h[0]  // 11000001-11010010-10001001-10001000
// CHECK-INST: svdot   za.d[w8, 0, vgx4], { z12.h - z15.h }, z2.h[0]
// CHECK-ENCODING: [0x88,0x89,0xd2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d28988 <unknown>

svdot   za.d[w10, 1, vgx4], {z0.h - z3.h}, z10.h[0]  // 11000001-11011010-11001000-00001001
// CHECK-INST: svdot   za.d[w10, 1, vgx4], { z0.h - z3.h }, z10.h[0]
// CHECK-ENCODING: [0x09,0xc8,0xda,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dac809 <unknown>

svdot   za.d[w10, 1], {z0.h - z3.h}, z10.h[0]  // 11000001-11011010-11001000-00001001
// CHECK-INST: svdot   za.d[w10, 1, vgx4], { z0.h - z3.h }, z10.h[0]
// CHECK-ENCODING: [0x09,0xc8,0xda,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dac809 <unknown>

svdot   za.d[w8, 5, vgx4], {z20.h - z23.h}, z14.h[0]  // 11000001-11011110-10001010-10001101
// CHECK-INST: svdot   za.d[w8, 5, vgx4], { z20.h - z23.h }, z14.h[0]
// CHECK-ENCODING: [0x8d,0x8a,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de8a8d <unknown>

svdot   za.d[w8, 5], {z20.h - z23.h}, z14.h[0]  // 11000001-11011110-10001010-10001101
// CHECK-INST: svdot   za.d[w8, 5, vgx4], { z20.h - z23.h }, z14.h[0]
// CHECK-ENCODING: [0x8d,0x8a,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de8a8d <unknown>

svdot   za.d[w11, 2, vgx4], {z8.h - z11.h}, z1.h[1]  // 11000001-11010001-11101101-00001010
// CHECK-INST: svdot   za.d[w11, 2, vgx4], { z8.h - z11.h }, z1.h[1]
// CHECK-ENCODING: [0x0a,0xed,0xd1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d1ed0a <unknown>

svdot   za.d[w11, 2], {z8.h - z11.h}, z1.h[1]  // 11000001-11010001-11101101-00001010
// CHECK-INST: svdot   za.d[w11, 2, vgx4], { z8.h - z11.h }, z1.h[1]
// CHECK-ENCODING: [0x0a,0xed,0xd1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d1ed0a <unknown>

svdot   za.d[w9, 7, vgx4], {z12.h - z15.h}, z11.h[0]  // 11000001-11011011-10101001-10001111
// CHECK-INST: svdot   za.d[w9, 7, vgx4], { z12.h - z15.h }, z11.h[0]
// CHECK-ENCODING: [0x8f,0xa9,0xdb,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dba98f <unknown>

svdot   za.d[w9, 7], {z12.h - z15.h}, z11.h[0]  // 11000001-11011011-10101001-10001111
// CHECK-INST: svdot   za.d[w9, 7, vgx4], { z12.h - z15.h }, z11.h[0]
// CHECK-ENCODING: [0x8f,0xa9,0xdb,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dba98f <unknown>

