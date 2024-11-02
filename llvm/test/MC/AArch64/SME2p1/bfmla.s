// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p1,+b16b16 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p1,+b16b16 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2p1,+b16b16 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p1,+b16b16 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2p1 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p1,+b16b16 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2p1,+b16b16 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

bfmla   za.h[w8, 0, vgx2], {z0.h, z1.h}, z0.h  // 11000001-01100000-00011100-00000000
// CHECK-INST: bfmla   za.h[w8, 0, vgx2], { z0.h, z1.h }, z0.h
// CHECK-ENCODING: [0x00,0x1c,0x60,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1601c00 <unknown>

bfmla   za.h[w8, 0], {z0.h - z1.h}, z0.h  // 11000001-01100000-00011100-00000000
// CHECK-INST: bfmla   za.h[w8, 0, vgx2], { z0.h, z1.h }, z0.h
// CHECK-ENCODING: [0x00,0x1c,0x60,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1601c00 <unknown>

bfmla   za.h[w10, 5, vgx2], {z10.h, z11.h}, z5.h  // 11000001-01100101-01011101-01000101
// CHECK-INST: bfmla   za.h[w10, 5, vgx2], { z10.h, z11.h }, z5.h
// CHECK-ENCODING: [0x45,0x5d,0x65,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1655d45 <unknown>

bfmla   za.h[w10, 5], {z10.h - z11.h}, z5.h  // 11000001-01100101-01011101-01000101
// CHECK-INST: bfmla   za.h[w10, 5, vgx2], { z10.h, z11.h }, z5.h
// CHECK-ENCODING: [0x45,0x5d,0x65,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1655d45 <unknown>

bfmla   za.h[w11, 7, vgx2], {z13.h, z14.h}, z8.h  // 11000001-01101000-01111101-10100111
// CHECK-INST: bfmla   za.h[w11, 7, vgx2], { z13.h, z14.h }, z8.h
// CHECK-ENCODING: [0xa7,0x7d,0x68,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1687da7 <unknown>

bfmla   za.h[w11, 7], {z13.h - z14.h}, z8.h  // 11000001-01101000-01111101-10100111
// CHECK-INST: bfmla   za.h[w11, 7, vgx2], { z13.h, z14.h }, z8.h
// CHECK-ENCODING: [0xa7,0x7d,0x68,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1687da7 <unknown>

bfmla   za.h[w11, 7, vgx2], {z31.h, z0.h}, z15.h  // 11000001-01101111-01111111-11100111
// CHECK-INST: bfmla   za.h[w11, 7, vgx2], { z31.h, z0.h }, z15.h
// CHECK-ENCODING: [0xe7,0x7f,0x6f,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c16f7fe7 <unknown>

bfmla   za.h[w11, 7], {z31.h - z0.h}, z15.h  // 11000001-01101111-01111111-11100111
// CHECK-INST: bfmla   za.h[w11, 7, vgx2], { z31.h, z0.h }, z15.h
// CHECK-ENCODING: [0xe7,0x7f,0x6f,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c16f7fe7 <unknown>

bfmla   za.h[w8, 5, vgx2], {z17.h, z18.h}, z0.h  // 11000001-01100000-00011110-00100101
// CHECK-INST: bfmla   za.h[w8, 5, vgx2], { z17.h, z18.h }, z0.h
// CHECK-ENCODING: [0x25,0x1e,0x60,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1601e25 <unknown>

bfmla   za.h[w8, 5], {z17.h - z18.h}, z0.h  // 11000001-01100000-00011110-00100101
// CHECK-INST: bfmla   za.h[w8, 5, vgx2], { z17.h, z18.h }, z0.h
// CHECK-ENCODING: [0x25,0x1e,0x60,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1601e25 <unknown>

bfmla   za.h[w8, 1, vgx2], {z1.h, z2.h}, z14.h  // 11000001-01101110-00011100-00100001
// CHECK-INST: bfmla   za.h[w8, 1, vgx2], { z1.h, z2.h }, z14.h
// CHECK-ENCODING: [0x21,0x1c,0x6e,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c16e1c21 <unknown>

bfmla   za.h[w8, 1], {z1.h - z2.h}, z14.h  // 11000001-01101110-00011100-00100001
// CHECK-INST: bfmla   za.h[w8, 1, vgx2], { z1.h, z2.h }, z14.h
// CHECK-ENCODING: [0x21,0x1c,0x6e,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c16e1c21 <unknown>

bfmla   za.h[w10, 0, vgx2], {z19.h, z20.h}, z4.h  // 11000001-01100100-01011110-01100000
// CHECK-INST: bfmla   za.h[w10, 0, vgx2], { z19.h, z20.h }, z4.h
// CHECK-ENCODING: [0x60,0x5e,0x64,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1645e60 <unknown>

bfmla   za.h[w10, 0], {z19.h - z20.h}, z4.h  // 11000001-01100100-01011110-01100000
// CHECK-INST: bfmla   za.h[w10, 0, vgx2], { z19.h, z20.h }, z4.h
// CHECK-ENCODING: [0x60,0x5e,0x64,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1645e60 <unknown>

bfmla   za.h[w8, 0, vgx2], {z12.h, z13.h}, z2.h  // 11000001-01100010-00011101-10000000
// CHECK-INST: bfmla   za.h[w8, 0, vgx2], { z12.h, z13.h }, z2.h
// CHECK-ENCODING: [0x80,0x1d,0x62,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1621d80 <unknown>

bfmla   za.h[w8, 0], {z12.h - z13.h}, z2.h  // 11000001-01100010-00011101-10000000
// CHECK-INST: bfmla   za.h[w8, 0, vgx2], { z12.h, z13.h }, z2.h
// CHECK-ENCODING: [0x80,0x1d,0x62,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1621d80 <unknown>

bfmla   za.h[w10, 1, vgx2], {z1.h, z2.h}, z10.h  // 11000001-01101010-01011100-00100001
// CHECK-INST: bfmla   za.h[w10, 1, vgx2], { z1.h, z2.h }, z10.h
// CHECK-ENCODING: [0x21,0x5c,0x6a,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c16a5c21 <unknown>

bfmla   za.h[w10, 1], {z1.h - z2.h}, z10.h  // 11000001-01101010-01011100-00100001
// CHECK-INST: bfmla   za.h[w10, 1, vgx2], { z1.h, z2.h }, z10.h
// CHECK-ENCODING: [0x21,0x5c,0x6a,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c16a5c21 <unknown>

bfmla   za.h[w8, 5, vgx2], {z22.h, z23.h}, z14.h  // 11000001-01101110-00011110-11000101
// CHECK-INST: bfmla   za.h[w8, 5, vgx2], { z22.h, z23.h }, z14.h
// CHECK-ENCODING: [0xc5,0x1e,0x6e,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c16e1ec5 <unknown>

bfmla   za.h[w8, 5], {z22.h - z23.h}, z14.h  // 11000001-01101110-00011110-11000101
// CHECK-INST: bfmla   za.h[w8, 5, vgx2], { z22.h, z23.h }, z14.h
// CHECK-ENCODING: [0xc5,0x1e,0x6e,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c16e1ec5 <unknown>

bfmla   za.h[w11, 2, vgx2], {z9.h, z10.h}, z1.h  // 11000001-01100001-01111101-00100010
// CHECK-INST: bfmla   za.h[w11, 2, vgx2], { z9.h, z10.h }, z1.h
// CHECK-ENCODING: [0x22,0x7d,0x61,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1617d22 <unknown>

bfmla   za.h[w11, 2], {z9.h - z10.h}, z1.h  // 11000001-01100001-01111101-00100010
// CHECK-INST: bfmla   za.h[w11, 2, vgx2], { z9.h, z10.h }, z1.h
// CHECK-ENCODING: [0x22,0x7d,0x61,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1617d22 <unknown>

bfmla   za.h[w9, 7, vgx2], {z12.h, z13.h}, z11.h  // 11000001-01101011-00111101-10000111
// CHECK-INST: bfmla   za.h[w9, 7, vgx2], { z12.h, z13.h }, z11.h
// CHECK-ENCODING: [0x87,0x3d,0x6b,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c16b3d87 <unknown>

bfmla   za.h[w9, 7], {z12.h - z13.h}, z11.h  // 11000001-01101011-00111101-10000111
// CHECK-INST: bfmla   za.h[w9, 7, vgx2], { z12.h, z13.h }, z11.h
// CHECK-ENCODING: [0x87,0x3d,0x6b,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c16b3d87 <unknown>

bfmla   za.h[w8, 0, vgx2], {z0.h, z1.h}, z0.h[0]  // 11000001-00010000-00010000-00100000
// CHECK-INST: bfmla   za.h[w8, 0, vgx2], { z0.h, z1.h }, z0.h[0]
// CHECK-ENCODING: [0x20,0x10,0x10,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1101020 <unknown>

bfmla   za.h[w8, 0], {z0.h - z1.h}, z0.h[0]  // 11000001-00010000-00010000-00100000
// CHECK-INST: bfmla   za.h[w8, 0, vgx2], { z0.h, z1.h }, z0.h[0]
// CHECK-ENCODING: [0x20,0x10,0x10,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1101020 <unknown>

bfmla   za.h[w10, 5, vgx2], {z10.h, z11.h}, z5.h[2]  // 11000001-00010101-01010101-01100101
// CHECK-INST: bfmla   za.h[w10, 5, vgx2], { z10.h, z11.h }, z5.h[2]
// CHECK-ENCODING: [0x65,0x55,0x15,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1155565 <unknown>

bfmla   za.h[w10, 5], {z10.h - z11.h}, z5.h[2]  // 11000001-00010101-01010101-01100101
// CHECK-INST: bfmla   za.h[w10, 5, vgx2], { z10.h, z11.h }, z5.h[2]
// CHECK-ENCODING: [0x65,0x55,0x15,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1155565 <unknown>

bfmla   za.h[w11, 7, vgx2], {z12.h, z13.h}, z8.h[6]  // 11000001-00011000-01111101-10100111
// CHECK-INST: bfmla   za.h[w11, 7, vgx2], { z12.h, z13.h }, z8.h[6]
// CHECK-ENCODING: [0xa7,0x7d,0x18,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1187da7 <unknown>

bfmla   za.h[w11, 7], {z12.h - z13.h}, z8.h[6]  // 11000001-00011000-01111101-10100111
// CHECK-INST: bfmla   za.h[w11, 7, vgx2], { z12.h, z13.h }, z8.h[6]
// CHECK-ENCODING: [0xa7,0x7d,0x18,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1187da7 <unknown>

bfmla   za.h[w11, 7, vgx2], {z30.h, z31.h}, z15.h[7]  // 11000001-00011111-01111111-11101111
// CHECK-INST: bfmla   za.h[w11, 7, vgx2], { z30.h, z31.h }, z15.h[7]
// CHECK-ENCODING: [0xef,0x7f,0x1f,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c11f7fef <unknown>

bfmla   za.h[w11, 7], {z30.h - z31.h}, z15.h[7]  // 11000001-00011111-01111111-11101111
// CHECK-INST: bfmla   za.h[w11, 7, vgx2], { z30.h, z31.h }, z15.h[7]
// CHECK-ENCODING: [0xef,0x7f,0x1f,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c11f7fef <unknown>

bfmla   za.h[w8, 5, vgx2], {z16.h, z17.h}, z0.h[6]  // 11000001-00010000-00011110-00100101
// CHECK-INST: bfmla   za.h[w8, 5, vgx2], { z16.h, z17.h }, z0.h[6]
// CHECK-ENCODING: [0x25,0x1e,0x10,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1101e25 <unknown>

bfmla   za.h[w8, 5], {z16.h - z17.h}, z0.h[6]  // 11000001-00010000-00011110-00100101
// CHECK-INST: bfmla   za.h[w8, 5, vgx2], { z16.h, z17.h }, z0.h[6]
// CHECK-ENCODING: [0x25,0x1e,0x10,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1101e25 <unknown>

bfmla   za.h[w8, 1, vgx2], {z0.h, z1.h}, z14.h[2]  // 11000001-00011110-00010100-00100001
// CHECK-INST: bfmla   za.h[w8, 1, vgx2], { z0.h, z1.h }, z14.h[2]
// CHECK-ENCODING: [0x21,0x14,0x1e,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c11e1421 <unknown>

bfmla   za.h[w8, 1], {z0.h - z1.h}, z14.h[2]  // 11000001-00011110-00010100-00100001
// CHECK-INST: bfmla   za.h[w8, 1, vgx2], { z0.h, z1.h }, z14.h[2]
// CHECK-ENCODING: [0x21,0x14,0x1e,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c11e1421 <unknown>

bfmla   za.h[w10, 0, vgx2], {z18.h, z19.h}, z4.h[3]  // 11000001-00010100-01010110-01101000
// CHECK-INST: bfmla   za.h[w10, 0, vgx2], { z18.h, z19.h }, z4.h[3]
// CHECK-ENCODING: [0x68,0x56,0x14,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1145668 <unknown>

bfmla   za.h[w10, 0], {z18.h - z19.h}, z4.h[3]  // 11000001-00010100-01010110-01101000
// CHECK-INST: bfmla   za.h[w10, 0, vgx2], { z18.h, z19.h }, z4.h[3]
// CHECK-ENCODING: [0x68,0x56,0x14,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1145668 <unknown>

bfmla   za.h[w8, 0, vgx2], {z12.h, z13.h}, z2.h[4]  // 11000001-00010010-00011001-10100000
// CHECK-INST: bfmla   za.h[w8, 0, vgx2], { z12.h, z13.h }, z2.h[4]
// CHECK-ENCODING: [0xa0,0x19,0x12,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c11219a0 <unknown>

bfmla   za.h[w8, 0], {z12.h - z13.h}, z2.h[4]  // 11000001-00010010-00011001-10100000
// CHECK-INST: bfmla   za.h[w8, 0, vgx2], { z12.h, z13.h }, z2.h[4]
// CHECK-ENCODING: [0xa0,0x19,0x12,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c11219a0 <unknown>

bfmla   za.h[w10, 1, vgx2], {z0.h, z1.h}, z10.h[4]  // 11000001-00011010-01011000-00100001
// CHECK-INST: bfmla   za.h[w10, 1, vgx2], { z0.h, z1.h }, z10.h[4]
// CHECK-ENCODING: [0x21,0x58,0x1a,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c11a5821 <unknown>

bfmla   za.h[w10, 1], {z0.h - z1.h}, z10.h[4]  // 11000001-00011010-01011000-00100001
// CHECK-INST: bfmla   za.h[w10, 1, vgx2], { z0.h, z1.h }, z10.h[4]
// CHECK-ENCODING: [0x21,0x58,0x1a,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c11a5821 <unknown>

bfmla   za.h[w8, 5, vgx2], {z22.h, z23.h}, z14.h[5]  // 11000001-00011110-00011010-11101101
// CHECK-INST: bfmla   za.h[w8, 5, vgx2], { z22.h, z23.h }, z14.h[5]
// CHECK-ENCODING: [0xed,0x1a,0x1e,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c11e1aed <unknown>

bfmla   za.h[w8, 5], {z22.h - z23.h}, z14.h[5]  // 11000001-00011110-00011010-11101101
// CHECK-INST: bfmla   za.h[w8, 5, vgx2], { z22.h, z23.h }, z14.h[5]
// CHECK-ENCODING: [0xed,0x1a,0x1e,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c11e1aed <unknown>

bfmla   za.h[w11, 2, vgx2], {z8.h, z9.h}, z1.h[2]  // 11000001-00010001-01110101-00100010
// CHECK-INST: bfmla   za.h[w11, 2, vgx2], { z8.h, z9.h }, z1.h[2]
// CHECK-ENCODING: [0x22,0x75,0x11,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1117522 <unknown>

bfmla   za.h[w11, 2], {z8.h - z9.h}, z1.h[2]  // 11000001-00010001-01110101-00100010
// CHECK-INST: bfmla   za.h[w11, 2, vgx2], { z8.h, z9.h }, z1.h[2]
// CHECK-ENCODING: [0x22,0x75,0x11,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1117522 <unknown>

bfmla   za.h[w9, 7, vgx2], {z12.h, z13.h}, z11.h[4]  // 11000001-00011011-00111001-10100111
// CHECK-INST: bfmla   za.h[w9, 7, vgx2], { z12.h, z13.h }, z11.h[4]
// CHECK-ENCODING: [0xa7,0x39,0x1b,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c11b39a7 <unknown>

bfmla   za.h[w9, 7], {z12.h - z13.h}, z11.h[4]  // 11000001-00011011-00111001-10100111
// CHECK-INST: bfmla   za.h[w9, 7, vgx2], { z12.h, z13.h }, z11.h[4]
// CHECK-ENCODING: [0xa7,0x39,0x1b,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c11b39a7 <unknown>

bfmla   za.h[w8, 0, vgx2], {z0.h, z1.h}, {z0.h, z1.h}  // 11000001, 11100000-00010000-00001000
// CHECK-INST: bfmla   za.h[w8, 0, vgx2], { z0.h, z1.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x08,0x10,0xe0,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e01008 <unknown>

bfmla   za.h[w8, 0], {z0.h - z1.h}, {z0.h - z1.h}  // 11000001-11100000-00010000-00001000
// CHECK-INST: bfmla   za.h[w8, 0, vgx2], { z0.h, z1.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x08,0x10,0xe0,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e01008 <unknown>

bfmla   za.h[w10, 5, vgx2], {z10.h, z11.h}, {z20.h, z21.h}  // 11000001, 11110100-01010001-01001101
// CHECK-INST: bfmla   za.h[w10, 5, vgx2], { z10.h, z11.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x4d,0x51,0xf4,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1f4514d <unknown>

bfmla   za.h[w10, 5], {z10.h - z11.h}, {z20.h - z21.h}  // 11000001-11110100-01010001-01001101
// CHECK-INST: bfmla   za.h[w10, 5, vgx2], { z10.h, z11.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x4d,0x51,0xf4,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1f4514d <unknown>

bfmla   za.h[w11, 7, vgx2], {z12.h, z13.h}, {z8.h, z9.h}  // 11000001, 11101000-01110001-10001111
// CHECK-INST: bfmla   za.h[w11, 7, vgx2], { z12.h, z13.h }, { z8.h, z9.h }
// CHECK-ENCODING: [0x8f,0x71,0xe8,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e8718f <unknown>

bfmla   za.h[w11, 7], {z12.h - z13.h}, {z8.h - z9.h}  // 11000001-11101000-01110001-10001111
// CHECK-INST: bfmla   za.h[w11, 7, vgx2], { z12.h, z13.h }, { z8.h, z9.h }
// CHECK-ENCODING: [0x8f,0x71,0xe8,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e8718f <unknown>

bfmla   za.h[w11, 7, vgx2], {z30.h, z31.h}, {z30.h, z31.h}  // 11000001, 11111110-01110011-11001111
// CHECK-INST: bfmla   za.h[w11, 7, vgx2], { z30.h, z31.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xcf,0x73,0xfe,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1fe73cf <unknown>

bfmla   za.h[w11, 7], {z30.h - z31.h}, {z30.h - z31.h}  // 11000001-11111110-01110011-11001111
// CHECK-INST: bfmla   za.h[w11, 7, vgx2], { z30.h, z31.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xcf,0x73,0xfe,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1fe73cf <unknown>

bfmla   za.h[w8, 5, vgx2], {z16.h, z17.h}, {z16.h, z17.h}  // 11000001, 11110000-00010010-00001101
// CHECK-INST: bfmla   za.h[w8, 5, vgx2], { z16.h, z17.h }, { z16.h, z17.h }
// CHECK-ENCODING: [0x0d,0x12,0xf0,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1f0120d <unknown>

bfmla   za.h[w8, 5], {z16.h - z17.h}, {z16.h - z17.h}  // 11000001-11110000-00010010-00001101
// CHECK-INST: bfmla   za.h[w8, 5, vgx2], { z16.h, z17.h }, { z16.h, z17.h }
// CHECK-ENCODING: [0x0d,0x12,0xf0,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1f0120d <unknown>

bfmla   za.h[w8, 1, vgx2], {z0.h, z1.h}, {z30.h, z31.h}  // 11000001, 11111110-00010000-00001001
// CHECK-INST: bfmla   za.h[w8, 1, vgx2], { z0.h, z1.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0x09,0x10,0xfe,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1fe1009 <unknown>

bfmla   za.h[w8, 1], {z0.h - z1.h}, {z30.h - z31.h}  // 11000001-11111110-00010000-00001001
// CHECK-INST: bfmla   za.h[w8, 1, vgx2], { z0.h, z1.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0x09,0x10,0xfe,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1fe1009 <unknown>

bfmla   za.h[w10, 0, vgx2], {z18.h, z19.h}, {z20.h, z21.h}  // 11000001, 11110100-01010010-01001000
// CHECK-INST: bfmla   za.h[w10, 0, vgx2], { z18.h, z19.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x48,0x52,0xf4,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1f45248 <unknown>

bfmla   za.h[w10, 0], {z18.h - z19.h}, {z20.h - z21.h}  // 11000001-11110100-01010010-01001000
// CHECK-INST: bfmla   za.h[w10, 0, vgx2], { z18.h, z19.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x48,0x52,0xf4,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1f45248 <unknown>

bfmla   za.h[w8, 0, vgx2], {z12.h, z13.h}, {z2.h, z3.h}  // 11000001, 11100010-00010001-10001000
// CHECK-INST: bfmla   za.h[w8, 0, vgx2], { z12.h, z13.h }, { z2.h, z3.h }
// CHECK-ENCODING: [0x88,0x11,0xe2,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e21188 <unknown>

bfmla   za.h[w8, 0], {z12.h - z13.h}, {z2.h - z3.h}  // 11000001-11100010-00010001-10001000
// CHECK-INST: bfmla   za.h[w8, 0, vgx2], { z12.h, z13.h }, { z2.h, z3.h }
// CHECK-ENCODING: [0x88,0x11,0xe2,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e21188 <unknown>

bfmla   za.h[w10, 1, vgx2], {z0.h, z1.h}, {z26.h, z27.h}  // 11000001, 11111010-01010000-00001001
// CHECK-INST: bfmla   za.h[w10, 1, vgx2], { z0.h, z1.h }, { z26.h, z27.h }
// CHECK-ENCODING: [0x09,0x50,0xfa,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1fa5009 <unknown>

bfmla   za.h[w10, 1], {z0.h - z1.h}, {z26.h - z27.h}  // 11000001-11111010-01010000-00001001
// CHECK-INST: bfmla   za.h[w10, 1, vgx2], { z0.h, z1.h }, { z26.h, z27.h }
// CHECK-ENCODING: [0x09,0x50,0xfa,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1fa5009 <unknown>

bfmla   za.h[w8, 5, vgx2], {z22.h, z23.h}, {z30.h, z31.h}  // 11000001, 11111110-00010010-11001101
// CHECK-INST: bfmla   za.h[w8, 5, vgx2], { z22.h, z23.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xcd,0x12,0xfe,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1fe12cd <unknown>

bfmla   za.h[w8, 5], {z22.h - z23.h}, {z30.h - z31.h}  // 11000001-11111110-00010010-11001101
// CHECK-INST: bfmla   za.h[w8, 5, vgx2], { z22.h, z23.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xcd,0x12,0xfe,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1fe12cd <unknown>

bfmla   za.h[w11, 2, vgx2], {z8.h, z9.h}, {z0.h, z1.h}  // 11000001, 11100000-01110001-00001010
// CHECK-INST: bfmla   za.h[w11, 2, vgx2], { z8.h, z9.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x0a,0x71,0xe0,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e0710a <unknown>

bfmla   za.h[w11, 2], {z8.h - z9.h}, {z0.h - z1.h}  // 11000001-11100000-01110001-00001010
// CHECK-INST: bfmla   za.h[w11, 2, vgx2], { z8.h, z9.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x0a,0x71,0xe0,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e0710a <unknown>

bfmla   za.h[w9, 7, vgx2], {z12.h, z13.h}, {z10.h, z11.h}  // 11000001, 11101010-00110001-10001111
// CHECK-INST: bfmla   za.h[w9, 7, vgx2], { z12.h, z13.h }, { z10.h, z11.h }
// CHECK-ENCODING: [0x8f,0x31,0xea,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1ea318f <unknown>

bfmla   za.h[w9, 7], {z12.h - z13.h}, {z10.h - z11.h}  // 11000001-11101010-00110001-10001111
// CHECK-INST: bfmla   za.h[w9, 7, vgx2], { z12.h, z13.h }, { z10.h, z11.h }
// CHECK-ENCODING: [0x8f,0x31,0xea,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1ea318f <unknown>

bfmla   za.h[w8, 0, vgx4], {z0.h - z3.h}, z0.h  // 11000001-01110000-00011100-00000000
// CHECK-INST: bfmla   za.h[w8, 0, vgx4], { z0.h - z3.h }, z0.h
// CHECK-ENCODING: [0x00,0x1c,0x70,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1701c00 <unknown>

bfmla   za.h[w8, 0], {z0.h - z3.h}, z0.h  // 11000001-01110000-00011100-00000000
// CHECK-INST: bfmla   za.h[w8, 0, vgx4], { z0.h - z3.h }, z0.h
// CHECK-ENCODING: [0x00,0x1c,0x70,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1701c00 <unknown>

bfmla   za.h[w10, 5, vgx4], {z10.h - z13.h}, z5.h  // 11000001-01110101-01011101-01000101
// CHECK-INST: bfmla   za.h[w10, 5, vgx4], { z10.h - z13.h }, z5.h
// CHECK-ENCODING: [0x45,0x5d,0x75,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1755d45 <unknown>

bfmla   za.h[w10, 5], {z10.h - z13.h}, z5.h  // 11000001-01110101-01011101-01000101
// CHECK-INST: bfmla   za.h[w10, 5, vgx4], { z10.h - z13.h }, z5.h
// CHECK-ENCODING: [0x45,0x5d,0x75,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1755d45 <unknown>

bfmla   za.h[w11, 7, vgx4], {z13.h - z16.h}, z8.h  // 11000001-01111000-01111101-10100111
// CHECK-INST: bfmla   za.h[w11, 7, vgx4], { z13.h - z16.h }, z8.h
// CHECK-ENCODING: [0xa7,0x7d,0x78,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1787da7 <unknown>

bfmla   za.h[w11, 7], {z13.h - z16.h}, z8.h  // 11000001-01111000-01111101-10100111
// CHECK-INST: bfmla   za.h[w11, 7, vgx4], { z13.h - z16.h }, z8.h
// CHECK-ENCODING: [0xa7,0x7d,0x78,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1787da7 <unknown>

bfmla   za.h[w11, 7, vgx4], {z31.h, z0.h, z1.h, z2.h}, z15.h  // 11000001-01111111-01111111-11100111
// CHECK-INST: bfmla   za.h[w11, 7, vgx4], { z31.h, z0.h, z1.h, z2.h }, z15.h
// CHECK-ENCODING: [0xe7,0x7f,0x7f,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c17f7fe7 <unknown>

bfmla   za.h[w11, 7], {z31.h, z0.h, z1.h, z2.h}, z15.h  // 11000001-01111111-01111111-11100111
// CHECK-INST: bfmla   za.h[w11, 7, vgx4], { z31.h, z0.h, z1.h, z2.h }, z15.h
// CHECK-ENCODING: [0xe7,0x7f,0x7f,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c17f7fe7 <unknown>

bfmla   za.h[w8, 5, vgx4], {z17.h - z20.h}, z0.h  // 11000001-01110000-00011110-00100101
// CHECK-INST: bfmla   za.h[w8, 5, vgx4], { z17.h - z20.h }, z0.h
// CHECK-ENCODING: [0x25,0x1e,0x70,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1701e25 <unknown>

bfmla   za.h[w8, 5], {z17.h - z20.h}, z0.h  // 11000001-01110000-00011110-00100101
// CHECK-INST: bfmla   za.h[w8, 5, vgx4], { z17.h - z20.h }, z0.h
// CHECK-ENCODING: [0x25,0x1e,0x70,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1701e25 <unknown>

bfmla   za.h[w8, 1, vgx4], {z1.h - z4.h}, z14.h  // 11000001-01111110-00011100-00100001
// CHECK-INST: bfmla   za.h[w8, 1, vgx4], { z1.h - z4.h }, z14.h
// CHECK-ENCODING: [0x21,0x1c,0x7e,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c17e1c21 <unknown>

bfmla   za.h[w8, 1], {z1.h - z4.h}, z14.h  // 11000001-01111110-00011100-00100001
// CHECK-INST: bfmla   za.h[w8, 1, vgx4], { z1.h - z4.h }, z14.h
// CHECK-ENCODING: [0x21,0x1c,0x7e,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c17e1c21 <unknown>

bfmla   za.h[w10, 0, vgx4], {z19.h - z22.h}, z4.h  // 11000001-01110100-01011110-01100000
// CHECK-INST: bfmla   za.h[w10, 0, vgx4], { z19.h - z22.h }, z4.h
// CHECK-ENCODING: [0x60,0x5e,0x74,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1745e60 <unknown>

bfmla   za.h[w10, 0], {z19.h - z22.h}, z4.h  // 11000001-01110100-01011110-01100000
// CHECK-INST: bfmla   za.h[w10, 0, vgx4], { z19.h - z22.h }, z4.h
// CHECK-ENCODING: [0x60,0x5e,0x74,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1745e60 <unknown>

bfmla   za.h[w8, 0, vgx4], {z12.h - z15.h}, z2.h  // 11000001-01110010-00011101-10000000
// CHECK-INST: bfmla   za.h[w8, 0, vgx4], { z12.h - z15.h }, z2.h
// CHECK-ENCODING: [0x80,0x1d,0x72,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1721d80 <unknown>

bfmla   za.h[w8, 0], {z12.h - z15.h}, z2.h  // 11000001-01110010-00011101-10000000
// CHECK-INST: bfmla   za.h[w8, 0, vgx4], { z12.h - z15.h }, z2.h
// CHECK-ENCODING: [0x80,0x1d,0x72,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1721d80 <unknown>

bfmla   za.h[w10, 1, vgx4], {z1.h - z4.h}, z10.h  // 11000001-01111010-01011100-00100001
// CHECK-INST: bfmla   za.h[w10, 1, vgx4], { z1.h - z4.h }, z10.h
// CHECK-ENCODING: [0x21,0x5c,0x7a,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c17a5c21 <unknown>

bfmla   za.h[w10, 1], {z1.h - z4.h}, z10.h  // 11000001-01111010-01011100-00100001
// CHECK-INST: bfmla   za.h[w10, 1, vgx4], { z1.h - z4.h }, z10.h
// CHECK-ENCODING: [0x21,0x5c,0x7a,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c17a5c21 <unknown>

bfmla   za.h[w8, 5, vgx4], {z22.h - z25.h}, z14.h  // 11000001-01111110-00011110-11000101
// CHECK-INST: bfmla   za.h[w8, 5, vgx4], { z22.h - z25.h }, z14.h
// CHECK-ENCODING: [0xc5,0x1e,0x7e,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c17e1ec5 <unknown>

bfmla   za.h[w8, 5], {z22.h - z25.h}, z14.h  // 11000001-01111110-00011110-11000101
// CHECK-INST: bfmla   za.h[w8, 5, vgx4], { z22.h - z25.h }, z14.h
// CHECK-ENCODING: [0xc5,0x1e,0x7e,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c17e1ec5 <unknown>

bfmla   za.h[w11, 2, vgx4], {z9.h - z12.h}, z1.h  // 11000001-01110001-01111101-00100010
// CHECK-INST: bfmla   za.h[w11, 2, vgx4], { z9.h - z12.h }, z1.h
// CHECK-ENCODING: [0x22,0x7d,0x71,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1717d22 <unknown>

bfmla   za.h[w11, 2], {z9.h - z12.h}, z1.h  // 11000001-01110001-01111101-00100010
// CHECK-INST: bfmla   za.h[w11, 2, vgx4], { z9.h - z12.h }, z1.h
// CHECK-ENCODING: [0x22,0x7d,0x71,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1717d22 <unknown>

bfmla   za.h[w9, 7, vgx4], {z12.h - z15.h}, z11.h  // 11000001-01111011-00111101-10000111
// CHECK-INST: bfmla   za.h[w9, 7, vgx4], { z12.h - z15.h }, z11.h
// CHECK-ENCODING: [0x87,0x3d,0x7b,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c17b3d87 <unknown>

bfmla   za.h[w9, 7], {z12.h - z15.h}, z11.h  // 11000001-01111011-00111101-10000111
// CHECK-INST: bfmla   za.h[w9, 7, vgx4], { z12.h - z15.h }, z11.h
// CHECK-ENCODING: [0x87,0x3d,0x7b,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c17b3d87 <unknown>

bfmla   za.h[w8, 0, vgx4], {z0.h - z3.h}, z0.h[0]  // 11000001-00010000-10010000-00100000
// CHECK-INST: bfmla   za.h[w8, 0, vgx4], { z0.h - z3.h }, z0.h[0]
// CHECK-ENCODING: [0x20,0x90,0x10,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1109020 <unknown>

bfmla   za.h[w8, 0], {z0.h - z3.h}, z0.h[0]  // 11000001-00010000-10010000-00100000
// CHECK-INST: bfmla   za.h[w8, 0, vgx4], { z0.h - z3.h }, z0.h[0]
// CHECK-ENCODING: [0x20,0x90,0x10,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1109020 <unknown>

bfmla   za.h[w10, 5, vgx4], {z8.h - z11.h}, z5.h[2]  // 11000001-00010101-11010101-00100101
// CHECK-INST: bfmla   za.h[w10, 5, vgx4], { z8.h - z11.h }, z5.h[2]
// CHECK-ENCODING: [0x25,0xd5,0x15,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c115d525 <unknown>

bfmla   za.h[w10, 5], {z8.h - z11.h}, z5.h[2]  // 11000001-00010101-11010101-00100101
// CHECK-INST: bfmla   za.h[w10, 5, vgx4], { z8.h - z11.h }, z5.h[2]
// CHECK-ENCODING: [0x25,0xd5,0x15,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c115d525 <unknown>

bfmla   za.h[w11, 7, vgx4], {z12.h - z15.h}, z8.h[6]  // 11000001-00011000-11111101-10100111
// CHECK-INST: bfmla   za.h[w11, 7, vgx4], { z12.h - z15.h }, z8.h[6]
// CHECK-ENCODING: [0xa7,0xfd,0x18,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c118fda7 <unknown>

bfmla   za.h[w11, 7], {z12.h - z15.h}, z8.h[6]  // 11000001-00011000-11111101-10100111
// CHECK-INST: bfmla   za.h[w11, 7, vgx4], { z12.h - z15.h }, z8.h[6]
// CHECK-ENCODING: [0xa7,0xfd,0x18,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c118fda7 <unknown>

bfmla   za.h[w11, 7, vgx4], {z28.h - z31.h}, z15.h[7]  // 11000001-00011111-11111111-10101111
// CHECK-INST: bfmla   za.h[w11, 7, vgx4], { z28.h - z31.h }, z15.h[7]
// CHECK-ENCODING: [0xaf,0xff,0x1f,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c11fffaf <unknown>

bfmla   za.h[w11, 7], {z28.h - z31.h}, z15.h[7]  // 11000001-00011111-11111111-10101111
// CHECK-INST: bfmla   za.h[w11, 7, vgx4], { z28.h - z31.h }, z15.h[7]
// CHECK-ENCODING: [0xaf,0xff,0x1f,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c11fffaf <unknown>

bfmla   za.h[w8, 5, vgx4], {z16.h - z19.h}, z0.h[6]  // 11000001-00010000-10011110-00100101
// CHECK-INST: bfmla   za.h[w8, 5, vgx4], { z16.h - z19.h }, z0.h[6]
// CHECK-ENCODING: [0x25,0x9e,0x10,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1109e25 <unknown>

bfmla   za.h[w8, 5], {z16.h - z19.h}, z0.h[6]  // 11000001-00010000-10011110-00100101
// CHECK-INST: bfmla   za.h[w8, 5, vgx4], { z16.h - z19.h }, z0.h[6]
// CHECK-ENCODING: [0x25,0x9e,0x10,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1109e25 <unknown>

bfmla   za.h[w8, 1, vgx4], {z0.h - z3.h}, z14.h[2]  // 11000001-00011110-10010100-00100001
// CHECK-INST: bfmla   za.h[w8, 1, vgx4], { z0.h - z3.h }, z14.h[2]
// CHECK-ENCODING: [0x21,0x94,0x1e,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c11e9421 <unknown>

bfmla   za.h[w8, 1], {z0.h - z3.h}, z14.h[2]  // 11000001-00011110-10010100-00100001
// CHECK-INST: bfmla   za.h[w8, 1, vgx4], { z0.h - z3.h }, z14.h[2]
// CHECK-ENCODING: [0x21,0x94,0x1e,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c11e9421 <unknown>

bfmla   za.h[w10, 0, vgx4], {z16.h - z19.h}, z4.h[3]  // 11000001-00010100-11010110-00101000
// CHECK-INST: bfmla   za.h[w10, 0, vgx4], { z16.h - z19.h }, z4.h[3]
// CHECK-ENCODING: [0x28,0xd6,0x14,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c114d628 <unknown>

bfmla   za.h[w10, 0], {z16.h - z19.h}, z4.h[3]  // 11000001-00010100-11010110-00101000
// CHECK-INST: bfmla   za.h[w10, 0, vgx4], { z16.h - z19.h }, z4.h[3]
// CHECK-ENCODING: [0x28,0xd6,0x14,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c114d628 <unknown>

bfmla   za.h[w8, 0, vgx4], {z12.h - z15.h}, z2.h[4]  // 11000001-00010010-10011001-10100000
// CHECK-INST: bfmla   za.h[w8, 0, vgx4], { z12.h - z15.h }, z2.h[4]
// CHECK-ENCODING: [0xa0,0x99,0x12,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c11299a0 <unknown>

bfmla   za.h[w8, 0], {z12.h - z15.h}, z2.h[4]  // 11000001-00010010-10011001-10100000
// CHECK-INST: bfmla   za.h[w8, 0, vgx4], { z12.h - z15.h }, z2.h[4]
// CHECK-ENCODING: [0xa0,0x99,0x12,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c11299a0 <unknown>

bfmla   za.h[w10, 1, vgx4], {z0.h - z3.h}, z10.h[4]  // 11000001-00011010-11011000-00100001
// CHECK-INST: bfmla   za.h[w10, 1, vgx4], { z0.h - z3.h }, z10.h[4]
// CHECK-ENCODING: [0x21,0xd8,0x1a,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c11ad821 <unknown>

bfmla   za.h[w10, 1], {z0.h - z3.h}, z10.h[4]  // 11000001-00011010-11011000-00100001
// CHECK-INST: bfmla   za.h[w10, 1, vgx4], { z0.h - z3.h }, z10.h[4]
// CHECK-ENCODING: [0x21,0xd8,0x1a,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c11ad821 <unknown>

bfmla   za.h[w8, 5, vgx4], {z20.h - z23.h}, z14.h[5]  // 11000001-00011110-10011010-10101101
// CHECK-INST: bfmla   za.h[w8, 5, vgx4], { z20.h - z23.h }, z14.h[5]
// CHECK-ENCODING: [0xad,0x9a,0x1e,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c11e9aad <unknown>

bfmla   za.h[w8, 5], {z20.h - z23.h}, z14.h[5]  // 11000001-00011110-10011010-10101101
// CHECK-INST: bfmla   za.h[w8, 5, vgx4], { z20.h - z23.h }, z14.h[5]
// CHECK-ENCODING: [0xad,0x9a,0x1e,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c11e9aad <unknown>

bfmla   za.h[w11, 2, vgx4], {z8.h - z11.h}, z1.h[2]  // 11000001-00010001-11110101-00100010
// CHECK-INST: bfmla   za.h[w11, 2, vgx4], { z8.h - z11.h }, z1.h[2]
// CHECK-ENCODING: [0x22,0xf5,0x11,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c111f522 <unknown>

bfmla   za.h[w11, 2], {z8.h - z11.h}, z1.h[2]  // 11000001-00010001-11110101-00100010
// CHECK-INST: bfmla   za.h[w11, 2, vgx4], { z8.h - z11.h }, z1.h[2]
// CHECK-ENCODING: [0x22,0xf5,0x11,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c111f522 <unknown>

bfmla   za.h[w9, 7, vgx4], {z12.h - z15.h}, z11.h[4]  // 11000001-00011011-10111001-10100111
// CHECK-INST: bfmla   za.h[w9, 7, vgx4], { z12.h - z15.h }, z11.h[4]
// CHECK-ENCODING: [0xa7,0xb9,0x1b,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c11bb9a7 <unknown>

bfmla   za.h[w9, 7], {z12.h - z15.h}, z11.h[4]  // 11000001-00011011-10111001-10100111
// CHECK-INST: bfmla   za.h[w9, 7, vgx4], { z12.h - z15.h }, z11.h[4]
// CHECK-ENCODING: [0xa7,0xb9,0x1b,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c11bb9a7 <unknown>

bfmla   za.h[w8, 0, vgx4], {z0.h - z3.h}, {z0.h - z3.h}  // 11000001-11100001-00010000-00001000
// CHECK-INST: bfmla   za.h[w8, 0, vgx4], { z0.h - z3.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x08,0x10,0xe1,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e11008 <unknown>

bfmla   za.h[w8, 0], {z0.h - z3.h}, {z0.h - z3.h}  // 11000001-11100001-00010000-00001000
// CHECK-INST: bfmla   za.h[w8, 0, vgx4], { z0.h - z3.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x08,0x10,0xe1,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e11008 <unknown>

bfmla   za.h[w10, 5, vgx4], {z8.h - z11.h}, {z20.h - z23.h}  // 11000001-11110101-01010001-00001101
// CHECK-INST: bfmla   za.h[w10, 5, vgx4], { z8.h - z11.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x0d,0x51,0xf5,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1f5510d <unknown>

bfmla   za.h[w10, 5], {z8.h - z11.h}, {z20.h - z23.h}  // 11000001-11110101-01010001-00001101
// CHECK-INST: bfmla   za.h[w10, 5, vgx4], { z8.h - z11.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x0d,0x51,0xf5,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1f5510d <unknown>

bfmla   za.h[w11, 7, vgx4], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-11101001-01110001-10001111
// CHECK-INST: bfmla   za.h[w11, 7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x8f,0x71,0xe9,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e9718f <unknown>

bfmla   za.h[w11, 7], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-11101001-01110001-10001111
// CHECK-INST: bfmla   za.h[w11, 7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x8f,0x71,0xe9,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e9718f <unknown>

bfmla   za.h[w11, 7, vgx4], {z28.h - z31.h}, {z28.h - z31.h}  // 11000001-11111101-01110011-10001111
// CHECK-INST: bfmla   za.h[w11, 7, vgx4], { z28.h - z31.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x8f,0x73,0xfd,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1fd738f <unknown>

bfmla   za.h[w11, 7], {z28.h - z31.h}, {z28.h - z31.h}  // 11000001-11111101-01110011-10001111
// CHECK-INST: bfmla   za.h[w11, 7, vgx4], { z28.h - z31.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x8f,0x73,0xfd,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1fd738f <unknown>

bfmla   za.h[w8, 5, vgx4], {z16.h - z19.h}, {z16.h - z19.h}  // 11000001-11110001-00010010-00001101
// CHECK-INST: bfmla   za.h[w8, 5, vgx4], { z16.h - z19.h }, { z16.h - z19.h }
// CHECK-ENCODING: [0x0d,0x12,0xf1,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1f1120d <unknown>

bfmla   za.h[w8, 5], {z16.h - z19.h}, {z16.h - z19.h}  // 11000001-11110001-00010010-00001101
// CHECK-INST: bfmla   za.h[w8, 5, vgx4], { z16.h - z19.h }, { z16.h - z19.h }
// CHECK-ENCODING: [0x0d,0x12,0xf1,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1f1120d <unknown>

bfmla   za.h[w8, 1, vgx4], {z0.h - z3.h}, {z28.h - z31.h}  // 11000001-11111101-00010000-00001001
// CHECK-INST: bfmla   za.h[w8, 1, vgx4], { z0.h - z3.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x09,0x10,0xfd,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1fd1009 <unknown>

bfmla   za.h[w8, 1], {z0.h - z3.h}, {z28.h - z31.h}  // 11000001-11111101-00010000-00001001
// CHECK-INST: bfmla   za.h[w8, 1, vgx4], { z0.h - z3.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x09,0x10,0xfd,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1fd1009 <unknown>

bfmla   za.h[w10, 0, vgx4], {z16.h - z19.h}, {z20.h - z23.h}  // 11000001-11110101-01010010-00001000
// CHECK-INST: bfmla   za.h[w10, 0, vgx4], { z16.h - z19.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x08,0x52,0xf5,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1f55208 <unknown>

bfmla   za.h[w10, 0], {z16.h - z19.h}, {z20.h - z23.h}  // 11000001-11110101-01010010-00001000
// CHECK-INST: bfmla   za.h[w10, 0, vgx4], { z16.h - z19.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x08,0x52,0xf5,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1f55208 <unknown>

bfmla   za.h[w8, 0, vgx4], {z12.h - z15.h}, {z0.h - z3.h}  // 11000001-11100001-00010001-10001000
// CHECK-INST: bfmla   za.h[w8, 0, vgx4], { z12.h - z15.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x88,0x11,0xe1,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e11188 <unknown>

bfmla   za.h[w8, 0], {z12.h - z15.h}, {z0.h - z3.h}  // 11000001-11100001-00010001-10001000
// CHECK-INST: bfmla   za.h[w8, 0, vgx4], { z12.h - z15.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x88,0x11,0xe1,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e11188 <unknown>

bfmla   za.h[w10, 1, vgx4], {z0.h - z3.h}, {z24.h - z27.h}  // 11000001-11111001-01010000-00001001
// CHECK-INST: bfmla   za.h[w10, 1, vgx4], { z0.h - z3.h }, { z24.h - z27.h }
// CHECK-ENCODING: [0x09,0x50,0xf9,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1f95009 <unknown>

bfmla   za.h[w10, 1], {z0.h - z3.h}, {z24.h - z27.h}  // 11000001-11111001-01010000-00001001
// CHECK-INST: bfmla   za.h[w10, 1, vgx4], { z0.h - z3.h }, { z24.h - z27.h }
// CHECK-ENCODING: [0x09,0x50,0xf9,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1f95009 <unknown>

bfmla   za.h[w8, 5, vgx4], {z20.h - z23.h}, {z28.h - z31.h}  // 11000001-11111101-00010010-10001101
// CHECK-INST: bfmla   za.h[w8, 5, vgx4], { z20.h - z23.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x8d,0x12,0xfd,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1fd128d <unknown>

bfmla   za.h[w8, 5], {z20.h - z23.h}, {z28.h - z31.h}  // 11000001-11111101-00010010-10001101
// CHECK-INST: bfmla   za.h[w8, 5, vgx4], { z20.h - z23.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x8d,0x12,0xfd,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1fd128d <unknown>

bfmla   za.h[w11, 2, vgx4], {z8.h - z11.h}, {z0.h - z3.h}  // 11000001-11100001-01110001-00001010
// CHECK-INST: bfmla   za.h[w11, 2, vgx4], { z8.h - z11.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x0a,0x71,0xe1,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e1710a <unknown>

bfmla   za.h[w11, 2], {z8.h - z11.h}, {z0.h - z3.h}  // 11000001-11100001-01110001-00001010
// CHECK-INST: bfmla   za.h[w11, 2, vgx4], { z8.h - z11.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x0a,0x71,0xe1,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e1710a <unknown>

bfmla   za.h[w9, 7, vgx4], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-11101001-00110001-10001111
// CHECK-INST: bfmla   za.h[w9, 7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x8f,0x31,0xe9,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e9318f <unknown>

bfmla   za.h[w9, 7], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-11101001-00110001-10001111
// CHECK-INST: bfmla   za.h[w9, 7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x8f,0x31,0xe9,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e9318f <unknown>
