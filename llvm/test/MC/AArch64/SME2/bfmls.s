// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme-b16b16 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme-b16b16 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme-b16b16 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme-b16b16 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme-b16b16 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme-b16b16 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

bfmls   za.h[w8, 0, vgx2], {z0.h, z1.h}, z0.h  // 11000001-01100000-00011100-00001000
// CHECK-INST: bfmls   za.h[w8, 0, vgx2], { z0.h, z1.h }, z0.h
// CHECK-ENCODING: [0x08,0x1c,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1601c08 <unknown>

bfmls   za.h[w8, 0], {z0.h - z1.h}, z0.h  // 11000001-01100000-00011100-00001000
// CHECK-INST: bfmls   za.h[w8, 0, vgx2], { z0.h, z1.h }, z0.h
// CHECK-ENCODING: [0x08,0x1c,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1601c08 <unknown>

bfmls   za.h[w10, 5, vgx2], {z10.h, z11.h}, z5.h  // 11000001-01100101-01011101-01001101
// CHECK-INST: bfmls   za.h[w10, 5, vgx2], { z10.h, z11.h }, z5.h
// CHECK-ENCODING: [0x4d,0x5d,0x65,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1655d4d <unknown>

bfmls   za.h[w10, 5], {z10.h - z11.h}, z5.h  // 11000001-01100101-01011101-01001101
// CHECK-INST: bfmls   za.h[w10, 5, vgx2], { z10.h, z11.h }, z5.h
// CHECK-ENCODING: [0x4d,0x5d,0x65,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1655d4d <unknown>

bfmls   za.h[w11, 7, vgx2], {z13.h, z14.h}, z8.h  // 11000001-01101000-01111101-10101111
// CHECK-INST: bfmls   za.h[w11, 7, vgx2], { z13.h, z14.h }, z8.h
// CHECK-ENCODING: [0xaf,0x7d,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1687daf <unknown>

bfmls   za.h[w11, 7], {z13.h - z14.h}, z8.h  // 11000001-01101000-01111101-10101111
// CHECK-INST: bfmls   za.h[w11, 7, vgx2], { z13.h, z14.h }, z8.h
// CHECK-ENCODING: [0xaf,0x7d,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1687daf <unknown>

bfmls   za.h[w11, 7, vgx2], {z31.h, z0.h}, z15.h  // 11000001-01101111-01111111-11101111
// CHECK-INST: bfmls   za.h[w11, 7, vgx2], { z31.h, z0.h }, z15.h
// CHECK-ENCODING: [0xef,0x7f,0x6f,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c16f7fef <unknown>

bfmls   za.h[w11, 7], {z31.h - z0.h}, z15.h  // 11000001-01101111-01111111-11101111
// CHECK-INST: bfmls   za.h[w11, 7, vgx2], { z31.h, z0.h }, z15.h
// CHECK-ENCODING: [0xef,0x7f,0x6f,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c16f7fef <unknown>

bfmls   za.h[w8, 5, vgx2], {z17.h, z18.h}, z0.h  // 11000001-01100000-00011110-00101101
// CHECK-INST: bfmls   za.h[w8, 5, vgx2], { z17.h, z18.h }, z0.h
// CHECK-ENCODING: [0x2d,0x1e,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1601e2d <unknown>

bfmls   za.h[w8, 5], {z17.h - z18.h}, z0.h  // 11000001-01100000-00011110-00101101
// CHECK-INST: bfmls   za.h[w8, 5, vgx2], { z17.h, z18.h }, z0.h
// CHECK-ENCODING: [0x2d,0x1e,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1601e2d <unknown>

bfmls   za.h[w8, 1, vgx2], {z1.h, z2.h}, z14.h  // 11000001-01101110-00011100-00101001
// CHECK-INST: bfmls   za.h[w8, 1, vgx2], { z1.h, z2.h }, z14.h
// CHECK-ENCODING: [0x29,0x1c,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c16e1c29 <unknown>

bfmls   za.h[w8, 1], {z1.h - z2.h}, z14.h  // 11000001-01101110-00011100-00101001
// CHECK-INST: bfmls   za.h[w8, 1, vgx2], { z1.h, z2.h }, z14.h
// CHECK-ENCODING: [0x29,0x1c,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c16e1c29 <unknown>

bfmls   za.h[w10, 0, vgx2], {z19.h, z20.h}, z4.h  // 11000001-01100100-01011110-01101000
// CHECK-INST: bfmls   za.h[w10, 0, vgx2], { z19.h, z20.h }, z4.h
// CHECK-ENCODING: [0x68,0x5e,0x64,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1645e68 <unknown>

bfmls   za.h[w10, 0], {z19.h - z20.h}, z4.h  // 11000001-01100100-01011110-01101000
// CHECK-INST: bfmls   za.h[w10, 0, vgx2], { z19.h, z20.h }, z4.h
// CHECK-ENCODING: [0x68,0x5e,0x64,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1645e68 <unknown>

bfmls   za.h[w8, 0, vgx2], {z12.h, z13.h}, z2.h  // 11000001-01100010-00011101-10001000
// CHECK-INST: bfmls   za.h[w8, 0, vgx2], { z12.h, z13.h }, z2.h
// CHECK-ENCODING: [0x88,0x1d,0x62,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1621d88 <unknown>

bfmls   za.h[w8, 0], {z12.h - z13.h}, z2.h  // 11000001-01100010-00011101-10001000
// CHECK-INST: bfmls   za.h[w8, 0, vgx2], { z12.h, z13.h }, z2.h
// CHECK-ENCODING: [0x88,0x1d,0x62,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1621d88 <unknown>

bfmls   za.h[w10, 1, vgx2], {z1.h, z2.h}, z10.h  // 11000001-01101010-01011100-00101001
// CHECK-INST: bfmls   za.h[w10, 1, vgx2], { z1.h, z2.h }, z10.h
// CHECK-ENCODING: [0x29,0x5c,0x6a,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c16a5c29 <unknown>

bfmls   za.h[w10, 1], {z1.h - z2.h}, z10.h  // 11000001-01101010-01011100-00101001
// CHECK-INST: bfmls   za.h[w10, 1, vgx2], { z1.h, z2.h }, z10.h
// CHECK-ENCODING: [0x29,0x5c,0x6a,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c16a5c29 <unknown>

bfmls   za.h[w8, 5, vgx2], {z22.h, z23.h}, z14.h  // 11000001-01101110-00011110-11001101
// CHECK-INST: bfmls   za.h[w8, 5, vgx2], { z22.h, z23.h }, z14.h
// CHECK-ENCODING: [0xcd,0x1e,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c16e1ecd <unknown>

bfmls   za.h[w8, 5], {z22.h - z23.h}, z14.h  // 11000001-01101110-00011110-11001101
// CHECK-INST: bfmls   za.h[w8, 5, vgx2], { z22.h, z23.h }, z14.h
// CHECK-ENCODING: [0xcd,0x1e,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c16e1ecd <unknown>

bfmls   za.h[w11, 2, vgx2], {z9.h, z10.h}, z1.h  // 11000001-01100001-01111101-00101010
// CHECK-INST: bfmls   za.h[w11, 2, vgx2], { z9.h, z10.h }, z1.h
// CHECK-ENCODING: [0x2a,0x7d,0x61,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1617d2a <unknown>

bfmls   za.h[w11, 2], {z9.h - z10.h}, z1.h  // 11000001-01100001-01111101-00101010
// CHECK-INST: bfmls   za.h[w11, 2, vgx2], { z9.h, z10.h }, z1.h
// CHECK-ENCODING: [0x2a,0x7d,0x61,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1617d2a <unknown>

bfmls   za.h[w9, 7, vgx2], {z12.h, z13.h}, z11.h  // 11000001-01101011-00111101-10001111
// CHECK-INST: bfmls   za.h[w9, 7, vgx2], { z12.h, z13.h }, z11.h
// CHECK-ENCODING: [0x8f,0x3d,0x6b,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c16b3d8f <unknown>

bfmls   za.h[w9, 7], {z12.h - z13.h}, z11.h  // 11000001-01101011-00111101-10001111
// CHECK-INST: bfmls   za.h[w9, 7, vgx2], { z12.h, z13.h }, z11.h
// CHECK-ENCODING: [0x8f,0x3d,0x6b,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c16b3d8f <unknown>

bfmls   za.h[w8, 0, vgx2], {z0.h, z1.h}, z0.h[0]  // 11000001-00010000-00010000-00110000
// CHECK-INST: bfmls   za.h[w8, 0, vgx2], { z0.h, z1.h }, z0.h[0]
// CHECK-ENCODING: [0x30,0x10,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1101030 <unknown>

bfmls   za.h[w8, 0], {z0.h - z1.h}, z0.h[0]  // 11000001-00010000-00010000-00110000
// CHECK-INST: bfmls   za.h[w8, 0, vgx2], { z0.h, z1.h }, z0.h[0]
// CHECK-ENCODING: [0x30,0x10,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1101030 <unknown>

bfmls   za.h[w10, 5, vgx2], {z10.h, z11.h}, z5.h[2]  // 11000001-00010101-01010101-01110101
// CHECK-INST: bfmls   za.h[w10, 5, vgx2], { z10.h, z11.h }, z5.h[2]
// CHECK-ENCODING: [0x75,0x55,0x15,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1155575 <unknown>

bfmls   za.h[w10, 5], {z10.h - z11.h}, z5.h[2]  // 11000001-00010101-01010101-01110101
// CHECK-INST: bfmls   za.h[w10, 5, vgx2], { z10.h, z11.h }, z5.h[2]
// CHECK-ENCODING: [0x75,0x55,0x15,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1155575 <unknown>

bfmls   za.h[w11, 7, vgx2], {z12.h, z13.h}, z8.h[6]  // 11000001-00011000-01111101-10110111
// CHECK-INST: bfmls   za.h[w11, 7, vgx2], { z12.h, z13.h }, z8.h[6]
// CHECK-ENCODING: [0xb7,0x7d,0x18,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1187db7 <unknown>

bfmls   za.h[w11, 7], {z12.h - z13.h}, z8.h[6]  // 11000001-00011000-01111101-10110111
// CHECK-INST: bfmls   za.h[w11, 7, vgx2], { z12.h, z13.h }, z8.h[6]
// CHECK-ENCODING: [0xb7,0x7d,0x18,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1187db7 <unknown>

bfmls   za.h[w11, 7, vgx2], {z30.h, z31.h}, z15.h[7]  // 11000001-00011111-01111111-11111111
// CHECK-INST: bfmls   za.h[w11, 7, vgx2], { z30.h, z31.h }, z15.h[7]
// CHECK-ENCODING: [0xff,0x7f,0x1f,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c11f7fff <unknown>

bfmls   za.h[w11, 7], {z30.h - z31.h}, z15.h[7]  // 11000001-00011111-01111111-11111111
// CHECK-INST: bfmls   za.h[w11, 7, vgx2], { z30.h, z31.h }, z15.h[7]
// CHECK-ENCODING: [0xff,0x7f,0x1f,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c11f7fff <unknown>

bfmls   za.h[w8, 5, vgx2], {z16.h, z17.h}, z0.h[6]  // 11000001-00010000-00011110-00110101
// CHECK-INST: bfmls   za.h[w8, 5, vgx2], { z16.h, z17.h }, z0.h[6]
// CHECK-ENCODING: [0x35,0x1e,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1101e35 <unknown>

bfmls   za.h[w8, 5], {z16.h - z17.h}, z0.h[6]  // 11000001-00010000-00011110-00110101
// CHECK-INST: bfmls   za.h[w8, 5, vgx2], { z16.h, z17.h }, z0.h[6]
// CHECK-ENCODING: [0x35,0x1e,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1101e35 <unknown>

bfmls   za.h[w8, 1, vgx2], {z0.h, z1.h}, z14.h[2]  // 11000001-00011110-00010100-00110001
// CHECK-INST: bfmls   za.h[w8, 1, vgx2], { z0.h, z1.h }, z14.h[2]
// CHECK-ENCODING: [0x31,0x14,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c11e1431 <unknown>

bfmls   za.h[w8, 1], {z0.h - z1.h}, z14.h[2]  // 11000001-00011110-00010100-00110001
// CHECK-INST: bfmls   za.h[w8, 1, vgx2], { z0.h, z1.h }, z14.h[2]
// CHECK-ENCODING: [0x31,0x14,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c11e1431 <unknown>

bfmls   za.h[w10, 0, vgx2], {z18.h, z19.h}, z4.h[3]  // 11000001-00010100-01010110-01111000
// CHECK-INST: bfmls   za.h[w10, 0, vgx2], { z18.h, z19.h }, z4.h[3]
// CHECK-ENCODING: [0x78,0x56,0x14,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1145678 <unknown>

bfmls   za.h[w10, 0], {z18.h - z19.h}, z4.h[3]  // 11000001-00010100-01010110-01111000
// CHECK-INST: bfmls   za.h[w10, 0, vgx2], { z18.h, z19.h }, z4.h[3]
// CHECK-ENCODING: [0x78,0x56,0x14,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1145678 <unknown>

bfmls   za.h[w8, 0, vgx2], {z12.h, z13.h}, z2.h[4]  // 11000001-00010010-00011001-10110000
// CHECK-INST: bfmls   za.h[w8, 0, vgx2], { z12.h, z13.h }, z2.h[4]
// CHECK-ENCODING: [0xb0,0x19,0x12,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c11219b0 <unknown>

bfmls   za.h[w8, 0], {z12.h - z13.h}, z2.h[4]  // 11000001-00010010-00011001-10110000
// CHECK-INST: bfmls   za.h[w8, 0, vgx2], { z12.h, z13.h }, z2.h[4]
// CHECK-ENCODING: [0xb0,0x19,0x12,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c11219b0 <unknown>

bfmls   za.h[w10, 1, vgx2], {z0.h, z1.h}, z10.h[4]  // 11000001-00011010-01011000-00110001
// CHECK-INST: bfmls   za.h[w10, 1, vgx2], { z0.h, z1.h }, z10.h[4]
// CHECK-ENCODING: [0x31,0x58,0x1a,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c11a5831 <unknown>

bfmls   za.h[w10, 1], {z0.h - z1.h}, z10.h[4]  // 11000001-00011010-01011000-00110001
// CHECK-INST: bfmls   za.h[w10, 1, vgx2], { z0.h, z1.h }, z10.h[4]
// CHECK-ENCODING: [0x31,0x58,0x1a,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c11a5831 <unknown>

bfmls   za.h[w8, 5, vgx2], {z22.h, z23.h}, z14.h[5]  // 11000001-00011110-00011010-11111101
// CHECK-INST: bfmls   za.h[w8, 5, vgx2], { z22.h, z23.h }, z14.h[5]
// CHECK-ENCODING: [0xfd,0x1a,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c11e1afd <unknown>

bfmls   za.h[w8, 5], {z22.h - z23.h}, z14.h[5]  // 11000001-00011110-00011010-11111101
// CHECK-INST: bfmls   za.h[w8, 5, vgx2], { z22.h, z23.h }, z14.h[5]
// CHECK-ENCODING: [0xfd,0x1a,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c11e1afd <unknown>

bfmls   za.h[w11, 2, vgx2], {z8.h, z9.h}, z1.h[2]  // 11000001-00010001-01110101-00110010
// CHECK-INST: bfmls   za.h[w11, 2, vgx2], { z8.h, z9.h }, z1.h[2]
// CHECK-ENCODING: [0x32,0x75,0x11,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1117532 <unknown>

bfmls   za.h[w11, 2], {z8.h - z9.h}, z1.h[2]  // 11000001-00010001-01110101-00110010
// CHECK-INST: bfmls   za.h[w11, 2, vgx2], { z8.h, z9.h }, z1.h[2]
// CHECK-ENCODING: [0x32,0x75,0x11,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1117532 <unknown>

bfmls   za.h[w9, 7, vgx2], {z12.h, z13.h}, z11.h[4]  // 11000001-00011011-00111001-10110111
// CHECK-INST: bfmls   za.h[w9, 7, vgx2], { z12.h, z13.h }, z11.h[4]
// CHECK-ENCODING: [0xb7,0x39,0x1b,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c11b39b7 <unknown>

bfmls   za.h[w9, 7], {z12.h - z13.h}, z11.h[4]  // 11000001-00011011-00111001-10110111
// CHECK-INST: bfmls   za.h[w9, 7, vgx2], { z12.h, z13.h }, z11.h[4]
// CHECK-ENCODING: [0xb7,0x39,0x1b,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c11b39b7 <unknown>

bfmls   za.h[w8, 0, vgx2], {z0.h, z1.h}, {z0.h, z1.h}  // 11000001, 11100000-00010000-00011000
// CHECK-INST: bfmls   za.h[w8, 0, vgx2], { z0.h, z1.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x18,0x10,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1e01018 <unknown>

bfmls   za.h[w8, 0], {z0.h - z1.h}, {z0.h - z1.h}  // 11000001-11100000-00010000-00011000
// CHECK-INST: bfmls   za.h[w8, 0, vgx2], { z0.h, z1.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x18,0x10,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1e01018 <unknown>

bfmls   za.h[w10, 5, vgx2], {z10.h, z11.h}, {z20.h, z21.h}  // 11000001, 11110100-01010001-01011101
// CHECK-INST: bfmls   za.h[w10, 5, vgx2], { z10.h, z11.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x5d,0x51,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1f4515d <unknown>

bfmls   za.h[w10, 5], {z10.h - z11.h}, {z20.h - z21.h}  // 11000001-11110100-01010001-01011101
// CHECK-INST: bfmls   za.h[w10, 5, vgx2], { z10.h, z11.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x5d,0x51,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1f4515d <unknown>

bfmls   za.h[w11, 7, vgx2], {z12.h, z13.h}, {z8.h, z9.h}  // 11000001, 11101000-01110001-10011111
// CHECK-INST: bfmls   za.h[w11, 7, vgx2], { z12.h, z13.h }, { z8.h, z9.h }
// CHECK-ENCODING: [0x9f,0x71,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1e8719f <unknown>

bfmls   za.h[w11, 7], {z12.h - z13.h}, {z8.h - z9.h}  // 11000001-11101000-01110001-10011111
// CHECK-INST: bfmls   za.h[w11, 7, vgx2], { z12.h, z13.h }, { z8.h, z9.h }
// CHECK-ENCODING: [0x9f,0x71,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1e8719f <unknown>

bfmls   za.h[w11, 7, vgx2], {z30.h, z31.h}, {z30.h, z31.h}  // 11000001, 11111110-01110011-11011111
// CHECK-INST: bfmls   za.h[w11, 7, vgx2], { z30.h, z31.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xdf,0x73,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1fe73df <unknown>

bfmls   za.h[w11, 7], {z30.h - z31.h}, {z30.h - z31.h}  // 11000001-11111110-01110011-11011111
// CHECK-INST: bfmls   za.h[w11, 7, vgx2], { z30.h, z31.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xdf,0x73,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1fe73df <unknown>

bfmls   za.h[w8, 5, vgx2], {z16.h, z17.h}, {z16.h, z17.h}  // 11000001, 11110000-00010010-00011101
// CHECK-INST: bfmls   za.h[w8, 5, vgx2], { z16.h, z17.h }, { z16.h, z17.h }
// CHECK-ENCODING: [0x1d,0x12,0xf0,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1f0121d <unknown>

bfmls   za.h[w8, 5], {z16.h - z17.h}, {z16.h - z17.h}  // 11000001-11110000-00010010-00011101
// CHECK-INST: bfmls   za.h[w8, 5, vgx2], { z16.h, z17.h }, { z16.h, z17.h }
// CHECK-ENCODING: [0x1d,0x12,0xf0,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1f0121d <unknown>

bfmls   za.h[w8, 1, vgx2], {z0.h, z1.h}, {z30.h, z31.h}  // 11000001, 11111110-00010000-00011001
// CHECK-INST: bfmls   za.h[w8, 1, vgx2], { z0.h, z1.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0x19,0x10,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1fe1019 <unknown>

bfmls   za.h[w8, 1], {z0.h - z1.h}, {z30.h - z31.h}  // 11000001-11111110-00010000-00011001
// CHECK-INST: bfmls   za.h[w8, 1, vgx2], { z0.h, z1.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0x19,0x10,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1fe1019 <unknown>

bfmls   za.h[w10, 0, vgx2], {z18.h, z19.h}, {z20.h, z21.h}  // 11000001, 11110100-01010010-01011000
// CHECK-INST: bfmls   za.h[w10, 0, vgx2], { z18.h, z19.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x58,0x52,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1f45258 <unknown>

bfmls   za.h[w10, 0], {z18.h - z19.h}, {z20.h - z21.h}  // 11000001-11110100-01010010-01011000
// CHECK-INST: bfmls   za.h[w10, 0, vgx2], { z18.h, z19.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x58,0x52,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1f45258 <unknown>

bfmls   za.h[w8, 0, vgx2], {z12.h, z13.h}, {z2.h, z3.h}  // 11000001, 11100010-00010001-10011000
// CHECK-INST: bfmls   za.h[w8, 0, vgx2], { z12.h, z13.h }, { z2.h, z3.h }
// CHECK-ENCODING: [0x98,0x11,0xe2,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1e21198 <unknown>

bfmls   za.h[w8, 0], {z12.h - z13.h}, {z2.h - z3.h}  // 11000001-11100010-00010001-10011000
// CHECK-INST: bfmls   za.h[w8, 0, vgx2], { z12.h, z13.h }, { z2.h, z3.h }
// CHECK-ENCODING: [0x98,0x11,0xe2,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1e21198 <unknown>

bfmls   za.h[w10, 1, vgx2], {z0.h, z1.h}, {z26.h, z27.h}  // 11000001, 11111010-01010000-00011001
// CHECK-INST: bfmls   za.h[w10, 1, vgx2], { z0.h, z1.h }, { z26.h, z27.h }
// CHECK-ENCODING: [0x19,0x50,0xfa,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1fa5019 <unknown>

bfmls   za.h[w10, 1], {z0.h - z1.h}, {z26.h - z27.h}  // 11000001-11111010-01010000-00011001
// CHECK-INST: bfmls   za.h[w10, 1, vgx2], { z0.h, z1.h }, { z26.h, z27.h }
// CHECK-ENCODING: [0x19,0x50,0xfa,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1fa5019 <unknown>

bfmls   za.h[w8, 5, vgx2], {z22.h, z23.h}, {z30.h, z31.h}  // 11000001, 11111110-00010010-11011101
// CHECK-INST: bfmls   za.h[w8, 5, vgx2], { z22.h, z23.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xdd,0x12,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1fe12dd <unknown>

bfmls   za.h[w8, 5], {z22.h - z23.h}, {z30.h - z31.h}  // 11000001-11111110-00010010-11011101
// CHECK-INST: bfmls   za.h[w8, 5, vgx2], { z22.h, z23.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xdd,0x12,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1fe12dd <unknown>

bfmls   za.h[w11, 2, vgx2], {z8.h, z9.h}, {z0.h, z1.h}  // 11000001, 11100000-01110001-00011010
// CHECK-INST: bfmls   za.h[w11, 2, vgx2], { z8.h, z9.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x1a,0x71,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1e0711a <unknown>

bfmls   za.h[w11, 2], {z8.h - z9.h}, {z0.h - z1.h}  // 11000001-11100000-01110001-00011010
// CHECK-INST: bfmls   za.h[w11, 2, vgx2], { z8.h, z9.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x1a,0x71,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1e0711a <unknown>

bfmls   za.h[w9, 7, vgx2], {z12.h, z13.h}, {z10.h, z11.h}  // 11000001, 11101010-00110001-10011111
// CHECK-INST: bfmls   za.h[w9, 7, vgx2], { z12.h, z13.h }, { z10.h, z11.h }
// CHECK-ENCODING: [0x9f,0x31,0xea,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1ea319f <unknown>

bfmls   za.h[w9, 7], {z12.h - z13.h}, {z10.h - z11.h}  // 11000001-11101010-00110001-10011111
// CHECK-INST: bfmls   za.h[w9, 7, vgx2], { z12.h, z13.h }, { z10.h, z11.h }
// CHECK-ENCODING: [0x9f,0x31,0xea,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1ea319f <unknown>

bfmls   za.h[w8, 0, vgx4], {z0.h - z3.h}, z0.h  // 11000001-01110000-00011100-00001000
// CHECK-INST: bfmls   za.h[w8, 0, vgx4], { z0.h - z3.h }, z0.h
// CHECK-ENCODING: [0x08,0x1c,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1701c08 <unknown>

bfmls   za.h[w8, 0], {z0.h - z3.h}, z0.h  // 11000001-01110000-00011100-00001000
// CHECK-INST: bfmls   za.h[w8, 0, vgx4], { z0.h - z3.h }, z0.h
// CHECK-ENCODING: [0x08,0x1c,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1701c08 <unknown>

bfmls   za.h[w10, 5, vgx4], {z10.h - z13.h}, z5.h  // 11000001-01110101-01011101-01001101
// CHECK-INST: bfmls   za.h[w10, 5, vgx4], { z10.h - z13.h }, z5.h
// CHECK-ENCODING: [0x4d,0x5d,0x75,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1755d4d <unknown>

bfmls   za.h[w10, 5], {z10.h - z13.h}, z5.h  // 11000001-01110101-01011101-01001101
// CHECK-INST: bfmls   za.h[w10, 5, vgx4], { z10.h - z13.h }, z5.h
// CHECK-ENCODING: [0x4d,0x5d,0x75,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1755d4d <unknown>

bfmls   za.h[w11, 7, vgx4], {z13.h - z16.h}, z8.h  // 11000001-01111000-01111101-10101111
// CHECK-INST: bfmls   za.h[w11, 7, vgx4], { z13.h - z16.h }, z8.h
// CHECK-ENCODING: [0xaf,0x7d,0x78,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1787daf <unknown>

bfmls   za.h[w11, 7], {z13.h - z16.h}, z8.h  // 11000001-01111000-01111101-10101111
// CHECK-INST: bfmls   za.h[w11, 7, vgx4], { z13.h - z16.h }, z8.h
// CHECK-ENCODING: [0xaf,0x7d,0x78,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1787daf <unknown>

bfmls   za.h[w11, 7, vgx4], {z31.h, z0.h, z1.h, z2.h}, z15.h  // 11000001-01111111-01111111-11101111
// CHECK-INST: bfmls   za.h[w11, 7, vgx4], { z31.h, z0.h, z1.h, z2.h }, z15.h
// CHECK-ENCODING: [0xef,0x7f,0x7f,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c17f7fef <unknown>

bfmls   za.h[w11, 7], {z31.h, z0.h, z1.h, z2.h}, z15.h  // 11000001-01111111-01111111-11101111
// CHECK-INST: bfmls   za.h[w11, 7, vgx4], { z31.h, z0.h, z1.h, z2.h }, z15.h
// CHECK-ENCODING: [0xef,0x7f,0x7f,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c17f7fef <unknown>

bfmls   za.h[w8, 5, vgx4], {z17.h - z20.h}, z0.h  // 11000001-01110000-00011110-00101101
// CHECK-INST: bfmls   za.h[w8, 5, vgx4], { z17.h - z20.h }, z0.h
// CHECK-ENCODING: [0x2d,0x1e,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1701e2d <unknown>

bfmls   za.h[w8, 5], {z17.h - z20.h}, z0.h  // 11000001-01110000-00011110-00101101
// CHECK-INST: bfmls   za.h[w8, 5, vgx4], { z17.h - z20.h }, z0.h
// CHECK-ENCODING: [0x2d,0x1e,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1701e2d <unknown>

bfmls   za.h[w8, 1, vgx4], {z1.h - z4.h}, z14.h  // 11000001-01111110-00011100-00101001
// CHECK-INST: bfmls   za.h[w8, 1, vgx4], { z1.h - z4.h }, z14.h
// CHECK-ENCODING: [0x29,0x1c,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c17e1c29 <unknown>

bfmls   za.h[w8, 1], {z1.h - z4.h}, z14.h  // 11000001-01111110-00011100-00101001
// CHECK-INST: bfmls   za.h[w8, 1, vgx4], { z1.h - z4.h }, z14.h
// CHECK-ENCODING: [0x29,0x1c,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c17e1c29 <unknown>

bfmls   za.h[w10, 0, vgx4], {z19.h - z22.h}, z4.h  // 11000001-01110100-01011110-01101000
// CHECK-INST: bfmls   za.h[w10, 0, vgx4], { z19.h - z22.h }, z4.h
// CHECK-ENCODING: [0x68,0x5e,0x74,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1745e68 <unknown>

bfmls   za.h[w10, 0], {z19.h - z22.h}, z4.h  // 11000001-01110100-01011110-01101000
// CHECK-INST: bfmls   za.h[w10, 0, vgx4], { z19.h - z22.h }, z4.h
// CHECK-ENCODING: [0x68,0x5e,0x74,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1745e68 <unknown>

bfmls   za.h[w8, 0, vgx4], {z12.h - z15.h}, z2.h  // 11000001-01110010-00011101-10001000
// CHECK-INST: bfmls   za.h[w8, 0, vgx4], { z12.h - z15.h }, z2.h
// CHECK-ENCODING: [0x88,0x1d,0x72,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1721d88 <unknown>

bfmls   za.h[w8, 0], {z12.h - z15.h}, z2.h  // 11000001-01110010-00011101-10001000
// CHECK-INST: bfmls   za.h[w8, 0, vgx4], { z12.h - z15.h }, z2.h
// CHECK-ENCODING: [0x88,0x1d,0x72,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1721d88 <unknown>

bfmls   za.h[w10, 1, vgx4], {z1.h - z4.h}, z10.h  // 11000001-01111010-01011100-00101001
// CHECK-INST: bfmls   za.h[w10, 1, vgx4], { z1.h - z4.h }, z10.h
// CHECK-ENCODING: [0x29,0x5c,0x7a,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c17a5c29 <unknown>

bfmls   za.h[w10, 1], {z1.h - z4.h}, z10.h  // 11000001-01111010-01011100-00101001
// CHECK-INST: bfmls   za.h[w10, 1, vgx4], { z1.h - z4.h }, z10.h
// CHECK-ENCODING: [0x29,0x5c,0x7a,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c17a5c29 <unknown>

bfmls   za.h[w8, 5, vgx4], {z22.h - z25.h}, z14.h  // 11000001-01111110-00011110-11001101
// CHECK-INST: bfmls   za.h[w8, 5, vgx4], { z22.h - z25.h }, z14.h
// CHECK-ENCODING: [0xcd,0x1e,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c17e1ecd <unknown>

bfmls   za.h[w8, 5], {z22.h - z25.h}, z14.h  // 11000001-01111110-00011110-11001101
// CHECK-INST: bfmls   za.h[w8, 5, vgx4], { z22.h - z25.h }, z14.h
// CHECK-ENCODING: [0xcd,0x1e,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c17e1ecd <unknown>

bfmls   za.h[w11, 2, vgx4], {z9.h - z12.h}, z1.h  // 11000001-01110001-01111101-00101010
// CHECK-INST: bfmls   za.h[w11, 2, vgx4], { z9.h - z12.h }, z1.h
// CHECK-ENCODING: [0x2a,0x7d,0x71,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1717d2a <unknown>

bfmls   za.h[w11, 2], {z9.h - z12.h}, z1.h  // 11000001-01110001-01111101-00101010
// CHECK-INST: bfmls   za.h[w11, 2, vgx4], { z9.h - z12.h }, z1.h
// CHECK-ENCODING: [0x2a,0x7d,0x71,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1717d2a <unknown>

bfmls   za.h[w9, 7, vgx4], {z12.h - z15.h}, z11.h  // 11000001-01111011-00111101-10001111
// CHECK-INST: bfmls   za.h[w9, 7, vgx4], { z12.h - z15.h }, z11.h
// CHECK-ENCODING: [0x8f,0x3d,0x7b,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c17b3d8f <unknown>

bfmls   za.h[w9, 7], {z12.h - z15.h}, z11.h  // 11000001-01111011-00111101-10001111
// CHECK-INST: bfmls   za.h[w9, 7, vgx4], { z12.h - z15.h }, z11.h
// CHECK-ENCODING: [0x8f,0x3d,0x7b,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c17b3d8f <unknown>

bfmls   za.h[w8, 0, vgx4], {z0.h - z3.h}, z0.h[0]  // 11000001-00010000-10010000-00110000
// CHECK-INST: bfmls   za.h[w8, 0, vgx4], { z0.h - z3.h }, z0.h[0]
// CHECK-ENCODING: [0x30,0x90,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1109030 <unknown>

bfmls   za.h[w8, 0], {z0.h - z3.h}, z0.h[0]  // 11000001-00010000-10010000-00110000
// CHECK-INST: bfmls   za.h[w8, 0, vgx4], { z0.h - z3.h }, z0.h[0]
// CHECK-ENCODING: [0x30,0x90,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1109030 <unknown>

bfmls   za.h[w10, 5, vgx4], {z8.h - z11.h}, z5.h[2]  // 11000001-00010101-11010101-00110101
// CHECK-INST: bfmls   za.h[w10, 5, vgx4], { z8.h - z11.h }, z5.h[2]
// CHECK-ENCODING: [0x35,0xd5,0x15,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c115d535 <unknown>

bfmls   za.h[w10, 5], {z8.h - z11.h}, z5.h[2]  // 11000001-00010101-11010101-00110101
// CHECK-INST: bfmls   za.h[w10, 5, vgx4], { z8.h - z11.h }, z5.h[2]
// CHECK-ENCODING: [0x35,0xd5,0x15,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c115d535 <unknown>

bfmls   za.h[w11, 7, vgx4], {z12.h - z15.h}, z8.h[6]  // 11000001-00011000-11111101-10110111
// CHECK-INST: bfmls   za.h[w11, 7, vgx4], { z12.h - z15.h }, z8.h[6]
// CHECK-ENCODING: [0xb7,0xfd,0x18,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c118fdb7 <unknown>

bfmls   za.h[w11, 7], {z12.h - z15.h}, z8.h[6]  // 11000001-00011000-11111101-10110111
// CHECK-INST: bfmls   za.h[w11, 7, vgx4], { z12.h - z15.h }, z8.h[6]
// CHECK-ENCODING: [0xb7,0xfd,0x18,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c118fdb7 <unknown>

bfmls   za.h[w11, 7, vgx4], {z28.h - z31.h}, z15.h[7]  // 11000001-00011111-11111111-10111111
// CHECK-INST: bfmls   za.h[w11, 7, vgx4], { z28.h - z31.h }, z15.h[7]
// CHECK-ENCODING: [0xbf,0xff,0x1f,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c11fffbf <unknown>

bfmls   za.h[w11, 7], {z28.h - z31.h}, z15.h[7]  // 11000001-00011111-11111111-10111111
// CHECK-INST: bfmls   za.h[w11, 7, vgx4], { z28.h - z31.h }, z15.h[7]
// CHECK-ENCODING: [0xbf,0xff,0x1f,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c11fffbf <unknown>

bfmls   za.h[w8, 5, vgx4], {z16.h - z19.h}, z0.h[6]  // 11000001-00010000-10011110-00110101
// CHECK-INST: bfmls   za.h[w8, 5, vgx4], { z16.h - z19.h }, z0.h[6]
// CHECK-ENCODING: [0x35,0x9e,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1109e35 <unknown>

bfmls   za.h[w8, 5], {z16.h - z19.h}, z0.h[6]  // 11000001-00010000-10011110-00110101
// CHECK-INST: bfmls   za.h[w8, 5, vgx4], { z16.h - z19.h }, z0.h[6]
// CHECK-ENCODING: [0x35,0x9e,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1109e35 <unknown>

bfmls   za.h[w8, 1, vgx4], {z0.h - z3.h}, z14.h[2]  // 11000001-00011110-10010100-00110001
// CHECK-INST: bfmls   za.h[w8, 1, vgx4], { z0.h - z3.h }, z14.h[2]
// CHECK-ENCODING: [0x31,0x94,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c11e9431 <unknown>

bfmls   za.h[w8, 1], {z0.h - z3.h}, z14.h[2]  // 11000001-00011110-10010100-00110001
// CHECK-INST: bfmls   za.h[w8, 1, vgx4], { z0.h - z3.h }, z14.h[2]
// CHECK-ENCODING: [0x31,0x94,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c11e9431 <unknown>

bfmls   za.h[w10, 0, vgx4], {z16.h - z19.h}, z4.h[3]  // 11000001-00010100-11010110-00111000
// CHECK-INST: bfmls   za.h[w10, 0, vgx4], { z16.h - z19.h }, z4.h[3]
// CHECK-ENCODING: [0x38,0xd6,0x14,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c114d638 <unknown>

bfmls   za.h[w10, 0], {z16.h - z19.h}, z4.h[3]  // 11000001-00010100-11010110-00111000
// CHECK-INST: bfmls   za.h[w10, 0, vgx4], { z16.h - z19.h }, z4.h[3]
// CHECK-ENCODING: [0x38,0xd6,0x14,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c114d638 <unknown>

bfmls   za.h[w8, 0, vgx4], {z12.h - z15.h}, z2.h[4]  // 11000001-00010010-10011001-10110000
// CHECK-INST: bfmls   za.h[w8, 0, vgx4], { z12.h - z15.h }, z2.h[4]
// CHECK-ENCODING: [0xb0,0x99,0x12,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c11299b0 <unknown>

bfmls   za.h[w8, 0], {z12.h - z15.h}, z2.h[4]  // 11000001-00010010-10011001-10110000
// CHECK-INST: bfmls   za.h[w8, 0, vgx4], { z12.h - z15.h }, z2.h[4]
// CHECK-ENCODING: [0xb0,0x99,0x12,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c11299b0 <unknown>

bfmls   za.h[w10, 1, vgx4], {z0.h - z3.h}, z10.h[4]  // 11000001-00011010-11011000-00110001
// CHECK-INST: bfmls   za.h[w10, 1, vgx4], { z0.h - z3.h }, z10.h[4]
// CHECK-ENCODING: [0x31,0xd8,0x1a,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c11ad831 <unknown>

bfmls   za.h[w10, 1], {z0.h - z3.h}, z10.h[4]  // 11000001-00011010-11011000-00110001
// CHECK-INST: bfmls   za.h[w10, 1, vgx4], { z0.h - z3.h }, z10.h[4]
// CHECK-ENCODING: [0x31,0xd8,0x1a,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c11ad831 <unknown>

bfmls   za.h[w8, 5, vgx4], {z20.h - z23.h}, z14.h[5]  // 11000001-00011110-10011010-10111101
// CHECK-INST: bfmls   za.h[w8, 5, vgx4], { z20.h - z23.h }, z14.h[5]
// CHECK-ENCODING: [0xbd,0x9a,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c11e9abd <unknown>

bfmls   za.h[w8, 5], {z20.h - z23.h}, z14.h[5]  // 11000001-00011110-10011010-10111101
// CHECK-INST: bfmls   za.h[w8, 5, vgx4], { z20.h - z23.h }, z14.h[5]
// CHECK-ENCODING: [0xbd,0x9a,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c11e9abd <unknown>

bfmls   za.h[w11, 2, vgx4], {z8.h - z11.h}, z1.h[2]  // 11000001-00010001-11110101-00110010
// CHECK-INST: bfmls   za.h[w11, 2, vgx4], { z8.h - z11.h }, z1.h[2]
// CHECK-ENCODING: [0x32,0xf5,0x11,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c111f532 <unknown>

bfmls   za.h[w11, 2], {z8.h - z11.h}, z1.h[2]  // 11000001-00010001-11110101-00110010
// CHECK-INST: bfmls   za.h[w11, 2, vgx4], { z8.h - z11.h }, z1.h[2]
// CHECK-ENCODING: [0x32,0xf5,0x11,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c111f532 <unknown>

bfmls   za.h[w9, 7, vgx4], {z12.h - z15.h}, z11.h[4]  // 11000001-00011011-10111001-10110111
// CHECK-INST: bfmls   za.h[w9, 7, vgx4], { z12.h - z15.h }, z11.h[4]
// CHECK-ENCODING: [0xb7,0xb9,0x1b,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c11bb9b7 <unknown>

bfmls   za.h[w9, 7], {z12.h - z15.h}, z11.h[4]  // 11000001-00011011-10111001-10110111
// CHECK-INST: bfmls   za.h[w9, 7, vgx4], { z12.h - z15.h }, z11.h[4]
// CHECK-ENCODING: [0xb7,0xb9,0x1b,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c11bb9b7 <unknown>

bfmls   za.h[w8, 0, vgx4], {z0.h - z3.h}, {z0.h - z3.h}  // 11000001-11100001-00010000-00011000
// CHECK-INST: bfmls   za.h[w8, 0, vgx4], { z0.h - z3.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x18,0x10,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1e11018 <unknown>

bfmls   za.h[w8, 0], {z0.h - z3.h}, {z0.h - z3.h}  // 11000001-11100001-00010000-00011000
// CHECK-INST: bfmls   za.h[w8, 0, vgx4], { z0.h - z3.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x18,0x10,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1e11018 <unknown>

bfmls   za.h[w10, 5, vgx4], {z8.h - z11.h}, {z20.h - z23.h}  // 11000001-11110101-01010001-00011101
// CHECK-INST: bfmls   za.h[w10, 5, vgx4], { z8.h - z11.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x1d,0x51,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1f5511d <unknown>

bfmls   za.h[w10, 5], {z8.h - z11.h}, {z20.h - z23.h}  // 11000001-11110101-01010001-00011101
// CHECK-INST: bfmls   za.h[w10, 5, vgx4], { z8.h - z11.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x1d,0x51,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1f5511d <unknown>

bfmls   za.h[w11, 7, vgx4], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-11101001-01110001-10011111
// CHECK-INST: bfmls   za.h[w11, 7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x9f,0x71,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1e9719f <unknown>

bfmls   za.h[w11, 7], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-11101001-01110001-10011111
// CHECK-INST: bfmls   za.h[w11, 7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x9f,0x71,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1e9719f <unknown>

bfmls   za.h[w11, 7, vgx4], {z28.h - z31.h}, {z28.h - z31.h}  // 11000001-11111101-01110011-10011111
// CHECK-INST: bfmls   za.h[w11, 7, vgx4], { z28.h - z31.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x9f,0x73,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1fd739f <unknown>

bfmls   za.h[w11, 7], {z28.h - z31.h}, {z28.h - z31.h}  // 11000001-11111101-01110011-10011111
// CHECK-INST: bfmls   za.h[w11, 7, vgx4], { z28.h - z31.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x9f,0x73,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1fd739f <unknown>

bfmls   za.h[w8, 5, vgx4], {z16.h - z19.h}, {z16.h - z19.h}  // 11000001-11110001-00010010-00011101
// CHECK-INST: bfmls   za.h[w8, 5, vgx4], { z16.h - z19.h }, { z16.h - z19.h }
// CHECK-ENCODING: [0x1d,0x12,0xf1,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1f1121d <unknown>

bfmls   za.h[w8, 5], {z16.h - z19.h}, {z16.h - z19.h}  // 11000001-11110001-00010010-00011101
// CHECK-INST: bfmls   za.h[w8, 5, vgx4], { z16.h - z19.h }, { z16.h - z19.h }
// CHECK-ENCODING: [0x1d,0x12,0xf1,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1f1121d <unknown>

bfmls   za.h[w8, 1, vgx4], {z0.h - z3.h}, {z28.h - z31.h}  // 11000001-11111101-00010000-00011001
// CHECK-INST: bfmls   za.h[w8, 1, vgx4], { z0.h - z3.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x19,0x10,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1fd1019 <unknown>

bfmls   za.h[w8, 1], {z0.h - z3.h}, {z28.h - z31.h}  // 11000001-11111101-00010000-00011001
// CHECK-INST: bfmls   za.h[w8, 1, vgx4], { z0.h - z3.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x19,0x10,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1fd1019 <unknown>

bfmls   za.h[w10, 0, vgx4], {z16.h - z19.h}, {z20.h - z23.h}  // 11000001-11110101-01010010-00011000
// CHECK-INST: bfmls   za.h[w10, 0, vgx4], { z16.h - z19.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x18,0x52,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1f55218 <unknown>

bfmls   za.h[w10, 0], {z16.h - z19.h}, {z20.h - z23.h}  // 11000001-11110101-01010010-00011000
// CHECK-INST: bfmls   za.h[w10, 0, vgx4], { z16.h - z19.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x18,0x52,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1f55218 <unknown>

bfmls   za.h[w8, 0, vgx4], {z12.h - z15.h}, {z0.h - z3.h}  // 11000001-11100001-00010001-10011000
// CHECK-INST: bfmls   za.h[w8, 0, vgx4], { z12.h - z15.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x98,0x11,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1e11198 <unknown>

bfmls   za.h[w8, 0], {z12.h - z15.h}, {z0.h - z3.h}  // 11000001-11100001-00010001-10011000
// CHECK-INST: bfmls   za.h[w8, 0, vgx4], { z12.h - z15.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x98,0x11,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1e11198 <unknown>

bfmls   za.h[w10, 1, vgx4], {z0.h - z3.h}, {z24.h - z27.h}  // 11000001-11111001-01010000-00011001
// CHECK-INST: bfmls   za.h[w10, 1, vgx4], { z0.h - z3.h }, { z24.h - z27.h }
// CHECK-ENCODING: [0x19,0x50,0xf9,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1f95019 <unknown>

bfmls   za.h[w10, 1], {z0.h - z3.h}, {z24.h - z27.h}  // 11000001-11111001-01010000-00011001
// CHECK-INST: bfmls   za.h[w10, 1, vgx4], { z0.h - z3.h }, { z24.h - z27.h }
// CHECK-ENCODING: [0x19,0x50,0xf9,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1f95019 <unknown>

bfmls   za.h[w8, 5, vgx4], {z20.h - z23.h}, {z28.h - z31.h}  // 11000001-11111101-00010010-10011101
// CHECK-INST: bfmls   za.h[w8, 5, vgx4], { z20.h - z23.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x9d,0x12,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1fd129d <unknown>

bfmls   za.h[w8, 5], {z20.h - z23.h}, {z28.h - z31.h}  // 11000001-11111101-00010010-10011101
// CHECK-INST: bfmls   za.h[w8, 5, vgx4], { z20.h - z23.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x9d,0x12,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1fd129d <unknown>

bfmls   za.h[w11, 2, vgx4], {z8.h - z11.h}, {z0.h - z3.h}  // 11000001-11100001-01110001-00011010
// CHECK-INST: bfmls   za.h[w11, 2, vgx4], { z8.h - z11.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x1a,0x71,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1e1711a <unknown>

bfmls   za.h[w11, 2], {z8.h - z11.h}, {z0.h - z3.h}  // 11000001-11100001-01110001-00011010
// CHECK-INST: bfmls   za.h[w11, 2, vgx4], { z8.h - z11.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x1a,0x71,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1e1711a <unknown>

bfmls   za.h[w9, 7, vgx4], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-11101001-00110001-10011111
// CHECK-INST: bfmls   za.h[w9, 7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x9f,0x31,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1e9319f <unknown>

bfmls   za.h[w9, 7], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-11101001-00110001-10011111
// CHECK-INST: bfmls   za.h[w9, 7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x9f,0x31,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme-b16b16
// CHECK-UNKNOWN: c1e9319f <unknown>
