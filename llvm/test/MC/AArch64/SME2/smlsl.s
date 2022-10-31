// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump --no-print-imm-hex -d --mattr=+sme2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump --no-print-imm-hex -d --mattr=-sme2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


smlsl   za.s[w8, 0:1], z0.h, z0.h  // 11000001-01100000-00001100-00001000
// CHECK-INST: smlsl   za.s[w8, 0:1], z0.h, z0.h
// CHECK-ENCODING: [0x08,0x0c,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1600c08 <unknown>

smlsl   za.s[w10, 10:11], z10.h, z5.h  // 11000001-01100101-01001101-01001101
// CHECK-INST: smlsl   za.s[w10, 10:11], z10.h, z5.h
// CHECK-ENCODING: [0x4d,0x4d,0x65,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1654d4d <unknown>

smlsl   za.s[w11, 14:15], z13.h, z8.h  // 11000001-01101000-01101101-10101111
// CHECK-INST: smlsl   za.s[w11, 14:15], z13.h, z8.h
// CHECK-ENCODING: [0xaf,0x6d,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1686daf <unknown>

smlsl   za.s[w11, 14:15], z31.h, z15.h  // 11000001-01101111-01101111-11101111
// CHECK-INST: smlsl   za.s[w11, 14:15], z31.h, z15.h
// CHECK-ENCODING: [0xef,0x6f,0x6f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16f6fef <unknown>

smlsl   za.s[w8, 10:11], z17.h, z0.h  // 11000001-01100000-00001110-00101101
// CHECK-INST: smlsl   za.s[w8, 10:11], z17.h, z0.h
// CHECK-ENCODING: [0x2d,0x0e,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1600e2d <unknown>

smlsl   za.s[w8, 2:3], z1.h, z14.h  // 11000001-01101110-00001100-00101001
// CHECK-INST: smlsl   za.s[w8, 2:3], z1.h, z14.h
// CHECK-ENCODING: [0x29,0x0c,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e0c29 <unknown>

smlsl   za.s[w10, 0:1], z19.h, z4.h  // 11000001-01100100-01001110-01101000
// CHECK-INST: smlsl   za.s[w10, 0:1], z19.h, z4.h
// CHECK-ENCODING: [0x68,0x4e,0x64,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1644e68 <unknown>

smlsl   za.s[w8, 0:1], z12.h, z2.h  // 11000001-01100010-00001101-10001000
// CHECK-INST: smlsl   za.s[w8, 0:1], z12.h, z2.h
// CHECK-ENCODING: [0x88,0x0d,0x62,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1620d88 <unknown>

smlsl   za.s[w10, 2:3], z1.h, z10.h  // 11000001-01101010-01001100-00101001
// CHECK-INST: smlsl   za.s[w10, 2:3], z1.h, z10.h
// CHECK-ENCODING: [0x29,0x4c,0x6a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16a4c29 <unknown>

smlsl   za.s[w8, 10:11], z22.h, z14.h  // 11000001-01101110-00001110-11001101
// CHECK-INST: smlsl   za.s[w8, 10:11], z22.h, z14.h
// CHECK-ENCODING: [0xcd,0x0e,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e0ecd <unknown>

smlsl   za.s[w11, 4:5], z9.h, z1.h  // 11000001-01100001-01101101-00101010
// CHECK-INST: smlsl   za.s[w11, 4:5], z9.h, z1.h
// CHECK-ENCODING: [0x2a,0x6d,0x61,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1616d2a <unknown>

smlsl   za.s[w9, 14:15], z12.h, z11.h  // 11000001-01101011-00101101-10001111
// CHECK-INST: smlsl   za.s[w9, 14:15], z12.h, z11.h
// CHECK-ENCODING: [0x8f,0x2d,0x6b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16b2d8f <unknown>


smlsl   za.s[w8, 0:1], z0.h, z0.h[0]  // 11000001-11000000-00010000-00001000
// CHECK-INST: smlsl   za.s[w8, 0:1], z0.h, z0.h[0]
// CHECK-ENCODING: [0x08,0x10,0xc0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1c01008 <unknown>

smlsl   za.s[w10, 10:11], z10.h, z5.h[1]  // 11000001-11000101-01010101-01001101
// CHECK-INST: smlsl   za.s[w10, 10:11], z10.h, z5.h[1]
// CHECK-ENCODING: [0x4d,0x55,0xc5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1c5554d <unknown>

smlsl   za.s[w11, 14:15], z13.h, z8.h[7]  // 11000001-11001000-11111101-10101111
// CHECK-INST: smlsl   za.s[w11, 14:15], z13.h, z8.h[7]
// CHECK-ENCODING: [0xaf,0xfd,0xc8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1c8fdaf <unknown>

smlsl   za.s[w11, 14:15], z31.h, z15.h[7]  // 11000001-11001111-11111111-11101111
// CHECK-INST: smlsl   za.s[w11, 14:15], z31.h, z15.h[7]
// CHECK-ENCODING: [0xef,0xff,0xcf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1cfffef <unknown>

smlsl   za.s[w8, 10:11], z17.h, z0.h[3]  // 11000001-11000000-00011110-00101101
// CHECK-INST: smlsl   za.s[w8, 10:11], z17.h, z0.h[3]
// CHECK-ENCODING: [0x2d,0x1e,0xc0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1c01e2d <unknown>

smlsl   za.s[w8, 2:3], z1.h, z14.h[5]  // 11000001-11001110-10010100-00101001
// CHECK-INST: smlsl   za.s[w8, 2:3], z1.h, z14.h[5]
// CHECK-ENCODING: [0x29,0x94,0xce,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ce9429 <unknown>

smlsl   za.s[w10, 0:1], z19.h, z4.h[1]  // 11000001-11000100-01010110-01101000
// CHECK-INST: smlsl   za.s[w10, 0:1], z19.h, z4.h[1]
// CHECK-ENCODING: [0x68,0x56,0xc4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1c45668 <unknown>

smlsl   za.s[w8, 0:1], z12.h, z2.h[2]  // 11000001-11000010-00011001-10001000
// CHECK-INST: smlsl   za.s[w8, 0:1], z12.h, z2.h[2]
// CHECK-ENCODING: [0x88,0x19,0xc2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1c21988 <unknown>

smlsl   za.s[w10, 2:3], z1.h, z10.h[6]  // 11000001-11001010-11011000-00101001
// CHECK-INST: smlsl   za.s[w10, 2:3], z1.h, z10.h[6]
// CHECK-ENCODING: [0x29,0xd8,0xca,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1cad829 <unknown>

smlsl   za.s[w8, 10:11], z22.h, z14.h[2]  // 11000001-11001110-00011010-11001101
// CHECK-INST: smlsl   za.s[w8, 10:11], z22.h, z14.h[2]
// CHECK-ENCODING: [0xcd,0x1a,0xce,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ce1acd <unknown>

smlsl   za.s[w11, 4:5], z9.h, z1.h[5]  // 11000001-11000001-11110101-00101010
// CHECK-INST: smlsl   za.s[w11, 4:5], z9.h, z1.h[5]
// CHECK-ENCODING: [0x2a,0xf5,0xc1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1c1f52a <unknown>

smlsl   za.s[w9, 14:15], z12.h, z11.h[6]  // 11000001-11001011-10111001-10001111
// CHECK-INST: smlsl   za.s[w9, 14:15], z12.h, z11.h[6]
// CHECK-ENCODING: [0x8f,0xb9,0xcb,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1cbb98f <unknown>


smlsl   za.s[w8, 0:1, vgx2], {z0.h, z1.h}, z0.h  // 11000001, 01100000, 00001000, 00001000
// CHECK, INST: smlsl   za.s[w8, 0:1, vgx2], { z0.h, z1.h }, z0.h
// CHECK-ENCODING: [0x08,0x08,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1600808 <unknown>

smlsl   za.s[w8, 0:1], {z0.h - z1.h}, z0.h  // 11000001-01100000-00001000-00001000
// CHECK, INST: smlsl   za.s[w8, 0:1, vgx2], { z0.h, z1.h }, z0.h
// CHECK-ENCODING: [0x08,0x08,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1600808 <unknown>

smlsl   za.s[w10, 2:3, vgx2], {z10.h, z11.h}, z5.h  // 11000001, 01100101, 01001001, 01001001
// CHECK, INST: smlsl   za.s[w10, 2:3, vgx2], { z10.h, z11.h }, z5.h
// CHECK-ENCODING: [0x49,0x49,0x65,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1654949 <unknown>

smlsl   za.s[w10, 2:3], {z10.h - z11.h}, z5.h  // 11000001-01100101-01001001-01001001
// CHECK, INST: smlsl   za.s[w10, 2:3, vgx2], { z10.h, z11.h }, z5.h
// CHECK-ENCODING: [0x49,0x49,0x65,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1654949 <unknown>

smlsl   za.s[w11, 6:7, vgx2], {z13.h, z14.h}, z8.h  // 11000001, 01101000, 01101001, 10101011
// CHECK, INST: smlsl   za.s[w11, 6:7, vgx2], { z13.h, z14.h }, z8.h
// CHECK-ENCODING: [0xab,0x69,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16869ab <unknown>

smlsl   za.s[w11, 6:7], {z13.h - z14.h}, z8.h  // 11000001-01101000-01101001-10101011
// CHECK, INST: smlsl   za.s[w11, 6:7, vgx2], { z13.h, z14.h }, z8.h
// CHECK-ENCODING: [0xab,0x69,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16869ab <unknown>

smlsl   za.s[w11, 6:7, vgx2], {z31.h, z0.h}, z15.h  // 11000001, 01101111, 01101011, 11101011
// CHECK, INST: smlsl   za.s[w11, 6:7, vgx2], { z31.h, z0.h }, z15.h
// CHECK-ENCODING: [0xeb,0x6b,0x6f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16f6beb <unknown>

smlsl   za.s[w11, 6:7], {z31.h - z0.h}, z15.h  // 11000001-01101111-01101011-11101011
// CHECK, INST: smlsl   za.s[w11, 6:7, vgx2], { z31.h, z0.h }, z15.h
// CHECK-ENCODING: [0xeb,0x6b,0x6f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16f6beb <unknown>

smlsl   za.s[w8, 2:3, vgx2], {z17.h, z18.h}, z0.h  // 11000001, 01100000, 00001010, 00101001
// CHECK, INST: smlsl   za.s[w8, 2:3, vgx2], { z17.h, z18.h }, z0.h
// CHECK-ENCODING: [0x29,0x0a,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1600a29 <unknown>

smlsl   za.s[w8, 2:3], {z17.h - z18.h}, z0.h  // 11000001-01100000-00001010-00101001
// CHECK, INST: smlsl   za.s[w8, 2:3, vgx2], { z17.h, z18.h }, z0.h
// CHECK-ENCODING: [0x29,0x0a,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1600a29 <unknown>

smlsl   za.s[w8, 2:3, vgx2], {z1.h, z2.h}, z14.h  // 11000001, 01101110, 00001000, 00101001
// CHECK, INST: smlsl   za.s[w8, 2:3, vgx2], { z1.h, z2.h }, z14.h
// CHECK-ENCODING: [0x29,0x08,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e0829 <unknown>

smlsl   za.s[w8, 2:3], {z1.h - z2.h}, z14.h  // 11000001-01101110-00001000-00101001
// CHECK, INST: smlsl   za.s[w8, 2:3, vgx2], { z1.h, z2.h }, z14.h
// CHECK-ENCODING: [0x29,0x08,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e0829 <unknown>

smlsl   za.s[w10, 0:1, vgx2], {z19.h, z20.h}, z4.h  // 11000001, 01100100, 01001010, 01101000
// CHECK, INST: smlsl   za.s[w10, 0:1, vgx2], { z19.h, z20.h }, z4.h
// CHECK-ENCODING: [0x68,0x4a,0x64,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1644a68 <unknown>

smlsl   za.s[w10, 0:1], {z19.h - z20.h}, z4.h  // 11000001-01100100-01001010-01101000
// CHECK, INST: smlsl   za.s[w10, 0:1, vgx2], { z19.h, z20.h }, z4.h
// CHECK-ENCODING: [0x68,0x4a,0x64,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1644a68 <unknown>

smlsl   za.s[w8, 0:1, vgx2], {z12.h, z13.h}, z2.h  // 11000001, 01100010, 00001001, 10001000
// CHECK, INST: smlsl   za.s[w8, 0:1, vgx2], { z12.h, z13.h }, z2.h
// CHECK-ENCODING: [0x88,0x09,0x62,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1620988 <unknown>

smlsl   za.s[w8, 0:1], {z12.h - z13.h}, z2.h  // 11000001-01100010-00001001-10001000
// CHECK, INST: smlsl   za.s[w8, 0:1, vgx2], { z12.h, z13.h }, z2.h
// CHECK-ENCODING: [0x88,0x09,0x62,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1620988 <unknown>

smlsl   za.s[w10, 2:3, vgx2], {z1.h, z2.h}, z10.h  // 11000001, 01101010, 01001000, 00101001
// CHECK, INST: smlsl   za.s[w10, 2:3, vgx2], { z1.h, z2.h }, z10.h
// CHECK-ENCODING: [0x29,0x48,0x6a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16a4829 <unknown>

smlsl   za.s[w10, 2:3], {z1.h - z2.h}, z10.h  // 11000001-01101010-01001000-00101001
// CHECK, INST: smlsl   za.s[w10, 2:3, vgx2], { z1.h, z2.h }, z10.h
// CHECK-ENCODING: [0x29,0x48,0x6a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16a4829 <unknown>

smlsl   za.s[w8, 2:3, vgx2], {z22.h, z23.h}, z14.h  // 11000001, 01101110, 00001010, 11001001
// CHECK, INST: smlsl   za.s[w8, 2:3, vgx2], { z22.h, z23.h }, z14.h
// CHECK-ENCODING: [0xc9,0x0a,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e0ac9 <unknown>

smlsl   za.s[w8, 2:3], {z22.h - z23.h}, z14.h  // 11000001-01101110-00001010-11001001
// CHECK, INST: smlsl   za.s[w8, 2:3, vgx2], { z22.h, z23.h }, z14.h
// CHECK-ENCODING: [0xc9,0x0a,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e0ac9 <unknown>

smlsl   za.s[w11, 4:5, vgx2], {z9.h, z10.h}, z1.h  // 11000001, 01100001, 01101001, 00101010
// CHECK, INST: smlsl   za.s[w11, 4:5, vgx2], { z9.h, z10.h }, z1.h
// CHECK-ENCODING: [0x2a,0x69,0x61,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c161692a <unknown>

smlsl   za.s[w11, 4:5], {z9.h - z10.h}, z1.h  // 11000001-01100001-01101001-00101010
// CHECK, INST: smlsl   za.s[w11, 4:5, vgx2], { z9.h, z10.h }, z1.h
// CHECK-ENCODING: [0x2a,0x69,0x61,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c161692a <unknown>

smlsl   za.s[w9, 6:7, vgx2], {z12.h, z13.h}, z11.h  // 11000001, 01101011, 00101001, 10001011
// CHECK, INST: smlsl   za.s[w9, 6:7, vgx2], { z12.h, z13.h }, z11.h
// CHECK-ENCODING: [0x8b,0x29,0x6b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16b298b <unknown>

smlsl   za.s[w9, 6:7], {z12.h - z13.h}, z11.h  // 11000001-01101011-00101001-10001011
// CHECK, INST: smlsl   za.s[w9, 6:7, vgx2], { z12.h, z13.h }, z11.h
// CHECK-ENCODING: [0x8b,0x29,0x6b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16b298b <unknown>


smlsl   za.s[w8, 0:1, vgx2], {z0.h, z1.h}, z0.h[0]  // 11000001, 11010000, 00010000, 00001000
// CHECK, INST: smlsl   za.s[w8, 0:1, vgx2], { z0.h, z1.h }, z0.h[0]
// CHECK-ENCODING: [0x08,0x10,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d01008 <unknown>

smlsl   za.s[w8, 0:1], {z0.h - z1.h}, z0.h[0]  // 11000001-11010000-00010000-00001000
// CHECK, INST: smlsl   za.s[w8, 0:1, vgx2], { z0.h, z1.h }, z0.h[0]
// CHECK-ENCODING: [0x08,0x10,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d01008 <unknown>

smlsl   za.s[w10, 2:3, vgx2], {z10.h, z11.h}, z5.h[3]  // 11000001, 11010101, 01010101, 01001101
// CHECK, INST: smlsl   za.s[w10, 2:3, vgx2], { z10.h, z11.h }, z5.h[3]
// CHECK-ENCODING: [0x4d,0x55,0xd5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d5554d <unknown>

smlsl   za.s[w10, 2:3], {z10.h - z11.h}, z5.h[3]  // 11000001-11010101-01010101-01001101
// CHECK, INST: smlsl   za.s[w10, 2:3, vgx2], { z10.h, z11.h }, z5.h[3]
// CHECK-ENCODING: [0x4d,0x55,0xd5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d5554d <unknown>

smlsl   za.s[w11, 6:7, vgx2], {z12.h, z13.h}, z8.h[7]  // 11000001, 11011000, 01111101, 10001111
// CHECK, INST: smlsl   za.s[w11, 6:7, vgx2], { z12.h, z13.h }, z8.h[7]
// CHECK-ENCODING: [0x8f,0x7d,0xd8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d87d8f <unknown>

smlsl   za.s[w11, 6:7], {z12.h - z13.h}, z8.h[7]  // 11000001-11011000-01111101-10001111
// CHECK, INST: smlsl   za.s[w11, 6:7, vgx2], { z12.h, z13.h }, z8.h[7]
// CHECK-ENCODING: [0x8f,0x7d,0xd8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d87d8f <unknown>

smlsl   za.s[w11, 6:7, vgx2], {z30.h, z31.h}, z15.h[7]  // 11000001, 11011111, 01111111, 11001111
// CHECK, INST: smlsl   za.s[w11, 6:7, vgx2], { z30.h, z31.h }, z15.h[7]
// CHECK-ENCODING: [0xcf,0x7f,0xdf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1df7fcf <unknown>

smlsl   za.s[w11, 6:7], {z30.h - z31.h}, z15.h[7]  // 11000001-11011111-01111111-11001111
// CHECK, INST: smlsl   za.s[w11, 6:7, vgx2], { z30.h, z31.h }, z15.h[7]
// CHECK-ENCODING: [0xcf,0x7f,0xdf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1df7fcf <unknown>

smlsl   za.s[w8, 2:3, vgx2], {z16.h, z17.h}, z0.h[7]  // 11000001, 11010000, 00011110, 00001101
// CHECK, INST: smlsl   za.s[w8, 2:3, vgx2], { z16.h, z17.h }, z0.h[7]
// CHECK-ENCODING: [0x0d,0x1e,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d01e0d <unknown>

smlsl   za.s[w8, 2:3], {z16.h - z17.h}, z0.h[7]  // 11000001-11010000-00011110-00001101
// CHECK, INST: smlsl   za.s[w8, 2:3, vgx2], { z16.h, z17.h }, z0.h[7]
// CHECK-ENCODING: [0x0d,0x1e,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d01e0d <unknown>

smlsl   za.s[w8, 2:3, vgx2], {z0.h, z1.h}, z14.h[2]  // 11000001, 11011110, 00010100, 00001001
// CHECK, INST: smlsl   za.s[w8, 2:3, vgx2], { z0.h, z1.h }, z14.h[2]
// CHECK-ENCODING: [0x09,0x14,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de1409 <unknown>

smlsl   za.s[w8, 2:3], {z0.h - z1.h}, z14.h[2]  // 11000001-11011110-00010100-00001001
// CHECK, INST: smlsl   za.s[w8, 2:3, vgx2], { z0.h, z1.h }, z14.h[2]
// CHECK-ENCODING: [0x09,0x14,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de1409 <unknown>

smlsl   za.s[w10, 0:1, vgx2], {z18.h, z19.h}, z4.h[2]  // 11000001, 11010100, 01010110, 01001000
// CHECK, INST: smlsl   za.s[w10, 0:1, vgx2], { z18.h, z19.h }, z4.h[2]
// CHECK-ENCODING: [0x48,0x56,0xd4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d45648 <unknown>

smlsl   za.s[w10, 0:1], {z18.h - z19.h}, z4.h[2]  // 11000001-11010100-01010110-01001000
// CHECK, INST: smlsl   za.s[w10, 0:1, vgx2], { z18.h, z19.h }, z4.h[2]
// CHECK-ENCODING: [0x48,0x56,0xd4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d45648 <unknown>

smlsl   za.s[w8, 0:1, vgx2], {z12.h, z13.h}, z2.h[4]  // 11000001, 11010010, 00011001, 10001000
// CHECK, INST: smlsl   za.s[w8, 0:1, vgx2], { z12.h, z13.h }, z2.h[4]
// CHECK-ENCODING: [0x88,0x19,0xd2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d21988 <unknown>

smlsl   za.s[w8, 0:1], {z12.h - z13.h}, z2.h[4]  // 11000001-11010010-00011001-10001000
// CHECK, INST: smlsl   za.s[w8, 0:1, vgx2], { z12.h, z13.h }, z2.h[4]
// CHECK-ENCODING: [0x88,0x19,0xd2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d21988 <unknown>

smlsl   za.s[w10, 2:3, vgx2], {z0.h, z1.h}, z10.h[4]  // 11000001, 11011010, 01011000, 00001001
// CHECK, INST: smlsl   za.s[w10, 2:3, vgx2], { z0.h, z1.h }, z10.h[4]
// CHECK-ENCODING: [0x09,0x58,0xda,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1da5809 <unknown>

smlsl   za.s[w10, 2:3], {z0.h - z1.h}, z10.h[4]  // 11000001-11011010-01011000-00001001
// CHECK, INST: smlsl   za.s[w10, 2:3, vgx2], { z0.h, z1.h }, z10.h[4]
// CHECK-ENCODING: [0x09,0x58,0xda,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1da5809 <unknown>

smlsl   za.s[w8, 2:3, vgx2], {z22.h, z23.h}, z14.h[5]  // 11000001, 11011110, 00011010, 11001101
// CHECK, INST: smlsl   za.s[w8, 2:3, vgx2], { z22.h, z23.h }, z14.h[5]
// CHECK-ENCODING: [0xcd,0x1a,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de1acd <unknown>

smlsl   za.s[w8, 2:3], {z22.h - z23.h}, z14.h[5]  // 11000001-11011110-00011010-11001101
// CHECK, INST: smlsl   za.s[w8, 2:3, vgx2], { z22.h, z23.h }, z14.h[5]
// CHECK-ENCODING: [0xcd,0x1a,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de1acd <unknown>

smlsl   za.s[w11, 4:5, vgx2], {z8.h, z9.h}, z1.h[2]  // 11000001, 11010001, 01110101, 00001010
// CHECK, INST: smlsl   za.s[w11, 4:5, vgx2], { z8.h, z9.h }, z1.h[2]
// CHECK-ENCODING: [0x0a,0x75,0xd1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d1750a <unknown>

smlsl   za.s[w11, 4:5], {z8.h - z9.h}, z1.h[2]  // 11000001-11010001-01110101-00001010
// CHECK, INST: smlsl   za.s[w11, 4:5, vgx2], { z8.h, z9.h }, z1.h[2]
// CHECK-ENCODING: [0x0a,0x75,0xd1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d1750a <unknown>

smlsl   za.s[w9, 6:7, vgx2], {z12.h, z13.h}, z11.h[5]  // 11000001, 11011011, 00111001, 10001111
// CHECK, INST: smlsl   za.s[w9, 6:7, vgx2], { z12.h, z13.h }, z11.h[5]
// CHECK-ENCODING: [0x8f,0x39,0xdb,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1db398f <unknown>

smlsl   za.s[w9, 6:7], {z12.h - z13.h}, z11.h[5]  // 11000001-11011011-00111001-10001111
// CHECK, INST: smlsl   za.s[w9, 6:7, vgx2], { z12.h, z13.h }, z11.h[5]
// CHECK-ENCODING: [0x8f,0x39,0xdb,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1db398f <unknown>


smlsl   za.s[w8, 0:1, vgx2], {z0.h, z1.h}, {z0.h, z1.h}  // 11000001, 11100000, 00001000, 00001000
// CHECK, INST: smlsl   za.s[w8, 0:1, vgx2], { z0.h, z1.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x08,0x08,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e00808 <unknown>

smlsl   za.s[w8, 0:1], {z0.h - z1.h}, {z0.h - z1.h}  // 11000001-11100000-00001000-00001000
// CHECK, INST: smlsl   za.s[w8, 0:1, vgx2], { z0.h, z1.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x08,0x08,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e00808 <unknown>

smlsl   za.s[w10, 2:3, vgx2], {z10.h, z11.h}, {z20.h, z21.h}  // 11000001, 11110100, 01001001, 01001001
// CHECK, INST: smlsl   za.s[w10, 2:3, vgx2], { z10.h, z11.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x49,0x49,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f44949 <unknown>

smlsl   za.s[w10, 2:3], {z10.h - z11.h}, {z20.h - z21.h}  // 11000001-11110100-01001001-01001001
// CHECK, INST: smlsl   za.s[w10, 2:3, vgx2], { z10.h, z11.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x49,0x49,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f44949 <unknown>

smlsl   za.s[w11, 6:7, vgx2], {z12.h, z13.h}, {z8.h, z9.h}  // 11000001, 11101000, 01101001, 10001011
// CHECK, INST: smlsl   za.s[w11, 6:7, vgx2], { z12.h, z13.h }, { z8.h, z9.h }
// CHECK-ENCODING: [0x8b,0x69,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e8698b <unknown>

smlsl   za.s[w11, 6:7], {z12.h - z13.h}, {z8.h - z9.h}  // 11000001-11101000-01101001-10001011
// CHECK, INST: smlsl   za.s[w11, 6:7, vgx2], { z12.h, z13.h }, { z8.h, z9.h }
// CHECK-ENCODING: [0x8b,0x69,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e8698b <unknown>

smlsl   za.s[w11, 6:7, vgx2], {z30.h, z31.h}, {z30.h, z31.h}  // 11000001, 11111110, 01101011, 11001011
// CHECK, INST: smlsl   za.s[w11, 6:7, vgx2], { z30.h, z31.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xcb,0x6b,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe6bcb <unknown>

smlsl   za.s[w11, 6:7], {z30.h - z31.h}, {z30.h - z31.h}  // 11000001-11111110-01101011-11001011
// CHECK, INST: smlsl   za.s[w11, 6:7, vgx2], { z30.h, z31.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xcb,0x6b,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe6bcb <unknown>

smlsl   za.s[w8, 2:3, vgx2], {z16.h, z17.h}, {z16.h, z17.h}  // 11000001, 11110000, 00001010, 00001001
// CHECK, INST: smlsl   za.s[w8, 2:3, vgx2], { z16.h, z17.h }, { z16.h, z17.h }
// CHECK-ENCODING: [0x09,0x0a,0xf0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f00a09 <unknown>

smlsl   za.s[w8, 2:3], {z16.h - z17.h}, {z16.h - z17.h}  // 11000001-11110000-00001010-00001001
// CHECK, INST: smlsl   za.s[w8, 2:3, vgx2], { z16.h, z17.h }, { z16.h, z17.h }
// CHECK-ENCODING: [0x09,0x0a,0xf0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f00a09 <unknown>

smlsl   za.s[w8, 2:3, vgx2], {z0.h, z1.h}, {z30.h, z31.h}  // 11000001, 11111110, 00001000, 00001001
// CHECK, INST: smlsl   za.s[w8, 2:3, vgx2], { z0.h, z1.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0x09,0x08,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe0809 <unknown>

smlsl   za.s[w8, 2:3], {z0.h - z1.h}, {z30.h - z31.h}  // 11000001-11111110-00001000-00001001
// CHECK, INST: smlsl   za.s[w8, 2:3, vgx2], { z0.h, z1.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0x09,0x08,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe0809 <unknown>

smlsl   za.s[w10, 0:1, vgx2], {z18.h, z19.h}, {z20.h, z21.h}  // 11000001, 11110100, 01001010, 01001000
// CHECK, INST: smlsl   za.s[w10, 0:1, vgx2], { z18.h, z19.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x48,0x4a,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f44a48 <unknown>

smlsl   za.s[w10, 0:1], {z18.h - z19.h}, {z20.h - z21.h}  // 11000001-11110100-01001010-01001000
// CHECK, INST: smlsl   za.s[w10, 0:1, vgx2], { z18.h, z19.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x48,0x4a,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f44a48 <unknown>

smlsl   za.s[w8, 0:1, vgx2], {z12.h, z13.h}, {z2.h, z3.h}  // 11000001, 11100010, 00001001, 10001000
// CHECK, INST: smlsl   za.s[w8, 0:1, vgx2], { z12.h, z13.h }, { z2.h, z3.h }
// CHECK-ENCODING: [0x88,0x09,0xe2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e20988 <unknown>

smlsl   za.s[w8, 0:1], {z12.h - z13.h}, {z2.h - z3.h}  // 11000001-11100010-00001001-10001000
// CHECK, INST: smlsl   za.s[w8, 0:1, vgx2], { z12.h, z13.h }, { z2.h, z3.h }
// CHECK-ENCODING: [0x88,0x09,0xe2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e20988 <unknown>

smlsl   za.s[w10, 2:3, vgx2], {z0.h, z1.h}, {z26.h, z27.h}  // 11000001, 11111010, 01001000, 00001001
// CHECK, INST: smlsl   za.s[w10, 2:3, vgx2], { z0.h, z1.h }, { z26.h, z27.h }
// CHECK-ENCODING: [0x09,0x48,0xfa,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fa4809 <unknown>

smlsl   za.s[w10, 2:3], {z0.h - z1.h}, {z26.h - z27.h}  // 11000001-11111010-01001000-00001001
// CHECK, INST: smlsl   za.s[w10, 2:3, vgx2], { z0.h, z1.h }, { z26.h, z27.h }
// CHECK-ENCODING: [0x09,0x48,0xfa,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fa4809 <unknown>

smlsl   za.s[w8, 2:3, vgx2], {z22.h, z23.h}, {z30.h, z31.h}  // 11000001, 11111110, 00001010, 11001001
// CHECK, INST: smlsl   za.s[w8, 2:3, vgx2], { z22.h, z23.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xc9,0x0a,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe0ac9 <unknown>

smlsl   za.s[w8, 2:3], {z22.h - z23.h}, {z30.h - z31.h}  // 11000001-11111110-00001010-11001001
// CHECK, INST: smlsl   za.s[w8, 2:3, vgx2], { z22.h, z23.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xc9,0x0a,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe0ac9 <unknown>

smlsl   za.s[w11, 4:5, vgx2], {z8.h, z9.h}, {z0.h, z1.h}  // 11000001, 11100000, 01101001, 00001010
// CHECK, INST: smlsl   za.s[w11, 4:5, vgx2], { z8.h, z9.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x0a,0x69,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e0690a <unknown>

smlsl   za.s[w11, 4:5], {z8.h - z9.h}, {z0.h - z1.h}  // 11000001-11100000-01101001-00001010
// CHECK, INST: smlsl   za.s[w11, 4:5, vgx2], { z8.h, z9.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x0a,0x69,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e0690a <unknown>

smlsl   za.s[w9, 6:7, vgx2], {z12.h, z13.h}, {z10.h, z11.h}  // 11000001, 11101010, 00101001, 10001011
// CHECK, INST: smlsl   za.s[w9, 6:7, vgx2], { z12.h, z13.h }, { z10.h, z11.h }
// CHECK-ENCODING: [0x8b,0x29,0xea,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ea298b <unknown>

smlsl   za.s[w9, 6:7], {z12.h - z13.h}, {z10.h - z11.h}  // 11000001-11101010-00101001-10001011
// CHECK, INST: smlsl   za.s[w9, 6:7, vgx2], { z12.h, z13.h }, { z10.h, z11.h }
// CHECK-ENCODING: [0x8b,0x29,0xea,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ea298b <unknown>


smlsl   za.s[w8, 0:1, vgx4], {z0.h - z3.h}, z0.h  // 11000001-01110000-00001000-00001000
// CHECK-INST: smlsl   za.s[w8, 0:1, vgx4], { z0.h - z3.h }, z0.h
// CHECK-ENCODING: [0x08,0x08,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1700808 <unknown>

smlsl   za.s[w8, 0:1], {z0.h - z3.h}, z0.h  // 11000001-01110000-00001000-00001000
// CHECK-INST: smlsl   za.s[w8, 0:1, vgx4], { z0.h - z3.h }, z0.h
// CHECK-ENCODING: [0x08,0x08,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1700808 <unknown>

smlsl   za.s[w10, 2:3, vgx4], {z10.h - z13.h}, z5.h  // 11000001-01110101-01001001-01001001
// CHECK-INST: smlsl   za.s[w10, 2:3, vgx4], { z10.h - z13.h }, z5.h
// CHECK-ENCODING: [0x49,0x49,0x75,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1754949 <unknown>

smlsl   za.s[w10, 2:3], {z10.h - z13.h}, z5.h  // 11000001-01110101-01001001-01001001
// CHECK-INST: smlsl   za.s[w10, 2:3, vgx4], { z10.h - z13.h }, z5.h
// CHECK-ENCODING: [0x49,0x49,0x75,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1754949 <unknown>

smlsl   za.s[w11, 6:7, vgx4], {z13.h - z16.h}, z8.h  // 11000001-01111000-01101001-10101011
// CHECK-INST: smlsl   za.s[w11, 6:7, vgx4], { z13.h - z16.h }, z8.h
// CHECK-ENCODING: [0xab,0x69,0x78,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17869ab <unknown>

smlsl   za.s[w11, 6:7], {z13.h - z16.h}, z8.h  // 11000001-01111000-01101001-10101011
// CHECK-INST: smlsl   za.s[w11, 6:7, vgx4], { z13.h - z16.h }, z8.h
// CHECK-ENCODING: [0xab,0x69,0x78,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17869ab <unknown>

smlsl   za.s[w11, 6:7, vgx4], {z31.h, z0.h, z1.h, z2.h}, z15.h  // 11000001-01111111-01101011-11101011
// CHECK-INST: smlsl   za.s[w11, 6:7, vgx4], { z31.h, z0.h, z1.h, z2.h }, z15.h
// CHECK-ENCODING: [0xeb,0x6b,0x7f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17f6beb <unknown>

smlsl   za.s[w11, 6:7], {z31.h, z0.h, z1.h, z2.h}, z15.h  // 11000001-01111111-01101011-11101011
// CHECK-INST: smlsl   za.s[w11, 6:7, vgx4], { z31.h, z0.h, z1.h, z2.h }, z15.h
// CHECK-ENCODING: [0xeb,0x6b,0x7f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17f6beb <unknown>

smlsl   za.s[w8, 2:3, vgx4], {z17.h - z20.h}, z0.h  // 11000001-01110000-00001010-00101001
// CHECK-INST: smlsl   za.s[w8, 2:3, vgx4], { z17.h - z20.h }, z0.h
// CHECK-ENCODING: [0x29,0x0a,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1700a29 <unknown>

smlsl   za.s[w8, 2:3], {z17.h - z20.h}, z0.h  // 11000001-01110000-00001010-00101001
// CHECK-INST: smlsl   za.s[w8, 2:3, vgx4], { z17.h - z20.h }, z0.h
// CHECK-ENCODING: [0x29,0x0a,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1700a29 <unknown>

smlsl   za.s[w8, 2:3, vgx4], {z1.h - z4.h}, z14.h  // 11000001-01111110-00001000-00101001
// CHECK-INST: smlsl   za.s[w8, 2:3, vgx4], { z1.h - z4.h }, z14.h
// CHECK-ENCODING: [0x29,0x08,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17e0829 <unknown>

smlsl   za.s[w8, 2:3], {z1.h - z4.h}, z14.h  // 11000001-01111110-00001000-00101001
// CHECK-INST: smlsl   za.s[w8, 2:3, vgx4], { z1.h - z4.h }, z14.h
// CHECK-ENCODING: [0x29,0x08,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17e0829 <unknown>

smlsl   za.s[w10, 0:1, vgx4], {z19.h - z22.h}, z4.h  // 11000001-01110100-01001010-01101000
// CHECK-INST: smlsl   za.s[w10, 0:1, vgx4], { z19.h - z22.h }, z4.h
// CHECK-ENCODING: [0x68,0x4a,0x74,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1744a68 <unknown>

smlsl   za.s[w10, 0:1], {z19.h - z22.h}, z4.h  // 11000001-01110100-01001010-01101000
// CHECK-INST: smlsl   za.s[w10, 0:1, vgx4], { z19.h - z22.h }, z4.h
// CHECK-ENCODING: [0x68,0x4a,0x74,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1744a68 <unknown>

smlsl   za.s[w8, 0:1, vgx4], {z12.h - z15.h}, z2.h  // 11000001-01110010-00001001-10001000
// CHECK-INST: smlsl   za.s[w8, 0:1, vgx4], { z12.h - z15.h }, z2.h
// CHECK-ENCODING: [0x88,0x09,0x72,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1720988 <unknown>

smlsl   za.s[w8, 0:1], {z12.h - z15.h}, z2.h  // 11000001-01110010-00001001-10001000
// CHECK-INST: smlsl   za.s[w8, 0:1, vgx4], { z12.h - z15.h }, z2.h
// CHECK-ENCODING: [0x88,0x09,0x72,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1720988 <unknown>

smlsl   za.s[w10, 2:3, vgx4], {z1.h - z4.h}, z10.h  // 11000001-01111010-01001000-00101001
// CHECK-INST: smlsl   za.s[w10, 2:3, vgx4], { z1.h - z4.h }, z10.h
// CHECK-ENCODING: [0x29,0x48,0x7a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17a4829 <unknown>

smlsl   za.s[w10, 2:3], {z1.h - z4.h}, z10.h  // 11000001-01111010-01001000-00101001
// CHECK-INST: smlsl   za.s[w10, 2:3, vgx4], { z1.h - z4.h }, z10.h
// CHECK-ENCODING: [0x29,0x48,0x7a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17a4829 <unknown>

smlsl   za.s[w8, 2:3, vgx4], {z22.h - z25.h}, z14.h  // 11000001-01111110-00001010-11001001
// CHECK-INST: smlsl   za.s[w8, 2:3, vgx4], { z22.h - z25.h }, z14.h
// CHECK-ENCODING: [0xc9,0x0a,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17e0ac9 <unknown>

smlsl   za.s[w8, 2:3], {z22.h - z25.h}, z14.h  // 11000001-01111110-00001010-11001001
// CHECK-INST: smlsl   za.s[w8, 2:3, vgx4], { z22.h - z25.h }, z14.h
// CHECK-ENCODING: [0xc9,0x0a,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17e0ac9 <unknown>

smlsl   za.s[w11, 4:5, vgx4], {z9.h - z12.h}, z1.h  // 11000001-01110001-01101001-00101010
// CHECK-INST: smlsl   za.s[w11, 4:5, vgx4], { z9.h - z12.h }, z1.h
// CHECK-ENCODING: [0x2a,0x69,0x71,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c171692a <unknown>

smlsl   za.s[w11, 4:5], {z9.h - z12.h}, z1.h  // 11000001-01110001-01101001-00101010
// CHECK-INST: smlsl   za.s[w11, 4:5, vgx4], { z9.h - z12.h }, z1.h
// CHECK-ENCODING: [0x2a,0x69,0x71,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c171692a <unknown>

smlsl   za.s[w9, 6:7, vgx4], {z12.h - z15.h}, z11.h  // 11000001-01111011-00101001-10001011
// CHECK-INST: smlsl   za.s[w9, 6:7, vgx4], { z12.h - z15.h }, z11.h
// CHECK-ENCODING: [0x8b,0x29,0x7b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17b298b <unknown>

smlsl   za.s[w9, 6:7], {z12.h - z15.h}, z11.h  // 11000001-01111011-00101001-10001011
// CHECK-INST: smlsl   za.s[w9, 6:7, vgx4], { z12.h - z15.h }, z11.h
// CHECK-ENCODING: [0x8b,0x29,0x7b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17b298b <unknown>


smlsl   za.s[w8, 0:1, vgx4], {z0.h - z3.h}, z0.h[0]  // 11000001-11010000-10010000-00001000
// CHECK-INST: smlsl   za.s[w8, 0:1, vgx4], { z0.h - z3.h }, z0.h[0]
// CHECK-ENCODING: [0x08,0x90,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d09008 <unknown>

smlsl   za.s[w8, 0:1], {z0.h - z3.h}, z0.h[0]  // 11000001-11010000-10010000-00001000
// CHECK-INST: smlsl   za.s[w8, 0:1, vgx4], { z0.h - z3.h }, z0.h[0]
// CHECK-ENCODING: [0x08,0x90,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d09008 <unknown>

smlsl   za.s[w10, 2:3, vgx4], {z8.h - z11.h}, z5.h[3]  // 11000001-11010101-11010101-00001101
// CHECK-INST: smlsl   za.s[w10, 2:3, vgx4], { z8.h - z11.h }, z5.h[3]
// CHECK-ENCODING: [0x0d,0xd5,0xd5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d5d50d <unknown>

smlsl   za.s[w10, 2:3], {z8.h - z11.h}, z5.h[3]  // 11000001-11010101-11010101-00001101
// CHECK-INST: smlsl   za.s[w10, 2:3, vgx4], { z8.h - z11.h }, z5.h[3]
// CHECK-ENCODING: [0x0d,0xd5,0xd5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d5d50d <unknown>

smlsl   za.s[w11, 6:7, vgx4], {z12.h - z15.h}, z8.h[7]  // 11000001-11011000-11111101-10001111
// CHECK-INST: smlsl   za.s[w11, 6:7, vgx4], { z12.h - z15.h }, z8.h[7]
// CHECK-ENCODING: [0x8f,0xfd,0xd8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d8fd8f <unknown>

smlsl   za.s[w11, 6:7], {z12.h - z15.h}, z8.h[7]  // 11000001-11011000-11111101-10001111
// CHECK-INST: smlsl   za.s[w11, 6:7, vgx4], { z12.h - z15.h }, z8.h[7]
// CHECK-ENCODING: [0x8f,0xfd,0xd8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d8fd8f <unknown>

smlsl   za.s[w11, 6:7, vgx4], {z28.h - z31.h}, z15.h[7]  // 11000001-11011111-11111111-10001111
// CHECK-INST: smlsl   za.s[w11, 6:7, vgx4], { z28.h - z31.h }, z15.h[7]
// CHECK-ENCODING: [0x8f,0xff,0xdf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dfff8f <unknown>

smlsl   za.s[w11, 6:7], {z28.h - z31.h}, z15.h[7]  // 11000001-11011111-11111111-10001111
// CHECK-INST: smlsl   za.s[w11, 6:7, vgx4], { z28.h - z31.h }, z15.h[7]
// CHECK-ENCODING: [0x8f,0xff,0xdf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dfff8f <unknown>

smlsl   za.s[w8, 2:3, vgx4], {z16.h - z19.h}, z0.h[7]  // 11000001-11010000-10011110-00001101
// CHECK-INST: smlsl   za.s[w8, 2:3, vgx4], { z16.h - z19.h }, z0.h[7]
// CHECK-ENCODING: [0x0d,0x9e,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d09e0d <unknown>

smlsl   za.s[w8, 2:3], {z16.h - z19.h}, z0.h[7]  // 11000001-11010000-10011110-00001101
// CHECK-INST: smlsl   za.s[w8, 2:3, vgx4], { z16.h - z19.h }, z0.h[7]
// CHECK-ENCODING: [0x0d,0x9e,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d09e0d <unknown>

smlsl   za.s[w8, 2:3, vgx4], {z0.h - z3.h}, z14.h[2]  // 11000001-11011110-10010100-00001001
// CHECK-INST: smlsl   za.s[w8, 2:3, vgx4], { z0.h - z3.h }, z14.h[2]
// CHECK-ENCODING: [0x09,0x94,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de9409 <unknown>

smlsl   za.s[w8, 2:3], {z0.h - z3.h}, z14.h[2]  // 11000001-11011110-10010100-00001001
// CHECK-INST: smlsl   za.s[w8, 2:3, vgx4], { z0.h - z3.h }, z14.h[2]
// CHECK-ENCODING: [0x09,0x94,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de9409 <unknown>

smlsl   za.s[w10, 0:1, vgx4], {z16.h - z19.h}, z4.h[2]  // 11000001-11010100-11010110-00001000
// CHECK-INST: smlsl   za.s[w10, 0:1, vgx4], { z16.h - z19.h }, z4.h[2]
// CHECK-ENCODING: [0x08,0xd6,0xd4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d4d608 <unknown>

smlsl   za.s[w10, 0:1], {z16.h - z19.h}, z4.h[2]  // 11000001-11010100-11010110-00001000
// CHECK-INST: smlsl   za.s[w10, 0:1, vgx4], { z16.h - z19.h }, z4.h[2]
// CHECK-ENCODING: [0x08,0xd6,0xd4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d4d608 <unknown>

smlsl   za.s[w8, 0:1, vgx4], {z12.h - z15.h}, z2.h[4]  // 11000001-11010010-10011001-10001000
// CHECK-INST: smlsl   za.s[w8, 0:1, vgx4], { z12.h - z15.h }, z2.h[4]
// CHECK-ENCODING: [0x88,0x99,0xd2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d29988 <unknown>

smlsl   za.s[w8, 0:1], {z12.h - z15.h}, z2.h[4]  // 11000001-11010010-10011001-10001000
// CHECK-INST: smlsl   za.s[w8, 0:1, vgx4], { z12.h - z15.h }, z2.h[4]
// CHECK-ENCODING: [0x88,0x99,0xd2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d29988 <unknown>

smlsl   za.s[w10, 2:3, vgx4], {z0.h - z3.h}, z10.h[4]  // 11000001-11011010-11011000-00001001
// CHECK-INST: smlsl   za.s[w10, 2:3, vgx4], { z0.h - z3.h }, z10.h[4]
// CHECK-ENCODING: [0x09,0xd8,0xda,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dad809 <unknown>

smlsl   za.s[w10, 2:3], {z0.h - z3.h}, z10.h[4]  // 11000001-11011010-11011000-00001001
// CHECK-INST: smlsl   za.s[w10, 2:3, vgx4], { z0.h - z3.h }, z10.h[4]
// CHECK-ENCODING: [0x09,0xd8,0xda,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dad809 <unknown>

smlsl   za.s[w8, 2:3, vgx4], {z20.h - z23.h}, z14.h[5]  // 11000001-11011110-10011010-10001101
// CHECK-INST: smlsl   za.s[w8, 2:3, vgx4], { z20.h - z23.h }, z14.h[5]
// CHECK-ENCODING: [0x8d,0x9a,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de9a8d <unknown>

smlsl   za.s[w8, 2:3], {z20.h - z23.h}, z14.h[5]  // 11000001-11011110-10011010-10001101
// CHECK-INST: smlsl   za.s[w8, 2:3, vgx4], { z20.h - z23.h }, z14.h[5]
// CHECK-ENCODING: [0x8d,0x9a,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de9a8d <unknown>

smlsl   za.s[w11, 4:5, vgx4], {z8.h - z11.h}, z1.h[2]  // 11000001-11010001-11110101-00001010
// CHECK-INST: smlsl   za.s[w11, 4:5, vgx4], { z8.h - z11.h }, z1.h[2]
// CHECK-ENCODING: [0x0a,0xf5,0xd1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d1f50a <unknown>

smlsl   za.s[w11, 4:5], {z8.h - z11.h}, z1.h[2]  // 11000001-11010001-11110101-00001010
// CHECK-INST: smlsl   za.s[w11, 4:5, vgx4], { z8.h - z11.h }, z1.h[2]
// CHECK-ENCODING: [0x0a,0xf5,0xd1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d1f50a <unknown>

smlsl   za.s[w9, 6:7, vgx4], {z12.h - z15.h}, z11.h[5]  // 11000001-11011011-10111001-10001111
// CHECK-INST: smlsl   za.s[w9, 6:7, vgx4], { z12.h - z15.h }, z11.h[5]
// CHECK-ENCODING: [0x8f,0xb9,0xdb,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dbb98f <unknown>

smlsl   za.s[w9, 6:7], {z12.h - z15.h}, z11.h[5]  // 11000001-11011011-10111001-10001111
// CHECK-INST: smlsl   za.s[w9, 6:7, vgx4], { z12.h - z15.h }, z11.h[5]
// CHECK-ENCODING: [0x8f,0xb9,0xdb,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dbb98f <unknown>


smlsl   za.s[w8, 0:1, vgx4], {z0.h - z3.h}, {z0.h - z3.h}  // 11000001-11100001-00001000-00001000
// CHECK-INST: smlsl   za.s[w8, 0:1, vgx4], { z0.h - z3.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x08,0x08,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e10808 <unknown>

smlsl   za.s[w8, 0:1], {z0.h - z3.h}, {z0.h - z3.h}  // 11000001-11100001-00001000-00001000
// CHECK-INST: smlsl   za.s[w8, 0:1, vgx4], { z0.h - z3.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x08,0x08,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e10808 <unknown>

smlsl   za.s[w10, 2:3, vgx4], {z8.h - z11.h}, {z20.h - z23.h}  // 11000001-11110101-01001001-00001001
// CHECK-INST: smlsl   za.s[w10, 2:3, vgx4], { z8.h - z11.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x09,0x49,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f54909 <unknown>

smlsl   za.s[w10, 2:3], {z8.h - z11.h}, {z20.h - z23.h}  // 11000001-11110101-01001001-00001001
// CHECK-INST: smlsl   za.s[w10, 2:3, vgx4], { z8.h - z11.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x09,0x49,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f54909 <unknown>

smlsl   za.s[w11, 6:7, vgx4], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-11101001-01101001-10001011
// CHECK-INST: smlsl   za.s[w11, 6:7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x8b,0x69,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e9698b <unknown>

smlsl   za.s[w11, 6:7], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-11101001-01101001-10001011
// CHECK-INST: smlsl   za.s[w11, 6:7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x8b,0x69,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e9698b <unknown>

smlsl   za.s[w11, 6:7, vgx4], {z28.h - z31.h}, {z28.h - z31.h}  // 11000001-11111101-01101011-10001011
// CHECK-INST: smlsl   za.s[w11, 6:7, vgx4], { z28.h - z31.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x8b,0x6b,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd6b8b <unknown>

smlsl   za.s[w11, 6:7], {z28.h - z31.h}, {z28.h - z31.h}  // 11000001-11111101-01101011-10001011
// CHECK-INST: smlsl   za.s[w11, 6:7, vgx4], { z28.h - z31.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x8b,0x6b,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd6b8b <unknown>

smlsl   za.s[w8, 2:3, vgx4], {z16.h - z19.h}, {z16.h - z19.h}  // 11000001-11110001-00001010-00001001
// CHECK-INST: smlsl   za.s[w8, 2:3, vgx4], { z16.h - z19.h }, { z16.h - z19.h }
// CHECK-ENCODING: [0x09,0x0a,0xf1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f10a09 <unknown>

smlsl   za.s[w8, 2:3], {z16.h - z19.h}, {z16.h - z19.h}  // 11000001-11110001-00001010-00001001
// CHECK-INST: smlsl   za.s[w8, 2:3, vgx4], { z16.h - z19.h }, { z16.h - z19.h }
// CHECK-ENCODING: [0x09,0x0a,0xf1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f10a09 <unknown>

smlsl   za.s[w8, 2:3, vgx4], {z0.h - z3.h}, {z28.h - z31.h}  // 11000001-11111101-00001000-00001001
// CHECK-INST: smlsl   za.s[w8, 2:3, vgx4], { z0.h - z3.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x09,0x08,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd0809 <unknown>

smlsl   za.s[w8, 2:3], {z0.h - z3.h}, {z28.h - z31.h}  // 11000001-11111101-00001000-00001001
// CHECK-INST: smlsl   za.s[w8, 2:3, vgx4], { z0.h - z3.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x09,0x08,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd0809 <unknown>

smlsl   za.s[w10, 0:1, vgx4], {z16.h - z19.h}, {z20.h - z23.h}  // 11000001-11110101-01001010-00001000
// CHECK-INST: smlsl   za.s[w10, 0:1, vgx4], { z16.h - z19.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x08,0x4a,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f54a08 <unknown>

smlsl   za.s[w10, 0:1], {z16.h - z19.h}, {z20.h - z23.h}  // 11000001-11110101-01001010-00001000
// CHECK-INST: smlsl   za.s[w10, 0:1, vgx4], { z16.h - z19.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x08,0x4a,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f54a08 <unknown>

smlsl   za.s[w8, 0:1, vgx4], {z12.h - z15.h}, {z0.h - z3.h}  // 11000001-11100001-00001001-10001000
// CHECK-INST: smlsl   za.s[w8, 0:1, vgx4], { z12.h - z15.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x88,0x09,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e10988 <unknown>

smlsl   za.s[w8, 0:1], {z12.h - z15.h}, {z0.h - z3.h}  // 11000001-11100001-00001001-10001000
// CHECK-INST: smlsl   za.s[w8, 0:1, vgx4], { z12.h - z15.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x88,0x09,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e10988 <unknown>

smlsl   za.s[w10, 2:3, vgx4], {z0.h - z3.h}, {z24.h - z27.h}  // 11000001-11111001-01001000-00001001
// CHECK-INST: smlsl   za.s[w10, 2:3, vgx4], { z0.h - z3.h }, { z24.h - z27.h }
// CHECK-ENCODING: [0x09,0x48,0xf9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f94809 <unknown>

smlsl   za.s[w10, 2:3], {z0.h - z3.h}, {z24.h - z27.h}  // 11000001-11111001-01001000-00001001
// CHECK-INST: smlsl   za.s[w10, 2:3, vgx4], { z0.h - z3.h }, { z24.h - z27.h }
// CHECK-ENCODING: [0x09,0x48,0xf9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f94809 <unknown>

smlsl   za.s[w8, 2:3, vgx4], {z20.h - z23.h}, {z28.h - z31.h}  // 11000001-11111101-00001010-10001001
// CHECK-INST: smlsl   za.s[w8, 2:3, vgx4], { z20.h - z23.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x89,0x0a,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd0a89 <unknown>

smlsl   za.s[w8, 2:3], {z20.h - z23.h}, {z28.h - z31.h}  // 11000001-11111101-00001010-10001001
// CHECK-INST: smlsl   za.s[w8, 2:3, vgx4], { z20.h - z23.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x89,0x0a,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd0a89 <unknown>

smlsl   za.s[w11, 4:5, vgx4], {z8.h - z11.h}, {z0.h - z3.h}  // 11000001-11100001-01101001-00001010
// CHECK-INST: smlsl   za.s[w11, 4:5, vgx4], { z8.h - z11.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x0a,0x69,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e1690a <unknown>

smlsl   za.s[w11, 4:5], {z8.h - z11.h}, {z0.h - z3.h}  // 11000001-11100001-01101001-00001010
// CHECK-INST: smlsl   za.s[w11, 4:5, vgx4], { z8.h - z11.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x0a,0x69,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e1690a <unknown>

smlsl   za.s[w9, 6:7, vgx4], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-11101001-00101001-10001011
// CHECK-INST: smlsl   za.s[w9, 6:7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x8b,0x29,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e9298b <unknown>

smlsl   za.s[w9, 6:7], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-11101001-00101001-10001011
// CHECK-INST: smlsl   za.s[w9, 6:7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x8b,0x29,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e9298b <unknown>

