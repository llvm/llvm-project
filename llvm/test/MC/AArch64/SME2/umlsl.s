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


umlsl   za.s[w8, 0:1], z0.h, z0.h  // 11000001-01100000-00001100-00011000
// CHECK-INST: umlsl   za.s[w8, 0:1], z0.h, z0.h
// CHECK-ENCODING: [0x18,0x0c,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1600c18 <unknown>

umlsl   za.s[w10, 10:11], z10.h, z5.h  // 11000001-01100101-01001101-01011101
// CHECK-INST: umlsl   za.s[w10, 10:11], z10.h, z5.h
// CHECK-ENCODING: [0x5d,0x4d,0x65,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1654d5d <unknown>

umlsl   za.s[w11, 14:15], z13.h, z8.h  // 11000001-01101000-01101101-10111111
// CHECK-INST: umlsl   za.s[w11, 14:15], z13.h, z8.h
// CHECK-ENCODING: [0xbf,0x6d,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1686dbf <unknown>

umlsl   za.s[w11, 14:15], z31.h, z15.h  // 11000001-01101111-01101111-11111111
// CHECK-INST: umlsl   za.s[w11, 14:15], z31.h, z15.h
// CHECK-ENCODING: [0xff,0x6f,0x6f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16f6fff <unknown>

umlsl   za.s[w8, 10:11], z17.h, z0.h  // 11000001-01100000-00001110-00111101
// CHECK-INST: umlsl   za.s[w8, 10:11], z17.h, z0.h
// CHECK-ENCODING: [0x3d,0x0e,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1600e3d <unknown>

umlsl   za.s[w8, 2:3], z1.h, z14.h  // 11000001-01101110-00001100-00111001
// CHECK-INST: umlsl   za.s[w8, 2:3], z1.h, z14.h
// CHECK-ENCODING: [0x39,0x0c,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e0c39 <unknown>

umlsl   za.s[w10, 0:1], z19.h, z4.h  // 11000001-01100100-01001110-01111000
// CHECK-INST: umlsl   za.s[w10, 0:1], z19.h, z4.h
// CHECK-ENCODING: [0x78,0x4e,0x64,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1644e78 <unknown>

umlsl   za.s[w8, 0:1], z12.h, z2.h  // 11000001-01100010-00001101-10011000
// CHECK-INST: umlsl   za.s[w8, 0:1], z12.h, z2.h
// CHECK-ENCODING: [0x98,0x0d,0x62,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1620d98 <unknown>

umlsl   za.s[w10, 2:3], z1.h, z10.h  // 11000001-01101010-01001100-00111001
// CHECK-INST: umlsl   za.s[w10, 2:3], z1.h, z10.h
// CHECK-ENCODING: [0x39,0x4c,0x6a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16a4c39 <unknown>

umlsl   za.s[w8, 10:11], z22.h, z14.h  // 11000001-01101110-00001110-11011101
// CHECK-INST: umlsl   za.s[w8, 10:11], z22.h, z14.h
// CHECK-ENCODING: [0xdd,0x0e,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e0edd <unknown>

umlsl   za.s[w11, 4:5], z9.h, z1.h  // 11000001-01100001-01101101-00111010
// CHECK-INST: umlsl   za.s[w11, 4:5], z9.h, z1.h
// CHECK-ENCODING: [0x3a,0x6d,0x61,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1616d3a <unknown>

umlsl   za.s[w9, 14:15], z12.h, z11.h  // 11000001-01101011-00101101-10011111
// CHECK-INST: umlsl   za.s[w9, 14:15], z12.h, z11.h
// CHECK-ENCODING: [0x9f,0x2d,0x6b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16b2d9f <unknown>


umlsl   za.s[w8, 0:1], z0.h, z0.h[0]  // 11000001-11000000-00010000-00011000
// CHECK-INST: umlsl   za.s[w8, 0:1], z0.h, z0.h[0]
// CHECK-ENCODING: [0x18,0x10,0xc0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1c01018 <unknown>

umlsl   za.s[w10, 10:11], z10.h, z5.h[1]  // 11000001-11000101-01010101-01011101
// CHECK-INST: umlsl   za.s[w10, 10:11], z10.h, z5.h[1]
// CHECK-ENCODING: [0x5d,0x55,0xc5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1c5555d <unknown>

umlsl   za.s[w11, 14:15], z13.h, z8.h[7]  // 11000001-11001000-11111101-10111111
// CHECK-INST: umlsl   za.s[w11, 14:15], z13.h, z8.h[7]
// CHECK-ENCODING: [0xbf,0xfd,0xc8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1c8fdbf <unknown>

umlsl   za.s[w11, 14:15], z31.h, z15.h[7]  // 11000001-11001111-11111111-11111111
// CHECK-INST: umlsl   za.s[w11, 14:15], z31.h, z15.h[7]
// CHECK-ENCODING: [0xff,0xff,0xcf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1cfffff <unknown>

umlsl   za.s[w8, 10:11], z17.h, z0.h[3]  // 11000001-11000000-00011110-00111101
// CHECK-INST: umlsl   za.s[w8, 10:11], z17.h, z0.h[3]
// CHECK-ENCODING: [0x3d,0x1e,0xc0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1c01e3d <unknown>

umlsl   za.s[w8, 2:3], z1.h, z14.h[5]  // 11000001-11001110-10010100-00111001
// CHECK-INST: umlsl   za.s[w8, 2:3], z1.h, z14.h[5]
// CHECK-ENCODING: [0x39,0x94,0xce,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ce9439 <unknown>

umlsl   za.s[w10, 0:1], z19.h, z4.h[1]  // 11000001-11000100-01010110-01111000
// CHECK-INST: umlsl   za.s[w10, 0:1], z19.h, z4.h[1]
// CHECK-ENCODING: [0x78,0x56,0xc4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1c45678 <unknown>

umlsl   za.s[w8, 0:1], z12.h, z2.h[2]  // 11000001-11000010-00011001-10011000
// CHECK-INST: umlsl   za.s[w8, 0:1], z12.h, z2.h[2]
// CHECK-ENCODING: [0x98,0x19,0xc2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1c21998 <unknown>

umlsl   za.s[w10, 2:3], z1.h, z10.h[6]  // 11000001-11001010-11011000-00111001
// CHECK-INST: umlsl   za.s[w10, 2:3], z1.h, z10.h[6]
// CHECK-ENCODING: [0x39,0xd8,0xca,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1cad839 <unknown>

umlsl   za.s[w8, 10:11], z22.h, z14.h[2]  // 11000001-11001110-00011010-11011101
// CHECK-INST: umlsl   za.s[w8, 10:11], z22.h, z14.h[2]
// CHECK-ENCODING: [0xdd,0x1a,0xce,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ce1add <unknown>

umlsl   za.s[w11, 4:5], z9.h, z1.h[5]  // 11000001-11000001-11110101-00111010
// CHECK-INST: umlsl   za.s[w11, 4:5], z9.h, z1.h[5]
// CHECK-ENCODING: [0x3a,0xf5,0xc1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1c1f53a <unknown>

umlsl   za.s[w9, 14:15], z12.h, z11.h[6]  // 11000001-11001011-10111001-10011111
// CHECK-INST: umlsl   za.s[w9, 14:15], z12.h, z11.h[6]
// CHECK-ENCODING: [0x9f,0xb9,0xcb,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1cbb99f <unknown>


umlsl   za.s[w8, 0:1, vgx2], {z0.h, z1.h}, z0.h  // 11000001-01100000-00001000-00011000
// CHECK-INST: umlsl   za.s[w8, 0:1, vgx2], { z0.h, z1.h }, z0.h
// CHECK-ENCODING: [0x18,0x08,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1600818 <unknown>

umlsl   za.s[w8, 0:1], {z0.h - z1.h}, z0.h  // 11000001-01100000-00001000-00011000
// CHECK-INST: umlsl   za.s[w8, 0:1, vgx2], { z0.h, z1.h }, z0.h
// CHECK-ENCODING: [0x18,0x08,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1600818 <unknown>

umlsl   za.s[w10, 2:3, vgx2], {z10.h, z11.h}, z5.h  // 11000001-01100101-01001001-01011001
// CHECK-INST: umlsl   za.s[w10, 2:3, vgx2], { z10.h, z11.h }, z5.h
// CHECK-ENCODING: [0x59,0x49,0x65,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1654959 <unknown>

umlsl   za.s[w10, 2:3], {z10.h - z11.h}, z5.h  // 11000001-01100101-01001001-01011001
// CHECK-INST: umlsl   za.s[w10, 2:3, vgx2], { z10.h, z11.h }, z5.h
// CHECK-ENCODING: [0x59,0x49,0x65,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1654959 <unknown>

umlsl   za.s[w11, 6:7, vgx2], {z13.h, z14.h}, z8.h  // 11000001-01101000-01101001-10111011
// CHECK-INST: umlsl   za.s[w11, 6:7, vgx2], { z13.h, z14.h }, z8.h
// CHECK-ENCODING: [0xbb,0x69,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16869bb <unknown>

umlsl   za.s[w11, 6:7], {z13.h - z14.h}, z8.h  // 11000001-01101000-01101001-10111011
// CHECK-INST: umlsl   za.s[w11, 6:7, vgx2], { z13.h, z14.h }, z8.h
// CHECK-ENCODING: [0xbb,0x69,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16869bb <unknown>

umlsl   za.s[w11, 6:7, vgx2], {z31.h, z0.h}, z15.h  // 11000001-01101111-01101011-11111011
// CHECK-INST: umlsl   za.s[w11, 6:7, vgx2], { z31.h, z0.h }, z15.h
// CHECK-ENCODING: [0xfb,0x6b,0x6f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16f6bfb <unknown>

umlsl   za.s[w11, 6:7], {z31.h - z0.h}, z15.h  // 11000001-01101111-01101011-11111011
// CHECK-INST: umlsl   za.s[w11, 6:7, vgx2], { z31.h, z0.h }, z15.h
// CHECK-ENCODING: [0xfb,0x6b,0x6f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16f6bfb <unknown>

umlsl   za.s[w8, 2:3, vgx2], {z17.h, z18.h}, z0.h  // 11000001-01100000-00001010-00111001
// CHECK-INST: umlsl   za.s[w8, 2:3, vgx2], { z17.h, z18.h }, z0.h
// CHECK-ENCODING: [0x39,0x0a,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1600a39 <unknown>

umlsl   za.s[w8, 2:3], {z17.h - z18.h}, z0.h  // 11000001-01100000-00001010-00111001
// CHECK-INST: umlsl   za.s[w8, 2:3, vgx2], { z17.h, z18.h }, z0.h
// CHECK-ENCODING: [0x39,0x0a,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1600a39 <unknown>

umlsl   za.s[w8, 2:3, vgx2], {z1.h, z2.h}, z14.h  // 11000001-01101110-00001000-00111001
// CHECK-INST: umlsl   za.s[w8, 2:3, vgx2], { z1.h, z2.h }, z14.h
// CHECK-ENCODING: [0x39,0x08,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e0839 <unknown>

umlsl   za.s[w8, 2:3], {z1.h - z2.h}, z14.h  // 11000001-01101110-00001000-00111001
// CHECK-INST: umlsl   za.s[w8, 2:3, vgx2], { z1.h, z2.h }, z14.h
// CHECK-ENCODING: [0x39,0x08,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e0839 <unknown>

umlsl   za.s[w10, 0:1, vgx2], {z19.h, z20.h}, z4.h  // 11000001-01100100-01001010-01111000
// CHECK-INST: umlsl   za.s[w10, 0:1, vgx2], { z19.h, z20.h }, z4.h
// CHECK-ENCODING: [0x78,0x4a,0x64,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1644a78 <unknown>

umlsl   za.s[w10, 0:1], {z19.h - z20.h}, z4.h  // 11000001-01100100-01001010-01111000
// CHECK-INST: umlsl   za.s[w10, 0:1, vgx2], { z19.h, z20.h }, z4.h
// CHECK-ENCODING: [0x78,0x4a,0x64,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1644a78 <unknown>

umlsl   za.s[w8, 0:1, vgx2], {z12.h, z13.h}, z2.h  // 11000001-01100010-00001001-10011000
// CHECK-INST: umlsl   za.s[w8, 0:1, vgx2], { z12.h, z13.h }, z2.h
// CHECK-ENCODING: [0x98,0x09,0x62,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1620998 <unknown>

umlsl   za.s[w8, 0:1], {z12.h - z13.h}, z2.h  // 11000001-01100010-00001001-10011000
// CHECK-INST: umlsl   za.s[w8, 0:1, vgx2], { z12.h, z13.h }, z2.h
// CHECK-ENCODING: [0x98,0x09,0x62,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1620998 <unknown>

umlsl   za.s[w10, 2:3, vgx2], {z1.h, z2.h}, z10.h  // 11000001-01101010-01001000-00111001
// CHECK-INST: umlsl   za.s[w10, 2:3, vgx2], { z1.h, z2.h }, z10.h
// CHECK-ENCODING: [0x39,0x48,0x6a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16a4839 <unknown>

umlsl   za.s[w10, 2:3], {z1.h - z2.h}, z10.h  // 11000001-01101010-01001000-00111001
// CHECK-INST: umlsl   za.s[w10, 2:3, vgx2], { z1.h, z2.h }, z10.h
// CHECK-ENCODING: [0x39,0x48,0x6a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16a4839 <unknown>

umlsl   za.s[w8, 2:3, vgx2], {z22.h, z23.h}, z14.h  // 11000001-01101110-00001010-11011001
// CHECK-INST: umlsl   za.s[w8, 2:3, vgx2], { z22.h, z23.h }, z14.h
// CHECK-ENCODING: [0xd9,0x0a,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e0ad9 <unknown>

umlsl   za.s[w8, 2:3], {z22.h - z23.h}, z14.h  // 11000001-01101110-00001010-11011001
// CHECK-INST: umlsl   za.s[w8, 2:3, vgx2], { z22.h, z23.h }, z14.h
// CHECK-ENCODING: [0xd9,0x0a,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e0ad9 <unknown>

umlsl   za.s[w11, 4:5, vgx2], {z9.h, z10.h}, z1.h  // 11000001-01100001-01101001-00111010
// CHECK-INST: umlsl   za.s[w11, 4:5, vgx2], { z9.h, z10.h }, z1.h
// CHECK-ENCODING: [0x3a,0x69,0x61,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c161693a <unknown>

umlsl   za.s[w11, 4:5], {z9.h - z10.h}, z1.h  // 11000001-01100001-01101001-00111010
// CHECK-INST: umlsl   za.s[w11, 4:5, vgx2], { z9.h, z10.h }, z1.h
// CHECK-ENCODING: [0x3a,0x69,0x61,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c161693a <unknown>

umlsl   za.s[w9, 6:7, vgx2], {z12.h, z13.h}, z11.h  // 11000001-01101011-00101001-10011011
// CHECK-INST: umlsl   za.s[w9, 6:7, vgx2], { z12.h, z13.h }, z11.h
// CHECK-ENCODING: [0x9b,0x29,0x6b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16b299b <unknown>

umlsl   za.s[w9, 6:7], {z12.h - z13.h}, z11.h  // 11000001-01101011-00101001-10011011
// CHECK-INST: umlsl   za.s[w9, 6:7, vgx2], { z12.h, z13.h }, z11.h
// CHECK-ENCODING: [0x9b,0x29,0x6b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16b299b <unknown>


umlsl   za.s[w8, 0:1, vgx2], {z0.h, z1.h}, z0.h[0]  // 11000001-11010000-00010000-00011000
// CHECK-INST: umlsl   za.s[w8, 0:1, vgx2], { z0.h, z1.h }, z0.h[0]
// CHECK-ENCODING: [0x18,0x10,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d01018 <unknown>

umlsl   za.s[w8, 0:1], {z0.h - z1.h}, z0.h[0]  // 11000001-11010000-00010000-00011000
// CHECK-INST: umlsl   za.s[w8, 0:1, vgx2], { z0.h, z1.h }, z0.h[0]
// CHECK-ENCODING: [0x18,0x10,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d01018 <unknown>

umlsl   za.s[w10, 2:3, vgx2], {z10.h, z11.h}, z5.h[3]  // 11000001-11010101-01010101-01011101
// CHECK-INST: umlsl   za.s[w10, 2:3, vgx2], { z10.h, z11.h }, z5.h[3]
// CHECK-ENCODING: [0x5d,0x55,0xd5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d5555d <unknown>

umlsl   za.s[w10, 2:3], {z10.h - z11.h}, z5.h[3]  // 11000001-11010101-01010101-01011101
// CHECK-INST: umlsl   za.s[w10, 2:3, vgx2], { z10.h, z11.h }, z5.h[3]
// CHECK-ENCODING: [0x5d,0x55,0xd5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d5555d <unknown>

umlsl   za.s[w11, 6:7, vgx2], {z12.h, z13.h}, z8.h[7]  // 11000001-11011000-01111101-10011111
// CHECK-INST: umlsl   za.s[w11, 6:7, vgx2], { z12.h, z13.h }, z8.h[7]
// CHECK-ENCODING: [0x9f,0x7d,0xd8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d87d9f <unknown>

umlsl   za.s[w11, 6:7], {z12.h - z13.h}, z8.h[7]  // 11000001-11011000-01111101-10011111
// CHECK-INST: umlsl   za.s[w11, 6:7, vgx2], { z12.h, z13.h }, z8.h[7]
// CHECK-ENCODING: [0x9f,0x7d,0xd8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d87d9f <unknown>

umlsl   za.s[w11, 6:7, vgx2], {z30.h, z31.h}, z15.h[7]  // 11000001-11011111-01111111-11011111
// CHECK-INST: umlsl   za.s[w11, 6:7, vgx2], { z30.h, z31.h }, z15.h[7]
// CHECK-ENCODING: [0xdf,0x7f,0xdf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1df7fdf <unknown>

umlsl   za.s[w11, 6:7], {z30.h - z31.h}, z15.h[7]  // 11000001-11011111-01111111-11011111
// CHECK-INST: umlsl   za.s[w11, 6:7, vgx2], { z30.h, z31.h }, z15.h[7]
// CHECK-ENCODING: [0xdf,0x7f,0xdf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1df7fdf <unknown>

umlsl   za.s[w8, 2:3, vgx2], {z16.h, z17.h}, z0.h[7]  // 11000001-11010000-00011110-00011101
// CHECK-INST: umlsl   za.s[w8, 2:3, vgx2], { z16.h, z17.h }, z0.h[7]
// CHECK-ENCODING: [0x1d,0x1e,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d01e1d <unknown>

umlsl   za.s[w8, 2:3], {z16.h - z17.h}, z0.h[7]  // 11000001-11010000-00011110-00011101
// CHECK-INST: umlsl   za.s[w8, 2:3, vgx2], { z16.h, z17.h }, z0.h[7]
// CHECK-ENCODING: [0x1d,0x1e,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d01e1d <unknown>

umlsl   za.s[w8, 2:3, vgx2], {z0.h, z1.h}, z14.h[2]  // 11000001-11011110-00010100-00011001
// CHECK-INST: umlsl   za.s[w8, 2:3, vgx2], { z0.h, z1.h }, z14.h[2]
// CHECK-ENCODING: [0x19,0x14,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de1419 <unknown>

umlsl   za.s[w8, 2:3], {z0.h - z1.h}, z14.h[2]  // 11000001-11011110-00010100-00011001
// CHECK-INST: umlsl   za.s[w8, 2:3, vgx2], { z0.h, z1.h }, z14.h[2]
// CHECK-ENCODING: [0x19,0x14,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de1419 <unknown>

umlsl   za.s[w10, 0:1, vgx2], {z18.h, z19.h}, z4.h[2]  // 11000001-11010100-01010110-01011000
// CHECK-INST: umlsl   za.s[w10, 0:1, vgx2], { z18.h, z19.h }, z4.h[2]
// CHECK-ENCODING: [0x58,0x56,0xd4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d45658 <unknown>

umlsl   za.s[w10, 0:1], {z18.h - z19.h}, z4.h[2]  // 11000001-11010100-01010110-01011000
// CHECK-INST: umlsl   za.s[w10, 0:1, vgx2], { z18.h, z19.h }, z4.h[2]
// CHECK-ENCODING: [0x58,0x56,0xd4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d45658 <unknown>

umlsl   za.s[w8, 0:1, vgx2], {z12.h, z13.h}, z2.h[4]  // 11000001-11010010-00011001-10011000
// CHECK-INST: umlsl   za.s[w8, 0:1, vgx2], { z12.h, z13.h }, z2.h[4]
// CHECK-ENCODING: [0x98,0x19,0xd2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d21998 <unknown>

umlsl   za.s[w8, 0:1], {z12.h - z13.h}, z2.h[4]  // 11000001-11010010-00011001-10011000
// CHECK-INST: umlsl   za.s[w8, 0:1, vgx2], { z12.h, z13.h }, z2.h[4]
// CHECK-ENCODING: [0x98,0x19,0xd2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d21998 <unknown>

umlsl   za.s[w10, 2:3, vgx2], {z0.h, z1.h}, z10.h[4]  // 11000001-11011010-01011000-00011001
// CHECK-INST: umlsl   za.s[w10, 2:3, vgx2], { z0.h, z1.h }, z10.h[4]
// CHECK-ENCODING: [0x19,0x58,0xda,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1da5819 <unknown>

umlsl   za.s[w10, 2:3], {z0.h - z1.h}, z10.h[4]  // 11000001-11011010-01011000-00011001
// CHECK-INST: umlsl   za.s[w10, 2:3, vgx2], { z0.h, z1.h }, z10.h[4]
// CHECK-ENCODING: [0x19,0x58,0xda,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1da5819 <unknown>

umlsl   za.s[w8, 2:3, vgx2], {z22.h, z23.h}, z14.h[5]  // 11000001-11011110-00011010-11011101
// CHECK-INST: umlsl   za.s[w8, 2:3, vgx2], { z22.h, z23.h }, z14.h[5]
// CHECK-ENCODING: [0xdd,0x1a,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de1add <unknown>

umlsl   za.s[w8, 2:3], {z22.h - z23.h}, z14.h[5]  // 11000001-11011110-00011010-11011101
// CHECK-INST: umlsl   za.s[w8, 2:3, vgx2], { z22.h, z23.h }, z14.h[5]
// CHECK-ENCODING: [0xdd,0x1a,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de1add <unknown>

umlsl   za.s[w11, 4:5, vgx2], {z8.h, z9.h}, z1.h[2]  // 11000001-11010001-01110101-00011010
// CHECK-INST: umlsl   za.s[w11, 4:5, vgx2], { z8.h, z9.h }, z1.h[2]
// CHECK-ENCODING: [0x1a,0x75,0xd1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d1751a <unknown>

umlsl   za.s[w11, 4:5], {z8.h - z9.h}, z1.h[2]  // 11000001-11010001-01110101-00011010
// CHECK-INST: umlsl   za.s[w11, 4:5, vgx2], { z8.h, z9.h }, z1.h[2]
// CHECK-ENCODING: [0x1a,0x75,0xd1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d1751a <unknown>

umlsl   za.s[w9, 6:7, vgx2], {z12.h, z13.h}, z11.h[5]  // 11000001-11011011-00111001-10011111
// CHECK-INST: umlsl   za.s[w9, 6:7, vgx2], { z12.h, z13.h }, z11.h[5]
// CHECK-ENCODING: [0x9f,0x39,0xdb,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1db399f <unknown>

umlsl   za.s[w9, 6:7], {z12.h - z13.h}, z11.h[5]  // 11000001-11011011-00111001-10011111
// CHECK-INST: umlsl   za.s[w9, 6:7, vgx2], { z12.h, z13.h }, z11.h[5]
// CHECK-ENCODING: [0x9f,0x39,0xdb,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1db399f <unknown>


umlsl   za.s[w8, 0:1, vgx2], {z0.h, z1.h}, {z0.h, z1.h}  // 11000001-11100000-00001000-00011000
// CHECK-INST: umlsl   za.s[w8, 0:1, vgx2], { z0.h, z1.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x18,0x08,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e00818 <unknown>

umlsl   za.s[w8, 0:1], {z0.h - z1.h}, {z0.h - z1.h}  // 11000001-11100000-00001000-00011000
// CHECK-INST: umlsl   za.s[w8, 0:1, vgx2], { z0.h, z1.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x18,0x08,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e00818 <unknown>

umlsl   za.s[w10, 2:3, vgx2], {z10.h, z11.h}, {z20.h, z21.h}  // 11000001-11110100-01001001-01011001
// CHECK-INST: umlsl   za.s[w10, 2:3, vgx2], { z10.h, z11.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x59,0x49,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f44959 <unknown>

umlsl   za.s[w10, 2:3], {z10.h - z11.h}, {z20.h - z21.h}  // 11000001-11110100-01001001-01011001
// CHECK-INST: umlsl   za.s[w10, 2:3, vgx2], { z10.h, z11.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x59,0x49,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f44959 <unknown>

umlsl   za.s[w11, 6:7, vgx2], {z12.h, z13.h}, {z8.h, z9.h}  // 11000001-11101000-01101001-10011011
// CHECK-INST: umlsl   za.s[w11, 6:7, vgx2], { z12.h, z13.h }, { z8.h, z9.h }
// CHECK-ENCODING: [0x9b,0x69,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e8699b <unknown>

umlsl   za.s[w11, 6:7], {z12.h - z13.h}, {z8.h - z9.h}  // 11000001-11101000-01101001-10011011
// CHECK-INST: umlsl   za.s[w11, 6:7, vgx2], { z12.h, z13.h }, { z8.h, z9.h }
// CHECK-ENCODING: [0x9b,0x69,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e8699b <unknown>

umlsl   za.s[w11, 6:7, vgx2], {z30.h, z31.h}, {z30.h, z31.h}  // 11000001-11111110-01101011-11011011
// CHECK-INST: umlsl   za.s[w11, 6:7, vgx2], { z30.h, z31.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xdb,0x6b,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe6bdb <unknown>

umlsl   za.s[w11, 6:7], {z30.h - z31.h}, {z30.h - z31.h}  // 11000001-11111110-01101011-11011011
// CHECK-INST: umlsl   za.s[w11, 6:7, vgx2], { z30.h, z31.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xdb,0x6b,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe6bdb <unknown>

umlsl   za.s[w8, 2:3, vgx2], {z16.h, z17.h}, {z16.h, z17.h}  // 11000001-11110000-00001010-00011001
// CHECK-INST: umlsl   za.s[w8, 2:3, vgx2], { z16.h, z17.h }, { z16.h, z17.h }
// CHECK-ENCODING: [0x19,0x0a,0xf0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f00a19 <unknown>

umlsl   za.s[w8, 2:3], {z16.h - z17.h}, {z16.h - z17.h}  // 11000001-11110000-00001010-00011001
// CHECK-INST: umlsl   za.s[w8, 2:3, vgx2], { z16.h, z17.h }, { z16.h, z17.h }
// CHECK-ENCODING: [0x19,0x0a,0xf0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f00a19 <unknown>

umlsl   za.s[w8, 2:3, vgx2], {z0.h, z1.h}, {z30.h, z31.h}  // 11000001-11111110-00001000-00011001
// CHECK-INST: umlsl   za.s[w8, 2:3, vgx2], { z0.h, z1.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0x19,0x08,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe0819 <unknown>

umlsl   za.s[w8, 2:3], {z0.h - z1.h}, {z30.h - z31.h}  // 11000001-11111110-00001000-00011001
// CHECK-INST: umlsl   za.s[w8, 2:3, vgx2], { z0.h, z1.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0x19,0x08,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe0819 <unknown>

umlsl   za.s[w10, 0:1, vgx2], {z18.h, z19.h}, {z20.h, z21.h}  // 11000001-11110100-01001010-01011000
// CHECK-INST: umlsl   za.s[w10, 0:1, vgx2], { z18.h, z19.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x58,0x4a,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f44a58 <unknown>

umlsl   za.s[w10, 0:1], {z18.h - z19.h}, {z20.h - z21.h}  // 11000001-11110100-01001010-01011000
// CHECK-INST: umlsl   za.s[w10, 0:1, vgx2], { z18.h, z19.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x58,0x4a,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f44a58 <unknown>

umlsl   za.s[w8, 0:1, vgx2], {z12.h, z13.h}, {z2.h, z3.h}  // 11000001-11100010-00001001-10011000
// CHECK-INST: umlsl   za.s[w8, 0:1, vgx2], { z12.h, z13.h }, { z2.h, z3.h }
// CHECK-ENCODING: [0x98,0x09,0xe2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e20998 <unknown>

umlsl   za.s[w8, 0:1], {z12.h - z13.h}, {z2.h - z3.h}  // 11000001-11100010-00001001-10011000
// CHECK-INST: umlsl   za.s[w8, 0:1, vgx2], { z12.h, z13.h }, { z2.h, z3.h }
// CHECK-ENCODING: [0x98,0x09,0xe2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e20998 <unknown>

umlsl   za.s[w10, 2:3, vgx2], {z0.h, z1.h}, {z26.h, z27.h}  // 11000001-11111010-01001000-00011001
// CHECK-INST: umlsl   za.s[w10, 2:3, vgx2], { z0.h, z1.h }, { z26.h, z27.h }
// CHECK-ENCODING: [0x19,0x48,0xfa,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fa4819 <unknown>

umlsl   za.s[w10, 2:3], {z0.h - z1.h}, {z26.h - z27.h}  // 11000001-11111010-01001000-00011001
// CHECK-INST: umlsl   za.s[w10, 2:3, vgx2], { z0.h, z1.h }, { z26.h, z27.h }
// CHECK-ENCODING: [0x19,0x48,0xfa,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fa4819 <unknown>

umlsl   za.s[w8, 2:3, vgx2], {z22.h, z23.h}, {z30.h, z31.h}  // 11000001-11111110-00001010-11011001
// CHECK-INST: umlsl   za.s[w8, 2:3, vgx2], { z22.h, z23.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xd9,0x0a,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe0ad9 <unknown>

umlsl   za.s[w8, 2:3], {z22.h - z23.h}, {z30.h - z31.h}  // 11000001-11111110-00001010-11011001
// CHECK-INST: umlsl   za.s[w8, 2:3, vgx2], { z22.h, z23.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xd9,0x0a,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe0ad9 <unknown>

umlsl   za.s[w11, 4:5, vgx2], {z8.h, z9.h}, {z0.h, z1.h}  // 11000001-11100000-01101001-00011010
// CHECK-INST: umlsl   za.s[w11, 4:5, vgx2], { z8.h, z9.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x1a,0x69,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e0691a <unknown>

umlsl   za.s[w11, 4:5], {z8.h - z9.h}, {z0.h - z1.h}  // 11000001-11100000-01101001-00011010
// CHECK-INST: umlsl   za.s[w11, 4:5, vgx2], { z8.h, z9.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x1a,0x69,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e0691a <unknown>

umlsl   za.s[w9, 6:7, vgx2], {z12.h, z13.h}, {z10.h, z11.h}  // 11000001-11101010-00101001-10011011
// CHECK-INST: umlsl   za.s[w9, 6:7, vgx2], { z12.h, z13.h }, { z10.h, z11.h }
// CHECK-ENCODING: [0x9b,0x29,0xea,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ea299b <unknown>

umlsl   za.s[w9, 6:7], {z12.h - z13.h}, {z10.h - z11.h}  // 11000001-11101010-00101001-10011011
// CHECK-INST: umlsl   za.s[w9, 6:7, vgx2], { z12.h, z13.h }, { z10.h, z11.h }
// CHECK-ENCODING: [0x9b,0x29,0xea,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ea299b <unknown>


umlsl   za.s[w8, 0:1, vgx4], {z0.h - z3.h}, z0.h  // 11000001-01110000-00001000-00011000
// CHECK-INST: umlsl   za.s[w8, 0:1, vgx4], { z0.h - z3.h }, z0.h
// CHECK-ENCODING: [0x18,0x08,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1700818 <unknown>

umlsl   za.s[w8, 0:1], {z0.h - z3.h}, z0.h  // 11000001-01110000-00001000-00011000
// CHECK-INST: umlsl   za.s[w8, 0:1, vgx4], { z0.h - z3.h }, z0.h
// CHECK-ENCODING: [0x18,0x08,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1700818 <unknown>

umlsl   za.s[w10, 2:3, vgx4], {z10.h - z13.h}, z5.h  // 11000001-01110101-01001001-01011001
// CHECK-INST: umlsl   za.s[w10, 2:3, vgx4], { z10.h - z13.h }, z5.h
// CHECK-ENCODING: [0x59,0x49,0x75,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1754959 <unknown>

umlsl   za.s[w10, 2:3], {z10.h - z13.h}, z5.h  // 11000001-01110101-01001001-01011001
// CHECK-INST: umlsl   za.s[w10, 2:3, vgx4], { z10.h - z13.h }, z5.h
// CHECK-ENCODING: [0x59,0x49,0x75,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1754959 <unknown>

umlsl   za.s[w11, 6:7, vgx4], {z13.h - z16.h}, z8.h  // 11000001-01111000-01101001-10111011
// CHECK-INST: umlsl   za.s[w11, 6:7, vgx4], { z13.h - z16.h }, z8.h
// CHECK-ENCODING: [0xbb,0x69,0x78,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17869bb <unknown>

umlsl   za.s[w11, 6:7], {z13.h - z16.h}, z8.h  // 11000001-01111000-01101001-10111011
// CHECK-INST: umlsl   za.s[w11, 6:7, vgx4], { z13.h - z16.h }, z8.h
// CHECK-ENCODING: [0xbb,0x69,0x78,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17869bb <unknown>

umlsl   za.s[w11, 6:7, vgx4], {z31.h, z0.h, z1.h, z2.h}, z15.h  // 11000001-01111111-01101011-11111011
// CHECK-INST: umlsl   za.s[w11, 6:7, vgx4], { z31.h, z0.h, z1.h, z2.h }, z15.h
// CHECK-ENCODING: [0xfb,0x6b,0x7f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17f6bfb <unknown>

umlsl   za.s[w11, 6:7], {z31.h, z0.h, z1.h, z2.h}, z15.h  // 11000001-01111111-01101011-11111011
// CHECK-INST: umlsl   za.s[w11, 6:7, vgx4], { z31.h, z0.h, z1.h, z2.h }, z15.h
// CHECK-ENCODING: [0xfb,0x6b,0x7f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17f6bfb <unknown>

umlsl   za.s[w8, 2:3, vgx4], {z17.h - z20.h}, z0.h  // 11000001-01110000-00001010-00111001
// CHECK-INST: umlsl   za.s[w8, 2:3, vgx4], { z17.h - z20.h }, z0.h
// CHECK-ENCODING: [0x39,0x0a,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1700a39 <unknown>

umlsl   za.s[w8, 2:3], {z17.h - z20.h}, z0.h  // 11000001-01110000-00001010-00111001
// CHECK-INST: umlsl   za.s[w8, 2:3, vgx4], { z17.h - z20.h }, z0.h
// CHECK-ENCODING: [0x39,0x0a,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1700a39 <unknown>

umlsl   za.s[w8, 2:3, vgx4], {z1.h - z4.h}, z14.h  // 11000001-01111110-00001000-00111001
// CHECK-INST: umlsl   za.s[w8, 2:3, vgx4], { z1.h - z4.h }, z14.h
// CHECK-ENCODING: [0x39,0x08,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17e0839 <unknown>

umlsl   za.s[w8, 2:3], {z1.h - z4.h}, z14.h  // 11000001-01111110-00001000-00111001
// CHECK-INST: umlsl   za.s[w8, 2:3, vgx4], { z1.h - z4.h }, z14.h
// CHECK-ENCODING: [0x39,0x08,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17e0839 <unknown>

umlsl   za.s[w10, 0:1, vgx4], {z19.h - z22.h}, z4.h  // 11000001-01110100-01001010-01111000
// CHECK-INST: umlsl   za.s[w10, 0:1, vgx4], { z19.h - z22.h }, z4.h
// CHECK-ENCODING: [0x78,0x4a,0x74,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1744a78 <unknown>

umlsl   za.s[w10, 0:1], {z19.h - z22.h}, z4.h  // 11000001-01110100-01001010-01111000
// CHECK-INST: umlsl   za.s[w10, 0:1, vgx4], { z19.h - z22.h }, z4.h
// CHECK-ENCODING: [0x78,0x4a,0x74,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1744a78 <unknown>

umlsl   za.s[w8, 0:1, vgx4], {z12.h - z15.h}, z2.h  // 11000001-01110010-00001001-10011000
// CHECK-INST: umlsl   za.s[w8, 0:1, vgx4], { z12.h - z15.h }, z2.h
// CHECK-ENCODING: [0x98,0x09,0x72,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1720998 <unknown>

umlsl   za.s[w8, 0:1], {z12.h - z15.h}, z2.h  // 11000001-01110010-00001001-10011000
// CHECK-INST: umlsl   za.s[w8, 0:1, vgx4], { z12.h - z15.h }, z2.h
// CHECK-ENCODING: [0x98,0x09,0x72,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1720998 <unknown>

umlsl   za.s[w10, 2:3, vgx4], {z1.h - z4.h}, z10.h  // 11000001-01111010-01001000-00111001
// CHECK-INST: umlsl   za.s[w10, 2:3, vgx4], { z1.h - z4.h }, z10.h
// CHECK-ENCODING: [0x39,0x48,0x7a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17a4839 <unknown>

umlsl   za.s[w10, 2:3], {z1.h - z4.h}, z10.h  // 11000001-01111010-01001000-00111001
// CHECK-INST: umlsl   za.s[w10, 2:3, vgx4], { z1.h - z4.h }, z10.h
// CHECK-ENCODING: [0x39,0x48,0x7a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17a4839 <unknown>

umlsl   za.s[w8, 2:3, vgx4], {z22.h - z25.h}, z14.h  // 11000001-01111110-00001010-11011001
// CHECK-INST: umlsl   za.s[w8, 2:3, vgx4], { z22.h - z25.h }, z14.h
// CHECK-ENCODING: [0xd9,0x0a,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17e0ad9 <unknown>

umlsl   za.s[w8, 2:3], {z22.h - z25.h}, z14.h  // 11000001-01111110-00001010-11011001
// CHECK-INST: umlsl   za.s[w8, 2:3, vgx4], { z22.h - z25.h }, z14.h
// CHECK-ENCODING: [0xd9,0x0a,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17e0ad9 <unknown>

umlsl   za.s[w11, 4:5, vgx4], {z9.h - z12.h}, z1.h  // 11000001-01110001-01101001-00111010
// CHECK-INST: umlsl   za.s[w11, 4:5, vgx4], { z9.h - z12.h }, z1.h
// CHECK-ENCODING: [0x3a,0x69,0x71,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c171693a <unknown>

umlsl   za.s[w11, 4:5], {z9.h - z12.h}, z1.h  // 11000001-01110001-01101001-00111010
// CHECK-INST: umlsl   za.s[w11, 4:5, vgx4], { z9.h - z12.h }, z1.h
// CHECK-ENCODING: [0x3a,0x69,0x71,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c171693a <unknown>

umlsl   za.s[w9, 6:7, vgx4], {z12.h - z15.h}, z11.h  // 11000001-01111011-00101001-10011011
// CHECK-INST: umlsl   za.s[w9, 6:7, vgx4], { z12.h - z15.h }, z11.h
// CHECK-ENCODING: [0x9b,0x29,0x7b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17b299b <unknown>

umlsl   za.s[w9, 6:7], {z12.h - z15.h}, z11.h  // 11000001-01111011-00101001-10011011
// CHECK-INST: umlsl   za.s[w9, 6:7, vgx4], { z12.h - z15.h }, z11.h
// CHECK-ENCODING: [0x9b,0x29,0x7b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17b299b <unknown>


umlsl   za.s[w8, 0:1, vgx4], {z0.h - z3.h}, z0.h[0]  // 11000001-11010000-10010000-00011000
// CHECK-INST: umlsl   za.s[w8, 0:1, vgx4], { z0.h - z3.h }, z0.h[0]
// CHECK-ENCODING: [0x18,0x90,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d09018 <unknown>

umlsl   za.s[w8, 0:1], {z0.h - z3.h}, z0.h[0]  // 11000001-11010000-10010000-00011000
// CHECK-INST: umlsl   za.s[w8, 0:1, vgx4], { z0.h - z3.h }, z0.h[0]
// CHECK-ENCODING: [0x18,0x90,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d09018 <unknown>

umlsl   za.s[w10, 2:3, vgx4], {z8.h - z11.h}, z5.h[3]  // 11000001-11010101-11010101-00011101
// CHECK-INST: umlsl   za.s[w10, 2:3, vgx4], { z8.h - z11.h }, z5.h[3]
// CHECK-ENCODING: [0x1d,0xd5,0xd5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d5d51d <unknown>

umlsl   za.s[w10, 2:3], {z8.h - z11.h}, z5.h[3]  // 11000001-11010101-11010101-00011101
// CHECK-INST: umlsl   za.s[w10, 2:3, vgx4], { z8.h - z11.h }, z5.h[3]
// CHECK-ENCODING: [0x1d,0xd5,0xd5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d5d51d <unknown>

umlsl   za.s[w11, 6:7, vgx4], {z12.h - z15.h}, z8.h[7]  // 11000001-11011000-11111101-10011111
// CHECK-INST: umlsl   za.s[w11, 6:7, vgx4], { z12.h - z15.h }, z8.h[7]
// CHECK-ENCODING: [0x9f,0xfd,0xd8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d8fd9f <unknown>

umlsl   za.s[w11, 6:7], {z12.h - z15.h}, z8.h[7]  // 11000001-11011000-11111101-10011111
// CHECK-INST: umlsl   za.s[w11, 6:7, vgx4], { z12.h - z15.h }, z8.h[7]
// CHECK-ENCODING: [0x9f,0xfd,0xd8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d8fd9f <unknown>

umlsl   za.s[w11, 6:7, vgx4], {z28.h - z31.h}, z15.h[7]  // 11000001-11011111-11111111-10011111
// CHECK-INST: umlsl   za.s[w11, 6:7, vgx4], { z28.h - z31.h }, z15.h[7]
// CHECK-ENCODING: [0x9f,0xff,0xdf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dfff9f <unknown>

umlsl   za.s[w11, 6:7], {z28.h - z31.h}, z15.h[7]  // 11000001-11011111-11111111-10011111
// CHECK-INST: umlsl   za.s[w11, 6:7, vgx4], { z28.h - z31.h }, z15.h[7]
// CHECK-ENCODING: [0x9f,0xff,0xdf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dfff9f <unknown>

umlsl   za.s[w8, 2:3, vgx4], {z16.h - z19.h}, z0.h[7]  // 11000001-11010000-10011110-00011101
// CHECK-INST: umlsl   za.s[w8, 2:3, vgx4], { z16.h - z19.h }, z0.h[7]
// CHECK-ENCODING: [0x1d,0x9e,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d09e1d <unknown>

umlsl   za.s[w8, 2:3], {z16.h - z19.h}, z0.h[7]  // 11000001-11010000-10011110-00011101
// CHECK-INST: umlsl   za.s[w8, 2:3, vgx4], { z16.h - z19.h }, z0.h[7]
// CHECK-ENCODING: [0x1d,0x9e,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d09e1d <unknown>

umlsl   za.s[w8, 2:3, vgx4], {z0.h - z3.h}, z14.h[2]  // 11000001-11011110-10010100-00011001
// CHECK-INST: umlsl   za.s[w8, 2:3, vgx4], { z0.h - z3.h }, z14.h[2]
// CHECK-ENCODING: [0x19,0x94,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de9419 <unknown>

umlsl   za.s[w8, 2:3], {z0.h - z3.h}, z14.h[2]  // 11000001-11011110-10010100-00011001
// CHECK-INST: umlsl   za.s[w8, 2:3, vgx4], { z0.h - z3.h }, z14.h[2]
// CHECK-ENCODING: [0x19,0x94,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de9419 <unknown>

umlsl   za.s[w10, 0:1, vgx4], {z16.h - z19.h}, z4.h[2]  // 11000001-11010100-11010110-00011000
// CHECK-INST: umlsl   za.s[w10, 0:1, vgx4], { z16.h - z19.h }, z4.h[2]
// CHECK-ENCODING: [0x18,0xd6,0xd4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d4d618 <unknown>

umlsl   za.s[w10, 0:1], {z16.h - z19.h}, z4.h[2]  // 11000001-11010100-11010110-00011000
// CHECK-INST: umlsl   za.s[w10, 0:1, vgx4], { z16.h - z19.h }, z4.h[2]
// CHECK-ENCODING: [0x18,0xd6,0xd4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d4d618 <unknown>

umlsl   za.s[w8, 0:1, vgx4], {z12.h - z15.h}, z2.h[4]  // 11000001-11010010-10011001-10011000
// CHECK-INST: umlsl   za.s[w8, 0:1, vgx4], { z12.h - z15.h }, z2.h[4]
// CHECK-ENCODING: [0x98,0x99,0xd2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d29998 <unknown>

umlsl   za.s[w8, 0:1], {z12.h - z15.h}, z2.h[4]  // 11000001-11010010-10011001-10011000
// CHECK-INST: umlsl   za.s[w8, 0:1, vgx4], { z12.h - z15.h }, z2.h[4]
// CHECK-ENCODING: [0x98,0x99,0xd2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d29998 <unknown>

umlsl   za.s[w10, 2:3, vgx4], {z0.h - z3.h}, z10.h[4]  // 11000001-11011010-11011000-00011001
// CHECK-INST: umlsl   za.s[w10, 2:3, vgx4], { z0.h - z3.h }, z10.h[4]
// CHECK-ENCODING: [0x19,0xd8,0xda,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dad819 <unknown>

umlsl   za.s[w10, 2:3], {z0.h - z3.h}, z10.h[4]  // 11000001-11011010-11011000-00011001
// CHECK-INST: umlsl   za.s[w10, 2:3, vgx4], { z0.h - z3.h }, z10.h[4]
// CHECK-ENCODING: [0x19,0xd8,0xda,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dad819 <unknown>

umlsl   za.s[w8, 2:3, vgx4], {z20.h - z23.h}, z14.h[5]  // 11000001-11011110-10011010-10011101
// CHECK-INST: umlsl   za.s[w8, 2:3, vgx4], { z20.h - z23.h }, z14.h[5]
// CHECK-ENCODING: [0x9d,0x9a,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de9a9d <unknown>

umlsl   za.s[w8, 2:3], {z20.h - z23.h}, z14.h[5]  // 11000001-11011110-10011010-10011101
// CHECK-INST: umlsl   za.s[w8, 2:3, vgx4], { z20.h - z23.h }, z14.h[5]
// CHECK-ENCODING: [0x9d,0x9a,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de9a9d <unknown>

umlsl   za.s[w11, 4:5, vgx4], {z8.h - z11.h}, z1.h[2]  // 11000001-11010001-11110101-00011010
// CHECK-INST: umlsl   za.s[w11, 4:5, vgx4], { z8.h - z11.h }, z1.h[2]
// CHECK-ENCODING: [0x1a,0xf5,0xd1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d1f51a <unknown>

umlsl   za.s[w11, 4:5], {z8.h - z11.h}, z1.h[2]  // 11000001-11010001-11110101-00011010
// CHECK-INST: umlsl   za.s[w11, 4:5, vgx4], { z8.h - z11.h }, z1.h[2]
// CHECK-ENCODING: [0x1a,0xf5,0xd1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d1f51a <unknown>

umlsl   za.s[w9, 6:7, vgx4], {z12.h - z15.h}, z11.h[5]  // 11000001-11011011-10111001-10011111
// CHECK-INST: umlsl   za.s[w9, 6:7, vgx4], { z12.h - z15.h }, z11.h[5]
// CHECK-ENCODING: [0x9f,0xb9,0xdb,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dbb99f <unknown>

umlsl   za.s[w9, 6:7], {z12.h - z15.h}, z11.h[5]  // 11000001-11011011-10111001-10011111
// CHECK-INST: umlsl   za.s[w9, 6:7, vgx4], { z12.h - z15.h }, z11.h[5]
// CHECK-ENCODING: [0x9f,0xb9,0xdb,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dbb99f <unknown>


umlsl   za.s[w8, 0:1, vgx4], {z0.h - z3.h}, {z0.h - z3.h}  // 11000001-11100001-00001000-00011000
// CHECK-INST: umlsl   za.s[w8, 0:1, vgx4], { z0.h - z3.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x18,0x08,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e10818 <unknown>

umlsl   za.s[w8, 0:1], {z0.h - z3.h}, {z0.h - z3.h}  // 11000001-11100001-00001000-00011000
// CHECK-INST: umlsl   za.s[w8, 0:1, vgx4], { z0.h - z3.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x18,0x08,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e10818 <unknown>

umlsl   za.s[w10, 2:3, vgx4], {z8.h - z11.h}, {z20.h - z23.h}  // 11000001-11110101-01001001-00011001
// CHECK-INST: umlsl   za.s[w10, 2:3, vgx4], { z8.h - z11.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x19,0x49,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f54919 <unknown>

umlsl   za.s[w10, 2:3], {z8.h - z11.h}, {z20.h - z23.h}  // 11000001-11110101-01001001-00011001
// CHECK-INST: umlsl   za.s[w10, 2:3, vgx4], { z8.h - z11.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x19,0x49,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f54919 <unknown>

umlsl   za.s[w11, 6:7, vgx4], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-11101001-01101001-10011011
// CHECK-INST: umlsl   za.s[w11, 6:7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x9b,0x69,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e9699b <unknown>

umlsl   za.s[w11, 6:7], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-11101001-01101001-10011011
// CHECK-INST: umlsl   za.s[w11, 6:7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x9b,0x69,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e9699b <unknown>

umlsl   za.s[w11, 6:7, vgx4], {z28.h - z31.h}, {z28.h - z31.h}  // 11000001-11111101-01101011-10011011
// CHECK-INST: umlsl   za.s[w11, 6:7, vgx4], { z28.h - z31.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x9b,0x6b,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd6b9b <unknown>

umlsl   za.s[w11, 6:7], {z28.h - z31.h}, {z28.h - z31.h}  // 11000001-11111101-01101011-10011011
// CHECK-INST: umlsl   za.s[w11, 6:7, vgx4], { z28.h - z31.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x9b,0x6b,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd6b9b <unknown>

umlsl   za.s[w8, 2:3, vgx4], {z16.h - z19.h}, {z16.h - z19.h}  // 11000001-11110001-00001010-00011001
// CHECK-INST: umlsl   za.s[w8, 2:3, vgx4], { z16.h - z19.h }, { z16.h - z19.h }
// CHECK-ENCODING: [0x19,0x0a,0xf1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f10a19 <unknown>

umlsl   za.s[w8, 2:3], {z16.h - z19.h}, {z16.h - z19.h}  // 11000001-11110001-00001010-00011001
// CHECK-INST: umlsl   za.s[w8, 2:3, vgx4], { z16.h - z19.h }, { z16.h - z19.h }
// CHECK-ENCODING: [0x19,0x0a,0xf1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f10a19 <unknown>

umlsl   za.s[w8, 2:3, vgx4], {z0.h - z3.h}, {z28.h - z31.h}  // 11000001-11111101-00001000-00011001
// CHECK-INST: umlsl   za.s[w8, 2:3, vgx4], { z0.h - z3.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x19,0x08,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd0819 <unknown>

umlsl   za.s[w8, 2:3], {z0.h - z3.h}, {z28.h - z31.h}  // 11000001-11111101-00001000-00011001
// CHECK-INST: umlsl   za.s[w8, 2:3, vgx4], { z0.h - z3.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x19,0x08,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd0819 <unknown>

umlsl   za.s[w10, 0:1, vgx4], {z16.h - z19.h}, {z20.h - z23.h}  // 11000001-11110101-01001010-00011000
// CHECK-INST: umlsl   za.s[w10, 0:1, vgx4], { z16.h - z19.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x18,0x4a,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f54a18 <unknown>

umlsl   za.s[w10, 0:1], {z16.h - z19.h}, {z20.h - z23.h}  // 11000001-11110101-01001010-00011000
// CHECK-INST: umlsl   za.s[w10, 0:1, vgx4], { z16.h - z19.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x18,0x4a,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f54a18 <unknown>

umlsl   za.s[w8, 0:1, vgx4], {z12.h - z15.h}, {z0.h - z3.h}  // 11000001-11100001-00001001-10011000
// CHECK-INST: umlsl   za.s[w8, 0:1, vgx4], { z12.h - z15.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x98,0x09,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e10998 <unknown>

umlsl   za.s[w8, 0:1], {z12.h - z15.h}, {z0.h - z3.h}  // 11000001-11100001-00001001-10011000
// CHECK-INST: umlsl   za.s[w8, 0:1, vgx4], { z12.h - z15.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x98,0x09,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e10998 <unknown>

umlsl   za.s[w10, 2:3, vgx4], {z0.h - z3.h}, {z24.h - z27.h}  // 11000001-11111001-01001000-00011001
// CHECK-INST: umlsl   za.s[w10, 2:3, vgx4], { z0.h - z3.h }, { z24.h - z27.h }
// CHECK-ENCODING: [0x19,0x48,0xf9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f94819 <unknown>

umlsl   za.s[w10, 2:3], {z0.h - z3.h}, {z24.h - z27.h}  // 11000001-11111001-01001000-00011001
// CHECK-INST: umlsl   za.s[w10, 2:3, vgx4], { z0.h - z3.h }, { z24.h - z27.h }
// CHECK-ENCODING: [0x19,0x48,0xf9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f94819 <unknown>

umlsl   za.s[w8, 2:3, vgx4], {z20.h - z23.h}, {z28.h - z31.h}  // 11000001-11111101-00001010-10011001
// CHECK-INST: umlsl   za.s[w8, 2:3, vgx4], { z20.h - z23.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x99,0x0a,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd0a99 <unknown>

umlsl   za.s[w8, 2:3], {z20.h - z23.h}, {z28.h - z31.h}  // 11000001-11111101-00001010-10011001
// CHECK-INST: umlsl   za.s[w8, 2:3, vgx4], { z20.h - z23.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x99,0x0a,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd0a99 <unknown>

umlsl   za.s[w11, 4:5, vgx4], {z8.h - z11.h}, {z0.h - z3.h}  // 11000001-11100001-01101001-00011010
// CHECK-INST: umlsl   za.s[w11, 4:5, vgx4], { z8.h - z11.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x1a,0x69,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e1691a <unknown>

umlsl   za.s[w11, 4:5], {z8.h - z11.h}, {z0.h - z3.h}  // 11000001-11100001-01101001-00011010
// CHECK-INST: umlsl   za.s[w11, 4:5, vgx4], { z8.h - z11.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x1a,0x69,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e1691a <unknown>

umlsl   za.s[w9, 6:7, vgx4], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-11101001-00101001-10011011
// CHECK-INST: umlsl   za.s[w9, 6:7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x9b,0x29,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e9299b <unknown>

umlsl   za.s[w9, 6:7], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-11101001-00101001-10011011
// CHECK-INST: umlsl   za.s[w9, 6:7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x9b,0x29,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e9299b <unknown>

