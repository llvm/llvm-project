// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump --no-print-imm-hex -d --mattr=+sme2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump --no-print-imm-hex -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


umlal   za.s[w8, 0:1], z0.h, z0.h  // 11000001-01100000-00001100-00010000
// CHECK-INST: umlal   za.s[w8, 0:1], z0.h, z0.h
// CHECK-ENCODING: [0x10,0x0c,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1600c10 <unknown>

umlal   za.s[w10, 10:11], z10.h, z5.h  // 11000001-01100101-01001101-01010101
// CHECK-INST: umlal   za.s[w10, 10:11], z10.h, z5.h
// CHECK-ENCODING: [0x55,0x4d,0x65,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1654d55 <unknown>

umlal   za.s[w11, 14:15], z13.h, z8.h  // 11000001-01101000-01101101-10110111
// CHECK-INST: umlal   za.s[w11, 14:15], z13.h, z8.h
// CHECK-ENCODING: [0xb7,0x6d,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1686db7 <unknown>

umlal   za.s[w11, 14:15], z31.h, z15.h  // 11000001-01101111-01101111-11110111
// CHECK-INST: umlal   za.s[w11, 14:15], z31.h, z15.h
// CHECK-ENCODING: [0xf7,0x6f,0x6f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16f6ff7 <unknown>

umlal   za.s[w8, 10:11], z17.h, z0.h  // 11000001-01100000-00001110-00110101
// CHECK-INST: umlal   za.s[w8, 10:11], z17.h, z0.h
// CHECK-ENCODING: [0x35,0x0e,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1600e35 <unknown>

umlal   za.s[w8, 2:3], z1.h, z14.h  // 11000001-01101110-00001100-00110001
// CHECK-INST: umlal   za.s[w8, 2:3], z1.h, z14.h
// CHECK-ENCODING: [0x31,0x0c,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e0c31 <unknown>

umlal   za.s[w10, 0:1], z19.h, z4.h  // 11000001-01100100-01001110-01110000
// CHECK-INST: umlal   za.s[w10, 0:1], z19.h, z4.h
// CHECK-ENCODING: [0x70,0x4e,0x64,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1644e70 <unknown>

umlal   za.s[w8, 0:1], z12.h, z2.h  // 11000001-01100010-00001101-10010000
// CHECK-INST: umlal   za.s[w8, 0:1], z12.h, z2.h
// CHECK-ENCODING: [0x90,0x0d,0x62,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1620d90 <unknown>

umlal   za.s[w10, 2:3], z1.h, z10.h  // 11000001-01101010-01001100-00110001
// CHECK-INST: umlal   za.s[w10, 2:3], z1.h, z10.h
// CHECK-ENCODING: [0x31,0x4c,0x6a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16a4c31 <unknown>

umlal   za.s[w8, 10:11], z22.h, z14.h  // 11000001-01101110-00001110-11010101
// CHECK-INST: umlal   za.s[w8, 10:11], z22.h, z14.h
// CHECK-ENCODING: [0xd5,0x0e,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e0ed5 <unknown>

umlal   za.s[w11, 4:5], z9.h, z1.h  // 11000001-01100001-01101101-00110010
// CHECK-INST: umlal   za.s[w11, 4:5], z9.h, z1.h
// CHECK-ENCODING: [0x32,0x6d,0x61,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1616d32 <unknown>

umlal   za.s[w9, 14:15], z12.h, z11.h  // 11000001-01101011-00101101-10010111
// CHECK-INST: umlal   za.s[w9, 14:15], z12.h, z11.h
// CHECK-ENCODING: [0x97,0x2d,0x6b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16b2d97 <unknown>


umlal   za.s[w8, 0:1], z0.h, z0.h[0]  // 11000001-11000000-00010000-00010000
// CHECK-INST: umlal   za.s[w8, 0:1], z0.h, z0.h[0]
// CHECK-ENCODING: [0x10,0x10,0xc0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1c01010 <unknown>

umlal   za.s[w10, 10:11], z10.h, z5.h[1]  // 11000001-11000101-01010101-01010101
// CHECK-INST: umlal   za.s[w10, 10:11], z10.h, z5.h[1]
// CHECK-ENCODING: [0x55,0x55,0xc5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1c55555 <unknown>

umlal   za.s[w11, 14:15], z13.h, z8.h[7]  // 11000001-11001000-11111101-10110111
// CHECK-INST: umlal   za.s[w11, 14:15], z13.h, z8.h[7]
// CHECK-ENCODING: [0xb7,0xfd,0xc8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1c8fdb7 <unknown>

umlal   za.s[w11, 14:15], z31.h, z15.h[7]  // 11000001-11001111-11111111-11110111
// CHECK-INST: umlal   za.s[w11, 14:15], z31.h, z15.h[7]
// CHECK-ENCODING: [0xf7,0xff,0xcf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1cffff7 <unknown>

umlal   za.s[w8, 10:11], z17.h, z0.h[3]  // 11000001-11000000-00011110-00110101
// CHECK-INST: umlal   za.s[w8, 10:11], z17.h, z0.h[3]
// CHECK-ENCODING: [0x35,0x1e,0xc0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1c01e35 <unknown>

umlal   za.s[w8, 2:3], z1.h, z14.h[5]  // 11000001-11001110-10010100-00110001
// CHECK-INST: umlal   za.s[w8, 2:3], z1.h, z14.h[5]
// CHECK-ENCODING: [0x31,0x94,0xce,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ce9431 <unknown>

umlal   za.s[w10, 0:1], z19.h, z4.h[1]  // 11000001-11000100-01010110-01110000
// CHECK-INST: umlal   za.s[w10, 0:1], z19.h, z4.h[1]
// CHECK-ENCODING: [0x70,0x56,0xc4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1c45670 <unknown>

umlal   za.s[w8, 0:1], z12.h, z2.h[2]  // 11000001-11000010-00011001-10010000
// CHECK-INST: umlal   za.s[w8, 0:1], z12.h, z2.h[2]
// CHECK-ENCODING: [0x90,0x19,0xc2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1c21990 <unknown>

umlal   za.s[w10, 2:3], z1.h, z10.h[6]  // 11000001-11001010-11011000-00110001
// CHECK-INST: umlal   za.s[w10, 2:3], z1.h, z10.h[6]
// CHECK-ENCODING: [0x31,0xd8,0xca,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1cad831 <unknown>

umlal   za.s[w8, 10:11], z22.h, z14.h[2]  // 11000001-11001110-00011010-11010101
// CHECK-INST: umlal   za.s[w8, 10:11], z22.h, z14.h[2]
// CHECK-ENCODING: [0xd5,0x1a,0xce,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ce1ad5 <unknown>

umlal   za.s[w11, 4:5], z9.h, z1.h[5]  // 11000001-11000001-11110101-00110010
// CHECK-INST: umlal   za.s[w11, 4:5], z9.h, z1.h[5]
// CHECK-ENCODING: [0x32,0xf5,0xc1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1c1f532 <unknown>

umlal   za.s[w9, 14:15], z12.h, z11.h[6]  // 11000001-11001011-10111001-10010111
// CHECK-INST: umlal   za.s[w9, 14:15], z12.h, z11.h[6]
// CHECK-ENCODING: [0x97,0xb9,0xcb,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1cbb997 <unknown>


umlal   za.s[w8, 0:1, vgx2], {z0.h, z1.h}, z0.h  // 11000001, 01100000, 00001000, 00010000
// CHECK, INST: umlal   za.s[w8, 0:1, vgx2], { z0.h, z1.h }, z0.h
// CHECK-ENCODING: [0x10,0x08,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1600810 <unknown>

umlal   za.s[w8, 0:1], {z0.h - z1.h}, z0.h  // 11000001-01100000-00001000-00010000
// CHECK, INST: umlal   za.s[w8, 0:1, vgx2], { z0.h, z1.h }, z0.h
// CHECK-ENCODING: [0x10,0x08,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1600810 <unknown>

umlal   za.s[w10, 2:3, vgx2], {z10.h, z11.h}, z5.h  // 11000001, 01100101, 01001001, 01010001
// CHECK, INST: umlal   za.s[w10, 2:3, vgx2], { z10.h, z11.h }, z5.h
// CHECK-ENCODING: [0x51,0x49,0x65,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1654951 <unknown>

umlal   za.s[w10, 2:3], {z10.h - z11.h}, z5.h  // 11000001-01100101-01001001-01010001
// CHECK, INST: umlal   za.s[w10, 2:3, vgx2], { z10.h, z11.h }, z5.h
// CHECK-ENCODING: [0x51,0x49,0x65,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1654951 <unknown>

umlal   za.s[w11, 6:7, vgx2], {z13.h, z14.h}, z8.h  // 11000001, 01101000, 01101001, 10110011
// CHECK, INST: umlal   za.s[w11, 6:7, vgx2], { z13.h, z14.h }, z8.h
// CHECK-ENCODING: [0xb3,0x69,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16869b3 <unknown>

umlal   za.s[w11, 6:7], {z13.h - z14.h}, z8.h  // 11000001-01101000-01101001-10110011
// CHECK, INST: umlal   za.s[w11, 6:7, vgx2], { z13.h, z14.h }, z8.h
// CHECK-ENCODING: [0xb3,0x69,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16869b3 <unknown>

umlal   za.s[w11, 6:7, vgx2], {z31.h, z0.h}, z15.h  // 11000001, 01101111, 01101011, 11110011
// CHECK, INST: umlal   za.s[w11, 6:7, vgx2], { z31.h, z0.h }, z15.h
// CHECK-ENCODING: [0xf3,0x6b,0x6f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16f6bf3 <unknown>

umlal   za.s[w11, 6:7], {z31.h - z0.h}, z15.h  // 11000001-01101111-01101011-11110011
// CHECK, INST: umlal   za.s[w11, 6:7, vgx2], { z31.h, z0.h }, z15.h
// CHECK-ENCODING: [0xf3,0x6b,0x6f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16f6bf3 <unknown>

umlal   za.s[w8, 2:3, vgx2], {z17.h, z18.h}, z0.h  // 11000001, 01100000, 00001010, 00110001
// CHECK, INST: umlal   za.s[w8, 2:3, vgx2], { z17.h, z18.h }, z0.h
// CHECK-ENCODING: [0x31,0x0a,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1600a31 <unknown>

umlal   za.s[w8, 2:3], {z17.h - z18.h}, z0.h  // 11000001-01100000-00001010-00110001
// CHECK, INST: umlal   za.s[w8, 2:3, vgx2], { z17.h, z18.h }, z0.h
// CHECK-ENCODING: [0x31,0x0a,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1600a31 <unknown>

umlal   za.s[w8, 2:3, vgx2], {z1.h, z2.h}, z14.h  // 11000001, 01101110, 00001000, 00110001
// CHECK, INST: umlal   za.s[w8, 2:3, vgx2], { z1.h, z2.h }, z14.h
// CHECK-ENCODING: [0x31,0x08,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e0831 <unknown>

umlal   za.s[w8, 2:3], {z1.h - z2.h}, z14.h  // 11000001-01101110-00001000-00110001
// CHECK, INST: umlal   za.s[w8, 2:3, vgx2], { z1.h, z2.h }, z14.h
// CHECK-ENCODING: [0x31,0x08,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e0831 <unknown>

umlal   za.s[w10, 0:1, vgx2], {z19.h, z20.h}, z4.h  // 11000001, 01100100, 01001010, 01110000
// CHECK, INST: umlal   za.s[w10, 0:1, vgx2], { z19.h, z20.h }, z4.h
// CHECK-ENCODING: [0x70,0x4a,0x64,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1644a70 <unknown>

umlal   za.s[w10, 0:1], {z19.h - z20.h}, z4.h  // 11000001-01100100-01001010-01110000
// CHECK, INST: umlal   za.s[w10, 0:1, vgx2], { z19.h, z20.h }, z4.h
// CHECK-ENCODING: [0x70,0x4a,0x64,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1644a70 <unknown>

umlal   za.s[w8, 0:1, vgx2], {z12.h, z13.h}, z2.h  // 11000001, 01100010, 00001001, 10010000
// CHECK, INST: umlal   za.s[w8, 0:1, vgx2], { z12.h, z13.h }, z2.h
// CHECK-ENCODING: [0x90,0x09,0x62,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1620990 <unknown>

umlal   za.s[w8, 0:1], {z12.h - z13.h}, z2.h  // 11000001-01100010-00001001-10010000
// CHECK, INST: umlal   za.s[w8, 0:1, vgx2], { z12.h, z13.h }, z2.h
// CHECK-ENCODING: [0x90,0x09,0x62,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1620990 <unknown>

umlal   za.s[w10, 2:3, vgx2], {z1.h, z2.h}, z10.h  // 11000001, 01101010, 01001000, 00110001
// CHECK, INST: umlal   za.s[w10, 2:3, vgx2], { z1.h, z2.h }, z10.h
// CHECK-ENCODING: [0x31,0x48,0x6a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16a4831 <unknown>

umlal   za.s[w10, 2:3], {z1.h - z2.h}, z10.h  // 11000001-01101010-01001000-00110001
// CHECK, INST: umlal   za.s[w10, 2:3, vgx2], { z1.h, z2.h }, z10.h
// CHECK-ENCODING: [0x31,0x48,0x6a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16a4831 <unknown>

umlal   za.s[w8, 2:3, vgx2], {z22.h, z23.h}, z14.h  // 11000001, 01101110, 00001010, 11010001
// CHECK, INST: umlal   za.s[w8, 2:3, vgx2], { z22.h, z23.h }, z14.h
// CHECK-ENCODING: [0xd1,0x0a,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e0ad1 <unknown>

umlal   za.s[w8, 2:3], {z22.h - z23.h}, z14.h  // 11000001-01101110-00001010-11010001
// CHECK, INST: umlal   za.s[w8, 2:3, vgx2], { z22.h, z23.h }, z14.h
// CHECK-ENCODING: [0xd1,0x0a,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e0ad1 <unknown>

umlal   za.s[w11, 4:5, vgx2], {z9.h, z10.h}, z1.h  // 11000001, 01100001, 01101001, 00110010
// CHECK, INST: umlal   za.s[w11, 4:5, vgx2], { z9.h, z10.h }, z1.h
// CHECK-ENCODING: [0x32,0x69,0x61,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1616932 <unknown>

umlal   za.s[w11, 4:5], {z9.h - z10.h}, z1.h  // 11000001-01100001-01101001-00110010
// CHECK, INST: umlal   za.s[w11, 4:5, vgx2], { z9.h, z10.h }, z1.h
// CHECK-ENCODING: [0x32,0x69,0x61,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1616932 <unknown>

umlal   za.s[w9, 6:7, vgx2], {z12.h, z13.h}, z11.h  // 11000001, 01101011, 00101001, 10010011
// CHECK, INST: umlal   za.s[w9, 6:7, vgx2], { z12.h, z13.h }, z11.h
// CHECK-ENCODING: [0x93,0x29,0x6b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16b2993 <unknown>

umlal   za.s[w9, 6:7], {z12.h - z13.h}, z11.h  // 11000001-01101011-00101001-10010011
// CHECK, INST: umlal   za.s[w9, 6:7, vgx2], { z12.h, z13.h }, z11.h
// CHECK-ENCODING: [0x93,0x29,0x6b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16b2993 <unknown>


umlal   za.s[w8, 0:1, vgx2], {z0.h, z1.h}, z0.h[0]  // 11000001, 11010000, 00010000, 00010000
// CHECK, INST: umlal   za.s[w8, 0:1, vgx2], { z0.h, z1.h }, z0.h[0]
// CHECK-ENCODING: [0x10,0x10,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d01010 <unknown>

umlal   za.s[w8, 0:1], {z0.h - z1.h}, z0.h[0]  // 11000001-11010000-00010000-00010000
// CHECK, INST: umlal   za.s[w8, 0:1, vgx2], { z0.h, z1.h }, z0.h[0]
// CHECK-ENCODING: [0x10,0x10,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d01010 <unknown>

umlal   za.s[w10, 2:3, vgx2], {z10.h, z11.h}, z5.h[3]  // 11000001, 11010101, 01010101, 01010101
// CHECK, INST: umlal   za.s[w10, 2:3, vgx2], { z10.h, z11.h }, z5.h[3]
// CHECK-ENCODING: [0x55,0x55,0xd5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d55555 <unknown>

umlal   za.s[w10, 2:3], {z10.h - z11.h}, z5.h[3]  // 11000001-11010101-01010101-01010101
// CHECK, INST: umlal   za.s[w10, 2:3, vgx2], { z10.h, z11.h }, z5.h[3]
// CHECK-ENCODING: [0x55,0x55,0xd5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d55555 <unknown>

umlal   za.s[w11, 6:7, vgx2], {z12.h, z13.h}, z8.h[7]  // 11000001, 11011000, 01111101, 10010111
// CHECK, INST: umlal   za.s[w11, 6:7, vgx2], { z12.h, z13.h }, z8.h[7]
// CHECK-ENCODING: [0x97,0x7d,0xd8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d87d97 <unknown>

umlal   za.s[w11, 6:7], {z12.h - z13.h}, z8.h[7]  // 11000001-11011000-01111101-10010111
// CHECK, INST: umlal   za.s[w11, 6:7, vgx2], { z12.h, z13.h }, z8.h[7]
// CHECK-ENCODING: [0x97,0x7d,0xd8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d87d97 <unknown>

umlal   za.s[w11, 6:7, vgx2], {z30.h, z31.h}, z15.h[7]  // 11000001, 11011111, 01111111, 11010111
// CHECK, INST: umlal   za.s[w11, 6:7, vgx2], { z30.h, z31.h }, z15.h[7]
// CHECK-ENCODING: [0xd7,0x7f,0xdf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1df7fd7 <unknown>

umlal   za.s[w11, 6:7], {z30.h - z31.h}, z15.h[7]  // 11000001-11011111-01111111-11010111
// CHECK, INST: umlal   za.s[w11, 6:7, vgx2], { z30.h, z31.h }, z15.h[7]
// CHECK-ENCODING: [0xd7,0x7f,0xdf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1df7fd7 <unknown>

umlal   za.s[w8, 2:3, vgx2], {z16.h, z17.h}, z0.h[7]  // 11000001, 11010000, 00011110, 00010101
// CHECK, INST: umlal   za.s[w8, 2:3, vgx2], { z16.h, z17.h }, z0.h[7]
// CHECK-ENCODING: [0x15,0x1e,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d01e15 <unknown>

umlal   za.s[w8, 2:3], {z16.h - z17.h}, z0.h[7]  // 11000001-11010000-00011110-00010101
// CHECK, INST: umlal   za.s[w8, 2:3, vgx2], { z16.h, z17.h }, z0.h[7]
// CHECK-ENCODING: [0x15,0x1e,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d01e15 <unknown>

umlal   za.s[w8, 2:3, vgx2], {z0.h, z1.h}, z14.h[2]  // 11000001, 11011110, 00010100, 00010001
// CHECK, INST: umlal   za.s[w8, 2:3, vgx2], { z0.h, z1.h }, z14.h[2]
// CHECK-ENCODING: [0x11,0x14,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de1411 <unknown>

umlal   za.s[w8, 2:3], {z0.h - z1.h}, z14.h[2]  // 11000001-11011110-00010100-00010001
// CHECK, INST: umlal   za.s[w8, 2:3, vgx2], { z0.h, z1.h }, z14.h[2]
// CHECK-ENCODING: [0x11,0x14,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de1411 <unknown>

umlal   za.s[w10, 0:1, vgx2], {z18.h, z19.h}, z4.h[2]  // 11000001, 11010100, 01010110, 01010000
// CHECK, INST: umlal   za.s[w10, 0:1, vgx2], { z18.h, z19.h }, z4.h[2]
// CHECK-ENCODING: [0x50,0x56,0xd4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d45650 <unknown>

umlal   za.s[w10, 0:1], {z18.h - z19.h}, z4.h[2]  // 11000001-11010100-01010110-01010000
// CHECK, INST: umlal   za.s[w10, 0:1, vgx2], { z18.h, z19.h }, z4.h[2]
// CHECK-ENCODING: [0x50,0x56,0xd4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d45650 <unknown>

umlal   za.s[w8, 0:1, vgx2], {z12.h, z13.h}, z2.h[4]  // 11000001, 11010010, 00011001, 10010000
// CHECK, INST: umlal   za.s[w8, 0:1, vgx2], { z12.h, z13.h }, z2.h[4]
// CHECK-ENCODING: [0x90,0x19,0xd2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d21990 <unknown>

umlal   za.s[w8, 0:1], {z12.h - z13.h}, z2.h[4]  // 11000001-11010010-00011001-10010000
// CHECK, INST: umlal   za.s[w8, 0:1, vgx2], { z12.h, z13.h }, z2.h[4]
// CHECK-ENCODING: [0x90,0x19,0xd2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d21990 <unknown>

umlal   za.s[w10, 2:3, vgx2], {z0.h, z1.h}, z10.h[4]  // 11000001, 11011010, 01011000, 00010001
// CHECK, INST: umlal   za.s[w10, 2:3, vgx2], { z0.h, z1.h }, z10.h[4]
// CHECK-ENCODING: [0x11,0x58,0xda,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1da5811 <unknown>

umlal   za.s[w10, 2:3], {z0.h - z1.h}, z10.h[4]  // 11000001-11011010-01011000-00010001
// CHECK, INST: umlal   za.s[w10, 2:3, vgx2], { z0.h, z1.h }, z10.h[4]
// CHECK-ENCODING: [0x11,0x58,0xda,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1da5811 <unknown>

umlal   za.s[w8, 2:3, vgx2], {z22.h, z23.h}, z14.h[5]  // 11000001, 11011110, 00011010, 11010101
// CHECK, INST: umlal   za.s[w8, 2:3, vgx2], { z22.h, z23.h }, z14.h[5]
// CHECK-ENCODING: [0xd5,0x1a,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de1ad5 <unknown>

umlal   za.s[w8, 2:3], {z22.h - z23.h}, z14.h[5]  // 11000001-11011110-00011010-11010101
// CHECK, INST: umlal   za.s[w8, 2:3, vgx2], { z22.h, z23.h }, z14.h[5]
// CHECK-ENCODING: [0xd5,0x1a,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de1ad5 <unknown>

umlal   za.s[w11, 4:5, vgx2], {z8.h, z9.h}, z1.h[2]  // 11000001, 11010001, 01110101, 00010010
// CHECK, INST: umlal   za.s[w11, 4:5, vgx2], { z8.h, z9.h }, z1.h[2]
// CHECK-ENCODING: [0x12,0x75,0xd1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d17512 <unknown>

umlal   za.s[w11, 4:5], {z8.h - z9.h}, z1.h[2]  // 11000001-11010001-01110101-00010010
// CHECK, INST: umlal   za.s[w11, 4:5, vgx2], { z8.h, z9.h }, z1.h[2]
// CHECK-ENCODING: [0x12,0x75,0xd1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d17512 <unknown>

umlal   za.s[w9, 6:7, vgx2], {z12.h, z13.h}, z11.h[5]  // 11000001, 11011011, 00111001, 10010111
// CHECK, INST: umlal   za.s[w9, 6:7, vgx2], { z12.h, z13.h }, z11.h[5]
// CHECK-ENCODING: [0x97,0x39,0xdb,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1db3997 <unknown>

umlal   za.s[w9, 6:7], {z12.h - z13.h}, z11.h[5]  // 11000001-11011011-00111001-10010111
// CHECK, INST: umlal   za.s[w9, 6:7, vgx2], { z12.h, z13.h }, z11.h[5]
// CHECK-ENCODING: [0x97,0x39,0xdb,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1db3997 <unknown>


umlal   za.s[w8, 0:1, vgx2], {z0.h, z1.h}, {z0.h, z1.h}  // 11000001, 11100000, 00001000, 00010000
// CHECK, INST: umlal   za.s[w8, 0:1, vgx2], { z0.h, z1.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x10,0x08,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e00810 <unknown>

umlal   za.s[w8, 0:1], {z0.h - z1.h}, {z0.h - z1.h}  // 11000001-11100000-00001000-00010000
// CHECK, INST: umlal   za.s[w8, 0:1, vgx2], { z0.h, z1.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x10,0x08,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e00810 <unknown>

umlal   za.s[w10, 2:3, vgx2], {z10.h, z11.h}, {z20.h, z21.h}  // 11000001, 11110100, 01001001, 01010001
// CHECK, INST: umlal   za.s[w10, 2:3, vgx2], { z10.h, z11.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x51,0x49,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f44951 <unknown>

umlal   za.s[w10, 2:3], {z10.h - z11.h}, {z20.h - z21.h}  // 11000001-11110100-01001001-01010001
// CHECK, INST: umlal   za.s[w10, 2:3, vgx2], { z10.h, z11.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x51,0x49,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f44951 <unknown>

umlal   za.s[w11, 6:7, vgx2], {z12.h, z13.h}, {z8.h, z9.h}  // 11000001, 11101000, 01101001, 10010011
// CHECK, INST: umlal   za.s[w11, 6:7, vgx2], { z12.h, z13.h }, { z8.h, z9.h }
// CHECK-ENCODING: [0x93,0x69,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e86993 <unknown>

umlal   za.s[w11, 6:7], {z12.h - z13.h}, {z8.h - z9.h}  // 11000001-11101000-01101001-10010011
// CHECK, INST: umlal   za.s[w11, 6:7, vgx2], { z12.h, z13.h }, { z8.h, z9.h }
// CHECK-ENCODING: [0x93,0x69,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e86993 <unknown>

umlal   za.s[w11, 6:7, vgx2], {z30.h, z31.h}, {z30.h, z31.h}  // 11000001, 11111110, 01101011, 11010011
// CHECK, INST: umlal   za.s[w11, 6:7, vgx2], { z30.h, z31.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xd3,0x6b,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe6bd3 <unknown>

umlal   za.s[w11, 6:7], {z30.h - z31.h}, {z30.h - z31.h}  // 11000001-11111110-01101011-11010011
// CHECK, INST: umlal   za.s[w11, 6:7, vgx2], { z30.h, z31.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xd3,0x6b,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe6bd3 <unknown>

umlal   za.s[w8, 2:3, vgx2], {z16.h, z17.h}, {z16.h, z17.h}  // 11000001, 11110000, 00001010, 00010001
// CHECK, INST: umlal   za.s[w8, 2:3, vgx2], { z16.h, z17.h }, { z16.h, z17.h }
// CHECK-ENCODING: [0x11,0x0a,0xf0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f00a11 <unknown>

umlal   za.s[w8, 2:3], {z16.h - z17.h}, {z16.h - z17.h}  // 11000001-11110000-00001010-00010001
// CHECK, INST: umlal   za.s[w8, 2:3, vgx2], { z16.h, z17.h }, { z16.h, z17.h }
// CHECK-ENCODING: [0x11,0x0a,0xf0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f00a11 <unknown>

umlal   za.s[w8, 2:3, vgx2], {z0.h, z1.h}, {z30.h, z31.h}  // 11000001, 11111110, 00001000, 00010001
// CHECK, INST: umlal   za.s[w8, 2:3, vgx2], { z0.h, z1.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0x11,0x08,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe0811 <unknown>

umlal   za.s[w8, 2:3], {z0.h - z1.h}, {z30.h - z31.h}  // 11000001-11111110-00001000-00010001
// CHECK, INST: umlal   za.s[w8, 2:3, vgx2], { z0.h, z1.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0x11,0x08,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe0811 <unknown>

umlal   za.s[w10, 0:1, vgx2], {z18.h, z19.h}, {z20.h, z21.h}  // 11000001, 11110100, 01001010, 01010000
// CHECK, INST: umlal   za.s[w10, 0:1, vgx2], { z18.h, z19.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x50,0x4a,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f44a50 <unknown>

umlal   za.s[w10, 0:1], {z18.h - z19.h}, {z20.h - z21.h}  // 11000001-11110100-01001010-01010000
// CHECK, INST: umlal   za.s[w10, 0:1, vgx2], { z18.h, z19.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x50,0x4a,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f44a50 <unknown>

umlal   za.s[w8, 0:1, vgx2], {z12.h, z13.h}, {z2.h, z3.h}  // 11000001, 11100010, 00001001, 10010000
// CHECK, INST: umlal   za.s[w8, 0:1, vgx2], { z12.h, z13.h }, { z2.h, z3.h }
// CHECK-ENCODING: [0x90,0x09,0xe2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e20990 <unknown>

umlal   za.s[w8, 0:1], {z12.h - z13.h}, {z2.h - z3.h}  // 11000001-11100010-00001001-10010000
// CHECK, INST: umlal   za.s[w8, 0:1, vgx2], { z12.h, z13.h }, { z2.h, z3.h }
// CHECK-ENCODING: [0x90,0x09,0xe2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e20990 <unknown>

umlal   za.s[w10, 2:3, vgx2], {z0.h, z1.h}, {z26.h, z27.h}  // 11000001, 11111010, 01001000, 00010001
// CHECK, INST: umlal   za.s[w10, 2:3, vgx2], { z0.h, z1.h }, { z26.h, z27.h }
// CHECK-ENCODING: [0x11,0x48,0xfa,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fa4811 <unknown>

umlal   za.s[w10, 2:3], {z0.h - z1.h}, {z26.h - z27.h}  // 11000001-11111010-01001000-00010001
// CHECK, INST: umlal   za.s[w10, 2:3, vgx2], { z0.h, z1.h }, { z26.h, z27.h }
// CHECK-ENCODING: [0x11,0x48,0xfa,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fa4811 <unknown>

umlal   za.s[w8, 2:3, vgx2], {z22.h, z23.h}, {z30.h, z31.h}  // 11000001, 11111110, 00001010, 11010001
// CHECK, INST: umlal   za.s[w8, 2:3, vgx2], { z22.h, z23.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xd1,0x0a,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe0ad1 <unknown>

umlal   za.s[w8, 2:3], {z22.h - z23.h}, {z30.h - z31.h}  // 11000001-11111110-00001010-11010001
// CHECK, INST: umlal   za.s[w8, 2:3, vgx2], { z22.h, z23.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xd1,0x0a,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe0ad1 <unknown>

umlal   za.s[w11, 4:5, vgx2], {z8.h, z9.h}, {z0.h, z1.h}  // 11000001, 11100000, 01101001, 00010010
// CHECK, INST: umlal   za.s[w11, 4:5, vgx2], { z8.h, z9.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x12,0x69,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e06912 <unknown>

umlal   za.s[w11, 4:5], {z8.h - z9.h}, {z0.h - z1.h}  // 11000001-11100000-01101001-00010010
// CHECK, INST: umlal   za.s[w11, 4:5, vgx2], { z8.h, z9.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x12,0x69,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e06912 <unknown>

umlal   za.s[w9, 6:7, vgx2], {z12.h, z13.h}, {z10.h, z11.h}  // 11000001, 11101010, 00101001, 10010011
// CHECK, INST: umlal   za.s[w9, 6:7, vgx2], { z12.h, z13.h }, { z10.h, z11.h }
// CHECK-ENCODING: [0x93,0x29,0xea,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ea2993 <unknown>

umlal   za.s[w9, 6:7], {z12.h - z13.h}, {z10.h - z11.h}  // 11000001-11101010-00101001-10010011
// CHECK, INST: umlal   za.s[w9, 6:7, vgx2], { z12.h, z13.h }, { z10.h, z11.h }
// CHECK-ENCODING: [0x93,0x29,0xea,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ea2993 <unknown>


umlal   za.s[w8, 0:1, vgx4], {z0.h - z3.h}, z0.h  // 11000001-01110000-00001000-00010000
// CHECK-INST: umlal   za.s[w8, 0:1, vgx4], { z0.h - z3.h }, z0.h
// CHECK-ENCODING: [0x10,0x08,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1700810 <unknown>

umlal   za.s[w8, 0:1], {z0.h - z3.h}, z0.h  // 11000001-01110000-00001000-00010000
// CHECK-INST: umlal   za.s[w8, 0:1, vgx4], { z0.h - z3.h }, z0.h
// CHECK-ENCODING: [0x10,0x08,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1700810 <unknown>

umlal   za.s[w10, 2:3, vgx4], {z10.h - z13.h}, z5.h  // 11000001-01110101-01001001-01010001
// CHECK-INST: umlal   za.s[w10, 2:3, vgx4], { z10.h - z13.h }, z5.h
// CHECK-ENCODING: [0x51,0x49,0x75,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1754951 <unknown>

umlal   za.s[w10, 2:3], {z10.h - z13.h}, z5.h  // 11000001-01110101-01001001-01010001
// CHECK-INST: umlal   za.s[w10, 2:3, vgx4], { z10.h - z13.h }, z5.h
// CHECK-ENCODING: [0x51,0x49,0x75,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1754951 <unknown>

umlal   za.s[w11, 6:7, vgx4], {z13.h - z16.h}, z8.h  // 11000001-01111000-01101001-10110011
// CHECK-INST: umlal   za.s[w11, 6:7, vgx4], { z13.h - z16.h }, z8.h
// CHECK-ENCODING: [0xb3,0x69,0x78,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17869b3 <unknown>

umlal   za.s[w11, 6:7], {z13.h - z16.h}, z8.h  // 11000001-01111000-01101001-10110011
// CHECK-INST: umlal   za.s[w11, 6:7, vgx4], { z13.h - z16.h }, z8.h
// CHECK-ENCODING: [0xb3,0x69,0x78,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17869b3 <unknown>

umlal   za.s[w11, 6:7, vgx4], {z31.h, z0.h, z1.h, z2.h}, z15.h  // 11000001-01111111-01101011-11110011
// CHECK-INST: umlal   za.s[w11, 6:7, vgx4], { z31.h, z0.h, z1.h, z2.h }, z15.h
// CHECK-ENCODING: [0xf3,0x6b,0x7f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17f6bf3 <unknown>

umlal   za.s[w11, 6:7], {z31.h, z0.h, z1.h, z2.h}, z15.h  // 11000001-01111111-01101011-11110011
// CHECK-INST: umlal   za.s[w11, 6:7, vgx4], { z31.h, z0.h, z1.h, z2.h }, z15.h
// CHECK-ENCODING: [0xf3,0x6b,0x7f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17f6bf3 <unknown>

umlal   za.s[w8, 2:3, vgx4], {z17.h - z20.h}, z0.h  // 11000001-01110000-00001010-00110001
// CHECK-INST: umlal   za.s[w8, 2:3, vgx4], { z17.h - z20.h }, z0.h
// CHECK-ENCODING: [0x31,0x0a,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1700a31 <unknown>

umlal   za.s[w8, 2:3], {z17.h - z20.h}, z0.h  // 11000001-01110000-00001010-00110001
// CHECK-INST: umlal   za.s[w8, 2:3, vgx4], { z17.h - z20.h }, z0.h
// CHECK-ENCODING: [0x31,0x0a,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1700a31 <unknown>

umlal   za.s[w8, 2:3, vgx4], {z1.h - z4.h}, z14.h  // 11000001-01111110-00001000-00110001
// CHECK-INST: umlal   za.s[w8, 2:3, vgx4], { z1.h - z4.h }, z14.h
// CHECK-ENCODING: [0x31,0x08,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17e0831 <unknown>

umlal   za.s[w8, 2:3], {z1.h - z4.h}, z14.h  // 11000001-01111110-00001000-00110001
// CHECK-INST: umlal   za.s[w8, 2:3, vgx4], { z1.h - z4.h }, z14.h
// CHECK-ENCODING: [0x31,0x08,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17e0831 <unknown>

umlal   za.s[w10, 0:1, vgx4], {z19.h - z22.h}, z4.h  // 11000001-01110100-01001010-01110000
// CHECK-INST: umlal   za.s[w10, 0:1, vgx4], { z19.h - z22.h }, z4.h
// CHECK-ENCODING: [0x70,0x4a,0x74,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1744a70 <unknown>

umlal   za.s[w10, 0:1], {z19.h - z22.h}, z4.h  // 11000001-01110100-01001010-01110000
// CHECK-INST: umlal   za.s[w10, 0:1, vgx4], { z19.h - z22.h }, z4.h
// CHECK-ENCODING: [0x70,0x4a,0x74,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1744a70 <unknown>

umlal   za.s[w8, 0:1, vgx4], {z12.h - z15.h}, z2.h  // 11000001-01110010-00001001-10010000
// CHECK-INST: umlal   za.s[w8, 0:1, vgx4], { z12.h - z15.h }, z2.h
// CHECK-ENCODING: [0x90,0x09,0x72,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1720990 <unknown>

umlal   za.s[w8, 0:1], {z12.h - z15.h}, z2.h  // 11000001-01110010-00001001-10010000
// CHECK-INST: umlal   za.s[w8, 0:1, vgx4], { z12.h - z15.h }, z2.h
// CHECK-ENCODING: [0x90,0x09,0x72,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1720990 <unknown>

umlal   za.s[w10, 2:3, vgx4], {z1.h - z4.h}, z10.h  // 11000001-01111010-01001000-00110001
// CHECK-INST: umlal   za.s[w10, 2:3, vgx4], { z1.h - z4.h }, z10.h
// CHECK-ENCODING: [0x31,0x48,0x7a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17a4831 <unknown>

umlal   za.s[w10, 2:3], {z1.h - z4.h}, z10.h  // 11000001-01111010-01001000-00110001
// CHECK-INST: umlal   za.s[w10, 2:3, vgx4], { z1.h - z4.h }, z10.h
// CHECK-ENCODING: [0x31,0x48,0x7a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17a4831 <unknown>

umlal   za.s[w8, 2:3, vgx4], {z22.h - z25.h}, z14.h  // 11000001-01111110-00001010-11010001
// CHECK-INST: umlal   za.s[w8, 2:3, vgx4], { z22.h - z25.h }, z14.h
// CHECK-ENCODING: [0xd1,0x0a,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17e0ad1 <unknown>

umlal   za.s[w8, 2:3], {z22.h - z25.h}, z14.h  // 11000001-01111110-00001010-11010001
// CHECK-INST: umlal   za.s[w8, 2:3, vgx4], { z22.h - z25.h }, z14.h
// CHECK-ENCODING: [0xd1,0x0a,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17e0ad1 <unknown>

umlal   za.s[w11, 4:5, vgx4], {z9.h - z12.h}, z1.h  // 11000001-01110001-01101001-00110010
// CHECK-INST: umlal   za.s[w11, 4:5, vgx4], { z9.h - z12.h }, z1.h
// CHECK-ENCODING: [0x32,0x69,0x71,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1716932 <unknown>

umlal   za.s[w11, 4:5], {z9.h - z12.h}, z1.h  // 11000001-01110001-01101001-00110010
// CHECK-INST: umlal   za.s[w11, 4:5, vgx4], { z9.h - z12.h }, z1.h
// CHECK-ENCODING: [0x32,0x69,0x71,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1716932 <unknown>

umlal   za.s[w9, 6:7, vgx4], {z12.h - z15.h}, z11.h  // 11000001-01111011-00101001-10010011
// CHECK-INST: umlal   za.s[w9, 6:7, vgx4], { z12.h - z15.h }, z11.h
// CHECK-ENCODING: [0x93,0x29,0x7b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17b2993 <unknown>

umlal   za.s[w9, 6:7], {z12.h - z15.h}, z11.h  // 11000001-01111011-00101001-10010011
// CHECK-INST: umlal   za.s[w9, 6:7, vgx4], { z12.h - z15.h }, z11.h
// CHECK-ENCODING: [0x93,0x29,0x7b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17b2993 <unknown>


umlal   za.s[w8, 0:1, vgx4], {z0.h - z3.h}, z0.h[0]  // 11000001-11010000-10010000-00010000
// CHECK-INST: umlal   za.s[w8, 0:1, vgx4], { z0.h - z3.h }, z0.h[0]
// CHECK-ENCODING: [0x10,0x90,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d09010 <unknown>

umlal   za.s[w8, 0:1], {z0.h - z3.h}, z0.h[0]  // 11000001-11010000-10010000-00010000
// CHECK-INST: umlal   za.s[w8, 0:1, vgx4], { z0.h - z3.h }, z0.h[0]
// CHECK-ENCODING: [0x10,0x90,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d09010 <unknown>

umlal   za.s[w10, 2:3, vgx4], {z8.h - z11.h}, z5.h[3]  // 11000001-11010101-11010101-00010101
// CHECK-INST: umlal   za.s[w10, 2:3, vgx4], { z8.h - z11.h }, z5.h[3]
// CHECK-ENCODING: [0x15,0xd5,0xd5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d5d515 <unknown>

umlal   za.s[w10, 2:3], {z8.h - z11.h}, z5.h[3]  // 11000001-11010101-11010101-00010101
// CHECK-INST: umlal   za.s[w10, 2:3, vgx4], { z8.h - z11.h }, z5.h[3]
// CHECK-ENCODING: [0x15,0xd5,0xd5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d5d515 <unknown>

umlal   za.s[w11, 6:7, vgx4], {z12.h - z15.h}, z8.h[7]  // 11000001-11011000-11111101-10010111
// CHECK-INST: umlal   za.s[w11, 6:7, vgx4], { z12.h - z15.h }, z8.h[7]
// CHECK-ENCODING: [0x97,0xfd,0xd8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d8fd97 <unknown>

umlal   za.s[w11, 6:7], {z12.h - z15.h}, z8.h[7]  // 11000001-11011000-11111101-10010111
// CHECK-INST: umlal   za.s[w11, 6:7, vgx4], { z12.h - z15.h }, z8.h[7]
// CHECK-ENCODING: [0x97,0xfd,0xd8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d8fd97 <unknown>

umlal   za.s[w11, 6:7, vgx4], {z28.h - z31.h}, z15.h[7]  // 11000001-11011111-11111111-10010111
// CHECK-INST: umlal   za.s[w11, 6:7, vgx4], { z28.h - z31.h }, z15.h[7]
// CHECK-ENCODING: [0x97,0xff,0xdf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dfff97 <unknown>

umlal   za.s[w11, 6:7], {z28.h - z31.h}, z15.h[7]  // 11000001-11011111-11111111-10010111
// CHECK-INST: umlal   za.s[w11, 6:7, vgx4], { z28.h - z31.h }, z15.h[7]
// CHECK-ENCODING: [0x97,0xff,0xdf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dfff97 <unknown>

umlal   za.s[w8, 2:3, vgx4], {z16.h - z19.h}, z0.h[7]  // 11000001-11010000-10011110-00010101
// CHECK-INST: umlal   za.s[w8, 2:3, vgx4], { z16.h - z19.h }, z0.h[7]
// CHECK-ENCODING: [0x15,0x9e,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d09e15 <unknown>

umlal   za.s[w8, 2:3], {z16.h - z19.h}, z0.h[7]  // 11000001-11010000-10011110-00010101
// CHECK-INST: umlal   za.s[w8, 2:3, vgx4], { z16.h - z19.h }, z0.h[7]
// CHECK-ENCODING: [0x15,0x9e,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d09e15 <unknown>

umlal   za.s[w8, 2:3, vgx4], {z0.h - z3.h}, z14.h[2]  // 11000001-11011110-10010100-00010001
// CHECK-INST: umlal   za.s[w8, 2:3, vgx4], { z0.h - z3.h }, z14.h[2]
// CHECK-ENCODING: [0x11,0x94,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de9411 <unknown>

umlal   za.s[w8, 2:3], {z0.h - z3.h}, z14.h[2]  // 11000001-11011110-10010100-00010001
// CHECK-INST: umlal   za.s[w8, 2:3, vgx4], { z0.h - z3.h }, z14.h[2]
// CHECK-ENCODING: [0x11,0x94,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de9411 <unknown>

umlal   za.s[w10, 0:1, vgx4], {z16.h - z19.h}, z4.h[2]  // 11000001-11010100-11010110-00010000
// CHECK-INST: umlal   za.s[w10, 0:1, vgx4], { z16.h - z19.h }, z4.h[2]
// CHECK-ENCODING: [0x10,0xd6,0xd4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d4d610 <unknown>

umlal   za.s[w10, 0:1], {z16.h - z19.h}, z4.h[2]  // 11000001-11010100-11010110-00010000
// CHECK-INST: umlal   za.s[w10, 0:1, vgx4], { z16.h - z19.h }, z4.h[2]
// CHECK-ENCODING: [0x10,0xd6,0xd4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d4d610 <unknown>

umlal   za.s[w8, 0:1, vgx4], {z12.h - z15.h}, z2.h[4]  // 11000001-11010010-10011001-10010000
// CHECK-INST: umlal   za.s[w8, 0:1, vgx4], { z12.h - z15.h }, z2.h[4]
// CHECK-ENCODING: [0x90,0x99,0xd2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d29990 <unknown>

umlal   za.s[w8, 0:1], {z12.h - z15.h}, z2.h[4]  // 11000001-11010010-10011001-10010000
// CHECK-INST: umlal   za.s[w8, 0:1, vgx4], { z12.h - z15.h }, z2.h[4]
// CHECK-ENCODING: [0x90,0x99,0xd2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d29990 <unknown>

umlal   za.s[w10, 2:3, vgx4], {z0.h - z3.h}, z10.h[4]  // 11000001-11011010-11011000-00010001
// CHECK-INST: umlal   za.s[w10, 2:3, vgx4], { z0.h - z3.h }, z10.h[4]
// CHECK-ENCODING: [0x11,0xd8,0xda,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dad811 <unknown>

umlal   za.s[w10, 2:3], {z0.h - z3.h}, z10.h[4]  // 11000001-11011010-11011000-00010001
// CHECK-INST: umlal   za.s[w10, 2:3, vgx4], { z0.h - z3.h }, z10.h[4]
// CHECK-ENCODING: [0x11,0xd8,0xda,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dad811 <unknown>

umlal   za.s[w8, 2:3, vgx4], {z20.h - z23.h}, z14.h[5]  // 11000001-11011110-10011010-10010101
// CHECK-INST: umlal   za.s[w8, 2:3, vgx4], { z20.h - z23.h }, z14.h[5]
// CHECK-ENCODING: [0x95,0x9a,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de9a95 <unknown>

umlal   za.s[w8, 2:3], {z20.h - z23.h}, z14.h[5]  // 11000001-11011110-10011010-10010101
// CHECK-INST: umlal   za.s[w8, 2:3, vgx4], { z20.h - z23.h }, z14.h[5]
// CHECK-ENCODING: [0x95,0x9a,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de9a95 <unknown>

umlal   za.s[w11, 4:5, vgx4], {z8.h - z11.h}, z1.h[2]  // 11000001-11010001-11110101-00010010
// CHECK-INST: umlal   za.s[w11, 4:5, vgx4], { z8.h - z11.h }, z1.h[2]
// CHECK-ENCODING: [0x12,0xf5,0xd1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d1f512 <unknown>

umlal   za.s[w11, 4:5], {z8.h - z11.h}, z1.h[2]  // 11000001-11010001-11110101-00010010
// CHECK-INST: umlal   za.s[w11, 4:5, vgx4], { z8.h - z11.h }, z1.h[2]
// CHECK-ENCODING: [0x12,0xf5,0xd1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d1f512 <unknown>

umlal   za.s[w9, 6:7, vgx4], {z12.h - z15.h}, z11.h[5]  // 11000001-11011011-10111001-10010111
// CHECK-INST: umlal   za.s[w9, 6:7, vgx4], { z12.h - z15.h }, z11.h[5]
// CHECK-ENCODING: [0x97,0xb9,0xdb,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dbb997 <unknown>

umlal   za.s[w9, 6:7], {z12.h - z15.h}, z11.h[5]  // 11000001-11011011-10111001-10010111
// CHECK-INST: umlal   za.s[w9, 6:7, vgx4], { z12.h - z15.h }, z11.h[5]
// CHECK-ENCODING: [0x97,0xb9,0xdb,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dbb997 <unknown>


umlal   za.s[w8, 0:1, vgx4], {z0.h - z3.h}, {z0.h - z3.h}  // 11000001-11100001-00001000-00010000
// CHECK-INST: umlal   za.s[w8, 0:1, vgx4], { z0.h - z3.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x10,0x08,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e10810 <unknown>

umlal   za.s[w8, 0:1], {z0.h - z3.h}, {z0.h - z3.h}  // 11000001-11100001-00001000-00010000
// CHECK-INST: umlal   za.s[w8, 0:1, vgx4], { z0.h - z3.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x10,0x08,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e10810 <unknown>

umlal   za.s[w10, 2:3, vgx4], {z8.h - z11.h}, {z20.h - z23.h}  // 11000001-11110101-01001001-00010001
// CHECK-INST: umlal   za.s[w10, 2:3, vgx4], { z8.h - z11.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x11,0x49,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f54911 <unknown>

umlal   za.s[w10, 2:3], {z8.h - z11.h}, {z20.h - z23.h}  // 11000001-11110101-01001001-00010001
// CHECK-INST: umlal   za.s[w10, 2:3, vgx4], { z8.h - z11.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x11,0x49,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f54911 <unknown>

umlal   za.s[w11, 6:7, vgx4], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-11101001-01101001-10010011
// CHECK-INST: umlal   za.s[w11, 6:7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x93,0x69,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e96993 <unknown>

umlal   za.s[w11, 6:7], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-11101001-01101001-10010011
// CHECK-INST: umlal   za.s[w11, 6:7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x93,0x69,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e96993 <unknown>

umlal   za.s[w11, 6:7, vgx4], {z28.h - z31.h}, {z28.h - z31.h}  // 11000001-11111101-01101011-10010011
// CHECK-INST: umlal   za.s[w11, 6:7, vgx4], { z28.h - z31.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x93,0x6b,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd6b93 <unknown>

umlal   za.s[w11, 6:7], {z28.h - z31.h}, {z28.h - z31.h}  // 11000001-11111101-01101011-10010011
// CHECK-INST: umlal   za.s[w11, 6:7, vgx4], { z28.h - z31.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x93,0x6b,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd6b93 <unknown>

umlal   za.s[w8, 2:3, vgx4], {z16.h - z19.h}, {z16.h - z19.h}  // 11000001-11110001-00001010-00010001
// CHECK-INST: umlal   za.s[w8, 2:3, vgx4], { z16.h - z19.h }, { z16.h - z19.h }
// CHECK-ENCODING: [0x11,0x0a,0xf1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f10a11 <unknown>

umlal   za.s[w8, 2:3], {z16.h - z19.h}, {z16.h - z19.h}  // 11000001-11110001-00001010-00010001
// CHECK-INST: umlal   za.s[w8, 2:3, vgx4], { z16.h - z19.h }, { z16.h - z19.h }
// CHECK-ENCODING: [0x11,0x0a,0xf1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f10a11 <unknown>

umlal   za.s[w8, 2:3, vgx4], {z0.h - z3.h}, {z28.h - z31.h}  // 11000001-11111101-00001000-00010001
// CHECK-INST: umlal   za.s[w8, 2:3, vgx4], { z0.h - z3.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x11,0x08,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd0811 <unknown>

umlal   za.s[w8, 2:3], {z0.h - z3.h}, {z28.h - z31.h}  // 11000001-11111101-00001000-00010001
// CHECK-INST: umlal   za.s[w8, 2:3, vgx4], { z0.h - z3.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x11,0x08,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd0811 <unknown>

umlal   za.s[w10, 0:1, vgx4], {z16.h - z19.h}, {z20.h - z23.h}  // 11000001-11110101-01001010-00010000
// CHECK-INST: umlal   za.s[w10, 0:1, vgx4], { z16.h - z19.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x10,0x4a,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f54a10 <unknown>

umlal   za.s[w10, 0:1], {z16.h - z19.h}, {z20.h - z23.h}  // 11000001-11110101-01001010-00010000
// CHECK-INST: umlal   za.s[w10, 0:1, vgx4], { z16.h - z19.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x10,0x4a,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f54a10 <unknown>

umlal   za.s[w8, 0:1, vgx4], {z12.h - z15.h}, {z0.h - z3.h}  // 11000001-11100001-00001001-10010000
// CHECK-INST: umlal   za.s[w8, 0:1, vgx4], { z12.h - z15.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x90,0x09,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e10990 <unknown>

umlal   za.s[w8, 0:1], {z12.h - z15.h}, {z0.h - z3.h}  // 11000001-11100001-00001001-10010000
// CHECK-INST: umlal   za.s[w8, 0:1, vgx4], { z12.h - z15.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x90,0x09,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e10990 <unknown>

umlal   za.s[w10, 2:3, vgx4], {z0.h - z3.h}, {z24.h - z27.h}  // 11000001-11111001-01001000-00010001
// CHECK-INST: umlal   za.s[w10, 2:3, vgx4], { z0.h - z3.h }, { z24.h - z27.h }
// CHECK-ENCODING: [0x11,0x48,0xf9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f94811 <unknown>

umlal   za.s[w10, 2:3], {z0.h - z3.h}, {z24.h - z27.h}  // 11000001-11111001-01001000-00010001
// CHECK-INST: umlal   za.s[w10, 2:3, vgx4], { z0.h - z3.h }, { z24.h - z27.h }
// CHECK-ENCODING: [0x11,0x48,0xf9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f94811 <unknown>

umlal   za.s[w8, 2:3, vgx4], {z20.h - z23.h}, {z28.h - z31.h}  // 11000001-11111101-00001010-10010001
// CHECK-INST: umlal   za.s[w8, 2:3, vgx4], { z20.h - z23.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x91,0x0a,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd0a91 <unknown>

umlal   za.s[w8, 2:3], {z20.h - z23.h}, {z28.h - z31.h}  // 11000001-11111101-00001010-10010001
// CHECK-INST: umlal   za.s[w8, 2:3, vgx4], { z20.h - z23.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x91,0x0a,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd0a91 <unknown>

umlal   za.s[w11, 4:5, vgx4], {z8.h - z11.h}, {z0.h - z3.h}  // 11000001-11100001-01101001-00010010
// CHECK-INST: umlal   za.s[w11, 4:5, vgx4], { z8.h - z11.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x12,0x69,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e16912 <unknown>

umlal   za.s[w11, 4:5], {z8.h - z11.h}, {z0.h - z3.h}  // 11000001-11100001-01101001-00010010
// CHECK-INST: umlal   za.s[w11, 4:5, vgx4], { z8.h - z11.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x12,0x69,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e16912 <unknown>

umlal   za.s[w9, 6:7, vgx4], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-11101001-00101001-10010011
// CHECK-INST: umlal   za.s[w9, 6:7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x93,0x29,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e92993 <unknown>

umlal   za.s[w9, 6:7], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-11101001-00101001-10010011
// CHECK-INST: umlal   za.s[w9, 6:7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x93,0x29,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e92993 <unknown>

