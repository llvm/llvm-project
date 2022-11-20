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


bfmlal  za.s[w8, 0:1], z0.h, z0.h  // 11000001-00100000-00001100-00010000
// CHECK-INST: bfmlal  za.s[w8, 0:1], z0.h, z0.h
// CHECK-ENCODING: [0x10,0x0c,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1200c10 <unknown>

bfmlal  za.s[w10, 10:11], z10.h, z5.h  // 11000001-00100101-01001101-01010101
// CHECK-INST: bfmlal  za.s[w10, 10:11], z10.h, z5.h
// CHECK-ENCODING: [0x55,0x4d,0x25,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1254d55 <unknown>

bfmlal  za.s[w11, 14:15], z13.h, z8.h  // 11000001-00101000-01101101-10110111
// CHECK-INST: bfmlal  za.s[w11, 14:15], z13.h, z8.h
// CHECK-ENCODING: [0xb7,0x6d,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1286db7 <unknown>

bfmlal  za.s[w11, 14:15], z31.h, z15.h  // 11000001-00101111-01101111-11110111
// CHECK-INST: bfmlal  za.s[w11, 14:15], z31.h, z15.h
// CHECK-ENCODING: [0xf7,0x6f,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12f6ff7 <unknown>

bfmlal  za.s[w8, 10:11], z17.h, z0.h  // 11000001-00100000-00001110-00110101
// CHECK-INST: bfmlal  za.s[w8, 10:11], z17.h, z0.h
// CHECK-ENCODING: [0x35,0x0e,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1200e35 <unknown>

bfmlal  za.s[w8, 2:3], z1.h, z14.h  // 11000001-00101110-00001100-00110001
// CHECK-INST: bfmlal  za.s[w8, 2:3], z1.h, z14.h
// CHECK-ENCODING: [0x31,0x0c,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e0c31 <unknown>

bfmlal  za.s[w10, 0:1], z19.h, z4.h  // 11000001-00100100-01001110-01110000
// CHECK-INST: bfmlal  za.s[w10, 0:1], z19.h, z4.h
// CHECK-ENCODING: [0x70,0x4e,0x24,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1244e70 <unknown>

bfmlal  za.s[w8, 0:1], z12.h, z2.h  // 11000001-00100010-00001101-10010000
// CHECK-INST: bfmlal  za.s[w8, 0:1], z12.h, z2.h
// CHECK-ENCODING: [0x90,0x0d,0x22,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1220d90 <unknown>

bfmlal  za.s[w10, 2:3], z1.h, z10.h  // 11000001-00101010-01001100-00110001
// CHECK-INST: bfmlal  za.s[w10, 2:3], z1.h, z10.h
// CHECK-ENCODING: [0x31,0x4c,0x2a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12a4c31 <unknown>

bfmlal  za.s[w8, 10:11], z22.h, z14.h  // 11000001-00101110-00001110-11010101
// CHECK-INST: bfmlal  za.s[w8, 10:11], z22.h, z14.h
// CHECK-ENCODING: [0xd5,0x0e,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e0ed5 <unknown>

bfmlal  za.s[w11, 4:5], z9.h, z1.h  // 11000001-00100001-01101101-00110010
// CHECK-INST: bfmlal  za.s[w11, 4:5], z9.h, z1.h
// CHECK-ENCODING: [0x32,0x6d,0x21,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1216d32 <unknown>

bfmlal  za.s[w9, 14:15], z12.h, z11.h  // 11000001-00101011-00101101-10010111
// CHECK-INST: bfmlal  za.s[w9, 14:15], z12.h, z11.h
// CHECK-ENCODING: [0x97,0x2d,0x2b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12b2d97 <unknown>


bfmlal  za.s[w8, 0:1], z0.h, z0.h[0]  // 11000001-10000000-00010000-00010000
// CHECK-INST: bfmlal  za.s[w8, 0:1], z0.h, z0.h[0]
// CHECK-ENCODING: [0x10,0x10,0x80,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1801010 <unknown>

bfmlal  za.s[w10, 10:11], z10.h, z5.h[1]  // 11000001-10000101-01010101-01010101
// CHECK-INST: bfmlal  za.s[w10, 10:11], z10.h, z5.h[1]
// CHECK-ENCODING: [0x55,0x55,0x85,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1855555 <unknown>

bfmlal  za.s[w11, 14:15], z13.h, z8.h[7]  // 11000001-10001000-11111101-10110111
// CHECK-INST: bfmlal  za.s[w11, 14:15], z13.h, z8.h[7]
// CHECK-ENCODING: [0xb7,0xfd,0x88,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c188fdb7 <unknown>

bfmlal  za.s[w11, 14:15], z31.h, z15.h[7]  // 11000001-10001111-11111111-11110111
// CHECK-INST: bfmlal  za.s[w11, 14:15], z31.h, z15.h[7]
// CHECK-ENCODING: [0xf7,0xff,0x8f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c18ffff7 <unknown>

bfmlal  za.s[w8, 10:11], z17.h, z0.h[3]  // 11000001-10000000-00011110-00110101
// CHECK-INST: bfmlal  za.s[w8, 10:11], z17.h, z0.h[3]
// CHECK-ENCODING: [0x35,0x1e,0x80,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1801e35 <unknown>

bfmlal  za.s[w8, 2:3], z1.h, z14.h[5]  // 11000001-10001110-10010100-00110001
// CHECK-INST: bfmlal  za.s[w8, 2:3], z1.h, z14.h[5]
// CHECK-ENCODING: [0x31,0x94,0x8e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c18e9431 <unknown>

bfmlal  za.s[w10, 0:1], z19.h, z4.h[1]  // 11000001-10000100-01010110-01110000
// CHECK-INST: bfmlal  za.s[w10, 0:1], z19.h, z4.h[1]
// CHECK-ENCODING: [0x70,0x56,0x84,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1845670 <unknown>

bfmlal  za.s[w8, 0:1], z12.h, z2.h[2]  // 11000001-10000010-00011001-10010000
// CHECK-INST: bfmlal  za.s[w8, 0:1], z12.h, z2.h[2]
// CHECK-ENCODING: [0x90,0x19,0x82,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1821990 <unknown>

bfmlal  za.s[w10, 2:3], z1.h, z10.h[6]  // 11000001-10001010-11011000-00110001
// CHECK-INST: bfmlal  za.s[w10, 2:3], z1.h, z10.h[6]
// CHECK-ENCODING: [0x31,0xd8,0x8a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c18ad831 <unknown>

bfmlal  za.s[w8, 10:11], z22.h, z14.h[2]  // 11000001-10001110-00011010-11010101
// CHECK-INST: bfmlal  za.s[w8, 10:11], z22.h, z14.h[2]
// CHECK-ENCODING: [0xd5,0x1a,0x8e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c18e1ad5 <unknown>

bfmlal  za.s[w11, 4:5], z9.h, z1.h[5]  // 11000001-10000001-11110101-00110010
// CHECK-INST: bfmlal  za.s[w11, 4:5], z9.h, z1.h[5]
// CHECK-ENCODING: [0x32,0xf5,0x81,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c181f532 <unknown>

bfmlal  za.s[w9, 14:15], z12.h, z11.h[6]  // 11000001-10001011-10111001-10010111
// CHECK-INST: bfmlal  za.s[w9, 14:15], z12.h, z11.h[6]
// CHECK-ENCODING: [0x97,0xb9,0x8b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c18bb997 <unknown>


bfmlal  za.s[w8, 0:1, vgx2], {z0.h, z1.h}, z0.h  // 11000001, 00100000, 00001000, 00010000
// CHECK, INST: bfmlal  za.s[w8, 0:1, vgx2], { z0.h, z1.h }, z0.h
// CHECK-ENCODING: [0x10,0x08,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1200810 <unknown>

bfmlal  za.s[w8, 0:1], {z0.h - z1.h}, z0.h  // 11000001-00100000-00001000-00010000
// CHECK, INST: bfmlal  za.s[w8, 0:1, vgx2], { z0.h, z1.h }, z0.h
// CHECK-ENCODING: [0x10,0x08,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1200810 <unknown>

bfmlal  za.s[w10, 2:3, vgx2], {z10.h, z11.h}, z5.h  // 11000001, 00100101, 01001001, 01010001
// CHECK, INST: bfmlal  za.s[w10, 2:3, vgx2], { z10.h, z11.h }, z5.h
// CHECK-ENCODING: [0x51,0x49,0x25,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1254951 <unknown>

bfmlal  za.s[w10, 2:3], {z10.h - z11.h}, z5.h  // 11000001-00100101-01001001-01010001
// CHECK, INST: bfmlal  za.s[w10, 2:3, vgx2], { z10.h, z11.h }, z5.h
// CHECK-ENCODING: [0x51,0x49,0x25,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1254951 <unknown>

bfmlal  za.s[w11, 6:7, vgx2], {z13.h, z14.h}, z8.h  // 11000001, 00101000, 01101001, 10110011
// CHECK, INST: bfmlal  za.s[w11, 6:7, vgx2], { z13.h, z14.h }, z8.h
// CHECK-ENCODING: [0xb3,0x69,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12869b3 <unknown>

bfmlal  za.s[w11, 6:7], {z13.h - z14.h}, z8.h  // 11000001-00101000-01101001-10110011
// CHECK, INST: bfmlal  za.s[w11, 6:7, vgx2], { z13.h, z14.h }, z8.h
// CHECK-ENCODING: [0xb3,0x69,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12869b3 <unknown>

bfmlal  za.s[w11, 6:7, vgx2], {z31.h, z0.h}, z15.h  // 11000001, 00101111, 01101011, 11110011
// CHECK, INST: bfmlal  za.s[w11, 6:7, vgx2], { z31.h, z0.h }, z15.h
// CHECK-ENCODING: [0xf3,0x6b,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12f6bf3 <unknown>

bfmlal  za.s[w11, 6:7], {z31.h - z0.h}, z15.h  // 11000001-00101111-01101011-11110011
// CHECK, INST: bfmlal  za.s[w11, 6:7, vgx2], { z31.h, z0.h }, z15.h
// CHECK-ENCODING: [0xf3,0x6b,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12f6bf3 <unknown>

bfmlal  za.s[w8, 2:3, vgx2], {z17.h, z18.h}, z0.h  // 11000001, 00100000, 00001010, 00110001
// CHECK, INST: bfmlal  za.s[w8, 2:3, vgx2], { z17.h, z18.h }, z0.h
// CHECK-ENCODING: [0x31,0x0a,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1200a31 <unknown>

bfmlal  za.s[w8, 2:3], {z17.h - z18.h}, z0.h  // 11000001-00100000-00001010-00110001
// CHECK, INST: bfmlal  za.s[w8, 2:3, vgx2], { z17.h, z18.h }, z0.h
// CHECK-ENCODING: [0x31,0x0a,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1200a31 <unknown>

bfmlal  za.s[w8, 2:3, vgx2], {z1.h, z2.h}, z14.h  // 11000001, 00101110, 00001000, 00110001
// CHECK, INST: bfmlal  za.s[w8, 2:3, vgx2], { z1.h, z2.h }, z14.h
// CHECK-ENCODING: [0x31,0x08,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e0831 <unknown>

bfmlal  za.s[w8, 2:3], {z1.h - z2.h}, z14.h  // 11000001-00101110-00001000-00110001
// CHECK, INST: bfmlal  za.s[w8, 2:3, vgx2], { z1.h, z2.h }, z14.h
// CHECK-ENCODING: [0x31,0x08,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e0831 <unknown>

bfmlal  za.s[w10, 0:1, vgx2], {z19.h, z20.h}, z4.h  // 11000001, 00100100, 01001010, 01110000
// CHECK, INST: bfmlal  za.s[w10, 0:1, vgx2], { z19.h, z20.h }, z4.h
// CHECK-ENCODING: [0x70,0x4a,0x24,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1244a70 <unknown>

bfmlal  za.s[w10, 0:1], {z19.h - z20.h}, z4.h  // 11000001-00100100-01001010-01110000
// CHECK, INST: bfmlal  za.s[w10, 0:1, vgx2], { z19.h, z20.h }, z4.h
// CHECK-ENCODING: [0x70,0x4a,0x24,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1244a70 <unknown>

bfmlal  za.s[w8, 0:1, vgx2], {z12.h, z13.h}, z2.h  // 11000001, 00100010, 00001001, 10010000
// CHECK, INST: bfmlal  za.s[w8, 0:1, vgx2], { z12.h, z13.h }, z2.h
// CHECK-ENCODING: [0x90,0x09,0x22,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1220990 <unknown>

bfmlal  za.s[w8, 0:1], {z12.h - z13.h}, z2.h  // 11000001-00100010-00001001-10010000
// CHECK, INST: bfmlal  za.s[w8, 0:1, vgx2], { z12.h, z13.h }, z2.h
// CHECK-ENCODING: [0x90,0x09,0x22,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1220990 <unknown>

bfmlal  za.s[w10, 2:3, vgx2], {z1.h, z2.h}, z10.h  // 11000001, 00101010, 01001000, 00110001
// CHECK, INST: bfmlal  za.s[w10, 2:3, vgx2], { z1.h, z2.h }, z10.h
// CHECK-ENCODING: [0x31,0x48,0x2a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12a4831 <unknown>

bfmlal  za.s[w10, 2:3], {z1.h - z2.h}, z10.h  // 11000001-00101010-01001000-00110001
// CHECK, INST: bfmlal  za.s[w10, 2:3, vgx2], { z1.h, z2.h }, z10.h
// CHECK-ENCODING: [0x31,0x48,0x2a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12a4831 <unknown>

bfmlal  za.s[w8, 2:3, vgx2], {z22.h, z23.h}, z14.h  // 11000001, 00101110, 00001010, 11010001
// CHECK, INST: bfmlal  za.s[w8, 2:3, vgx2], { z22.h, z23.h }, z14.h
// CHECK-ENCODING: [0xd1,0x0a,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e0ad1 <unknown>

bfmlal  za.s[w8, 2:3], {z22.h - z23.h}, z14.h  // 11000001-00101110-00001010-11010001
// CHECK, INST: bfmlal  za.s[w8, 2:3, vgx2], { z22.h, z23.h }, z14.h
// CHECK-ENCODING: [0xd1,0x0a,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e0ad1 <unknown>

bfmlal  za.s[w11, 4:5, vgx2], {z9.h, z10.h}, z1.h  // 11000001, 00100001, 01101001, 00110010
// CHECK, INST: bfmlal  za.s[w11, 4:5, vgx2], { z9.h, z10.h }, z1.h
// CHECK-ENCODING: [0x32,0x69,0x21,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1216932 <unknown>

bfmlal  za.s[w11, 4:5], {z9.h - z10.h}, z1.h  // 11000001-00100001-01101001-00110010
// CHECK, INST: bfmlal  za.s[w11, 4:5, vgx2], { z9.h, z10.h }, z1.h
// CHECK-ENCODING: [0x32,0x69,0x21,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1216932 <unknown>

bfmlal  za.s[w9, 6:7, vgx2], {z12.h, z13.h}, z11.h  // 11000001, 00101011, 00101001, 10010011
// CHECK, INST: bfmlal  za.s[w9, 6:7, vgx2], { z12.h, z13.h }, z11.h
// CHECK-ENCODING: [0x93,0x29,0x2b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12b2993 <unknown>

bfmlal  za.s[w9, 6:7], {z12.h - z13.h}, z11.h  // 11000001-00101011-00101001-10010011
// CHECK, INST: bfmlal  za.s[w9, 6:7, vgx2], { z12.h, z13.h }, z11.h
// CHECK-ENCODING: [0x93,0x29,0x2b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12b2993 <unknown>


bfmlal  za.s[w8, 0:1, vgx2], {z0.h, z1.h}, z0.h[0]  // 11000001, 10010000, 00010000, 00010000
// CHECK, INST: bfmlal  za.s[w8, 0:1, vgx2], { z0.h, z1.h }, z0.h[0]
// CHECK-ENCODING: [0x10,0x10,0x90,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1901010 <unknown>

bfmlal  za.s[w8, 0:1], {z0.h - z1.h}, z0.h[0]  // 11000001-10010000-00010000-00010000
// CHECK, INST: bfmlal  za.s[w8, 0:1, vgx2], { z0.h, z1.h }, z0.h[0]
// CHECK-ENCODING: [0x10,0x10,0x90,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1901010 <unknown>

bfmlal  za.s[w10, 2:3, vgx2], {z10.h, z11.h}, z5.h[3]  // 11000001, 10010101, 01010101, 01010101
// CHECK, INST: bfmlal  za.s[w10, 2:3, vgx2], { z10.h, z11.h }, z5.h[3]
// CHECK-ENCODING: [0x55,0x55,0x95,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1955555 <unknown>

bfmlal  za.s[w10, 2:3], {z10.h - z11.h}, z5.h[3]  // 11000001-10010101-01010101-01010101
// CHECK, INST: bfmlal  za.s[w10, 2:3, vgx2], { z10.h, z11.h }, z5.h[3]
// CHECK-ENCODING: [0x55,0x55,0x95,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1955555 <unknown>

bfmlal  za.s[w11, 6:7, vgx2], {z12.h, z13.h}, z8.h[7]  // 11000001, 10011000, 01111101, 10010111
// CHECK, INST: bfmlal  za.s[w11, 6:7, vgx2], { z12.h, z13.h }, z8.h[7]
// CHECK-ENCODING: [0x97,0x7d,0x98,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1987d97 <unknown>

bfmlal  za.s[w11, 6:7], {z12.h - z13.h}, z8.h[7]  // 11000001-10011000-01111101-10010111
// CHECK, INST: bfmlal  za.s[w11, 6:7, vgx2], { z12.h, z13.h }, z8.h[7]
// CHECK-ENCODING: [0x97,0x7d,0x98,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1987d97 <unknown>

bfmlal  za.s[w11, 6:7, vgx2], {z30.h, z31.h}, z15.h[7]  // 11000001, 10011111, 01111111, 11010111
// CHECK, INST: bfmlal  za.s[w11, 6:7, vgx2], { z30.h, z31.h }, z15.h[7]
// CHECK-ENCODING: [0xd7,0x7f,0x9f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c19f7fd7 <unknown>

bfmlal  za.s[w11, 6:7], {z30.h - z31.h}, z15.h[7]  // 11000001-10011111-01111111-11010111
// CHECK, INST: bfmlal  za.s[w11, 6:7, vgx2], { z30.h, z31.h }, z15.h[7]
// CHECK-ENCODING: [0xd7,0x7f,0x9f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c19f7fd7 <unknown>

bfmlal  za.s[w8, 2:3, vgx2], {z16.h, z17.h}, z0.h[7]  // 11000001, 10010000, 00011110, 00010101
// CHECK, INST: bfmlal  za.s[w8, 2:3, vgx2], { z16.h, z17.h }, z0.h[7]
// CHECK-ENCODING: [0x15,0x1e,0x90,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1901e15 <unknown>

bfmlal  za.s[w8, 2:3], {z16.h - z17.h}, z0.h[7]  // 11000001-10010000-00011110-00010101
// CHECK, INST: bfmlal  za.s[w8, 2:3, vgx2], { z16.h, z17.h }, z0.h[7]
// CHECK-ENCODING: [0x15,0x1e,0x90,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1901e15 <unknown>

bfmlal  za.s[w8, 2:3, vgx2], {z0.h, z1.h}, z14.h[2]  // 11000001, 10011110, 00010100, 00010001
// CHECK, INST: bfmlal  za.s[w8, 2:3, vgx2], { z0.h, z1.h }, z14.h[2]
// CHECK-ENCODING: [0x11,0x14,0x9e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c19e1411 <unknown>

bfmlal  za.s[w8, 2:3], {z0.h - z1.h}, z14.h[2]  // 11000001-10011110-00010100-00010001
// CHECK, INST: bfmlal  za.s[w8, 2:3, vgx2], { z0.h, z1.h }, z14.h[2]
// CHECK-ENCODING: [0x11,0x14,0x9e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c19e1411 <unknown>

bfmlal  za.s[w10, 0:1, vgx2], {z18.h, z19.h}, z4.h[2]  // 11000001, 10010100, 01010110, 01010000
// CHECK, INST: bfmlal  za.s[w10, 0:1, vgx2], { z18.h, z19.h }, z4.h[2]
// CHECK-ENCODING: [0x50,0x56,0x94,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1945650 <unknown>

bfmlal  za.s[w10, 0:1], {z18.h - z19.h}, z4.h[2]  // 11000001-10010100-01010110-01010000
// CHECK, INST: bfmlal  za.s[w10, 0:1, vgx2], { z18.h, z19.h }, z4.h[2]
// CHECK-ENCODING: [0x50,0x56,0x94,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1945650 <unknown>

bfmlal  za.s[w8, 0:1, vgx2], {z12.h, z13.h}, z2.h[4]  // 11000001, 10010010, 00011001, 10010000
// CHECK, INST: bfmlal  za.s[w8, 0:1, vgx2], { z12.h, z13.h }, z2.h[4]
// CHECK-ENCODING: [0x90,0x19,0x92,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1921990 <unknown>

bfmlal  za.s[w8, 0:1], {z12.h - z13.h}, z2.h[4]  // 11000001-10010010-00011001-10010000
// CHECK, INST: bfmlal  za.s[w8, 0:1, vgx2], { z12.h, z13.h }, z2.h[4]
// CHECK-ENCODING: [0x90,0x19,0x92,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1921990 <unknown>

bfmlal  za.s[w10, 2:3, vgx2], {z0.h, z1.h}, z10.h[4]  // 11000001, 10011010, 01011000, 00010001
// CHECK, INST: bfmlal  za.s[w10, 2:3, vgx2], { z0.h, z1.h }, z10.h[4]
// CHECK-ENCODING: [0x11,0x58,0x9a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c19a5811 <unknown>

bfmlal  za.s[w10, 2:3], {z0.h - z1.h}, z10.h[4]  // 11000001-10011010-01011000-00010001
// CHECK, INST: bfmlal  za.s[w10, 2:3, vgx2], { z0.h, z1.h }, z10.h[4]
// CHECK-ENCODING: [0x11,0x58,0x9a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c19a5811 <unknown>

bfmlal  za.s[w8, 2:3, vgx2], {z22.h, z23.h}, z14.h[5]  // 11000001, 10011110, 00011010, 11010101
// CHECK, INST: bfmlal  za.s[w8, 2:3, vgx2], { z22.h, z23.h }, z14.h[5]
// CHECK-ENCODING: [0xd5,0x1a,0x9e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c19e1ad5 <unknown>

bfmlal  za.s[w8, 2:3], {z22.h - z23.h}, z14.h[5]  // 11000001-10011110-00011010-11010101
// CHECK, INST: bfmlal  za.s[w8, 2:3, vgx2], { z22.h, z23.h }, z14.h[5]
// CHECK-ENCODING: [0xd5,0x1a,0x9e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c19e1ad5 <unknown>

bfmlal  za.s[w11, 4:5, vgx2], {z8.h, z9.h}, z1.h[2]  // 11000001, 10010001, 01110101, 00010010
// CHECK, INST: bfmlal  za.s[w11, 4:5, vgx2], { z8.h, z9.h }, z1.h[2]
// CHECK-ENCODING: [0x12,0x75,0x91,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1917512 <unknown>

bfmlal  za.s[w11, 4:5], {z8.h - z9.h}, z1.h[2]  // 11000001-10010001-01110101-00010010
// CHECK, INST: bfmlal  za.s[w11, 4:5, vgx2], { z8.h, z9.h }, z1.h[2]
// CHECK-ENCODING: [0x12,0x75,0x91,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1917512 <unknown>

bfmlal  za.s[w9, 6:7, vgx2], {z12.h, z13.h}, z11.h[5]  // 11000001, 10011011, 00111001, 10010111
// CHECK, INST: bfmlal  za.s[w9, 6:7, vgx2], { z12.h, z13.h }, z11.h[5]
// CHECK-ENCODING: [0x97,0x39,0x9b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c19b3997 <unknown>

bfmlal  za.s[w9, 6:7], {z12.h - z13.h}, z11.h[5]  // 11000001-10011011-00111001-10010111
// CHECK, INST: bfmlal  za.s[w9, 6:7, vgx2], { z12.h, z13.h }, z11.h[5]
// CHECK-ENCODING: [0x97,0x39,0x9b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c19b3997 <unknown>


bfmlal  za.s[w8, 0:1, vgx2], {z0.h, z1.h}, {z0.h, z1.h}  // 11000001, 10100000, 00001000, 00010000
// CHECK, INST: bfmlal  za.s[w8, 0:1, vgx2], { z0.h, z1.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x10,0x08,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a00810 <unknown>

bfmlal  za.s[w8, 0:1], {z0.h - z1.h}, {z0.h - z1.h}  // 11000001-10100000-00001000-00010000
// CHECK, INST: bfmlal  za.s[w8, 0:1, vgx2], { z0.h, z1.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x10,0x08,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a00810 <unknown>

bfmlal  za.s[w10, 2:3, vgx2], {z10.h, z11.h}, {z20.h, z21.h}  // 11000001, 10110100, 01001001, 01010001
// CHECK, INST: bfmlal  za.s[w10, 2:3, vgx2], { z10.h, z11.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x51,0x49,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b44951 <unknown>

bfmlal  za.s[w10, 2:3], {z10.h - z11.h}, {z20.h - z21.h}  // 11000001-10110100-01001001-01010001
// CHECK, INST: bfmlal  za.s[w10, 2:3, vgx2], { z10.h, z11.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x51,0x49,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b44951 <unknown>

bfmlal  za.s[w11, 6:7, vgx2], {z12.h, z13.h}, {z8.h, z9.h}  // 11000001, 10101000, 01101001, 10010011
// CHECK, INST: bfmlal  za.s[w11, 6:7, vgx2], { z12.h, z13.h }, { z8.h, z9.h }
// CHECK-ENCODING: [0x93,0x69,0xa8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a86993 <unknown>

bfmlal  za.s[w11, 6:7], {z12.h - z13.h}, {z8.h - z9.h}  // 11000001-10101000-01101001-10010011
// CHECK, INST: bfmlal  za.s[w11, 6:7, vgx2], { z12.h, z13.h }, { z8.h, z9.h }
// CHECK-ENCODING: [0x93,0x69,0xa8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a86993 <unknown>

bfmlal  za.s[w11, 6:7, vgx2], {z30.h, z31.h}, {z30.h, z31.h}  // 11000001, 10111110, 01101011, 11010011
// CHECK, INST: bfmlal  za.s[w11, 6:7, vgx2], { z30.h, z31.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xd3,0x6b,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be6bd3 <unknown>

bfmlal  za.s[w11, 6:7], {z30.h - z31.h}, {z30.h - z31.h}  // 11000001-10111110-01101011-11010011
// CHECK, INST: bfmlal  za.s[w11, 6:7, vgx2], { z30.h, z31.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xd3,0x6b,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be6bd3 <unknown>

bfmlal  za.s[w8, 2:3, vgx2], {z16.h, z17.h}, {z16.h, z17.h}  // 11000001, 10110000, 00001010, 00010001
// CHECK, INST: bfmlal  za.s[w8, 2:3, vgx2], { z16.h, z17.h }, { z16.h, z17.h }
// CHECK-ENCODING: [0x11,0x0a,0xb0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b00a11 <unknown>

bfmlal  za.s[w8, 2:3], {z16.h - z17.h}, {z16.h - z17.h}  // 11000001-10110000-00001010-00010001
// CHECK, INST: bfmlal  za.s[w8, 2:3, vgx2], { z16.h, z17.h }, { z16.h, z17.h }
// CHECK-ENCODING: [0x11,0x0a,0xb0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b00a11 <unknown>

bfmlal  za.s[w8, 2:3, vgx2], {z0.h, z1.h}, {z30.h, z31.h}  // 11000001, 10111110, 00001000, 00010001
// CHECK, INST: bfmlal  za.s[w8, 2:3, vgx2], { z0.h, z1.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0x11,0x08,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be0811 <unknown>

bfmlal  za.s[w8, 2:3], {z0.h - z1.h}, {z30.h - z31.h}  // 11000001-10111110-00001000-00010001
// CHECK, INST: bfmlal  za.s[w8, 2:3, vgx2], { z0.h, z1.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0x11,0x08,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be0811 <unknown>

bfmlal  za.s[w10, 0:1, vgx2], {z18.h, z19.h}, {z20.h, z21.h}  // 11000001, 10110100, 01001010, 01010000
// CHECK, INST: bfmlal  za.s[w10, 0:1, vgx2], { z18.h, z19.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x50,0x4a,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b44a50 <unknown>

bfmlal  za.s[w10, 0:1], {z18.h - z19.h}, {z20.h - z21.h}  // 11000001-10110100-01001010-01010000
// CHECK, INST: bfmlal  za.s[w10, 0:1, vgx2], { z18.h, z19.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x50,0x4a,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b44a50 <unknown>

bfmlal  za.s[w8, 0:1, vgx2], {z12.h, z13.h}, {z2.h, z3.h}  // 11000001, 10100010, 00001001, 10010000
// CHECK, INST: bfmlal  za.s[w8, 0:1, vgx2], { z12.h, z13.h }, { z2.h, z3.h }
// CHECK-ENCODING: [0x90,0x09,0xa2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a20990 <unknown>

bfmlal  za.s[w8, 0:1], {z12.h - z13.h}, {z2.h - z3.h}  // 11000001-10100010-00001001-10010000
// CHECK, INST: bfmlal  za.s[w8, 0:1, vgx2], { z12.h, z13.h }, { z2.h, z3.h }
// CHECK-ENCODING: [0x90,0x09,0xa2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a20990 <unknown>

bfmlal  za.s[w10, 2:3, vgx2], {z0.h, z1.h}, {z26.h, z27.h}  // 11000001, 10111010, 01001000, 00010001
// CHECK, INST: bfmlal  za.s[w10, 2:3, vgx2], { z0.h, z1.h }, { z26.h, z27.h }
// CHECK-ENCODING: [0x11,0x48,0xba,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ba4811 <unknown>

bfmlal  za.s[w10, 2:3], {z0.h - z1.h}, {z26.h - z27.h}  // 11000001-10111010-01001000-00010001
// CHECK, INST: bfmlal  za.s[w10, 2:3, vgx2], { z0.h, z1.h }, { z26.h, z27.h }
// CHECK-ENCODING: [0x11,0x48,0xba,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ba4811 <unknown>

bfmlal  za.s[w8, 2:3, vgx2], {z22.h, z23.h}, {z30.h, z31.h}  // 11000001, 10111110, 00001010, 11010001
// CHECK, INST: bfmlal  za.s[w8, 2:3, vgx2], { z22.h, z23.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xd1,0x0a,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be0ad1 <unknown>

bfmlal  za.s[w8, 2:3], {z22.h - z23.h}, {z30.h - z31.h}  // 11000001-10111110-00001010-11010001
// CHECK, INST: bfmlal  za.s[w8, 2:3, vgx2], { z22.h, z23.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xd1,0x0a,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be0ad1 <unknown>

bfmlal  za.s[w11, 4:5, vgx2], {z8.h, z9.h}, {z0.h, z1.h}  // 11000001, 10100000, 01101001, 00010010
// CHECK, INST: bfmlal  za.s[w11, 4:5, vgx2], { z8.h, z9.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x12,0x69,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a06912 <unknown>

bfmlal  za.s[w11, 4:5], {z8.h - z9.h}, {z0.h - z1.h}  // 11000001-10100000-01101001-00010010
// CHECK, INST: bfmlal  za.s[w11, 4:5, vgx2], { z8.h, z9.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x12,0x69,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a06912 <unknown>

bfmlal  za.s[w9, 6:7, vgx2], {z12.h, z13.h}, {z10.h, z11.h}  // 11000001, 10101010, 00101001, 10010011
// CHECK, INST: bfmlal  za.s[w9, 6:7, vgx2], { z12.h, z13.h }, { z10.h, z11.h }
// CHECK-ENCODING: [0x93,0x29,0xaa,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1aa2993 <unknown>

bfmlal  za.s[w9, 6:7], {z12.h - z13.h}, {z10.h - z11.h}  // 11000001-10101010-00101001-10010011
// CHECK, INST: bfmlal  za.s[w9, 6:7, vgx2], { z12.h, z13.h }, { z10.h, z11.h }
// CHECK-ENCODING: [0x93,0x29,0xaa,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1aa2993 <unknown>


bfmlal  za.s[w8, 0:1, vgx4], {z0.h - z3.h}, z0.h  // 11000001-00110000-00001000-00010000
// CHECK-INST: bfmlal  za.s[w8, 0:1, vgx4], { z0.h - z3.h }, z0.h
// CHECK-ENCODING: [0x10,0x08,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1300810 <unknown>

bfmlal  za.s[w8, 0:1], {z0.h - z3.h}, z0.h  // 11000001-00110000-00001000-00010000
// CHECK-INST: bfmlal  za.s[w8, 0:1, vgx4], { z0.h - z3.h }, z0.h
// CHECK-ENCODING: [0x10,0x08,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1300810 <unknown>

bfmlal  za.s[w10, 2:3, vgx4], {z10.h - z13.h}, z5.h  // 11000001-00110101-01001001-01010001
// CHECK-INST: bfmlal  za.s[w10, 2:3, vgx4], { z10.h - z13.h }, z5.h
// CHECK-ENCODING: [0x51,0x49,0x35,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1354951 <unknown>

bfmlal  za.s[w10, 2:3], {z10.h - z13.h}, z5.h  // 11000001-00110101-01001001-01010001
// CHECK-INST: bfmlal  za.s[w10, 2:3, vgx4], { z10.h - z13.h }, z5.h
// CHECK-ENCODING: [0x51,0x49,0x35,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1354951 <unknown>

bfmlal  za.s[w11, 6:7, vgx4], {z13.h - z16.h}, z8.h  // 11000001-00111000-01101001-10110011
// CHECK-INST: bfmlal  za.s[w11, 6:7, vgx4], { z13.h - z16.h }, z8.h
// CHECK-ENCODING: [0xb3,0x69,0x38,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13869b3 <unknown>

bfmlal  za.s[w11, 6:7], {z13.h - z16.h}, z8.h  // 11000001-00111000-01101001-10110011
// CHECK-INST: bfmlal  za.s[w11, 6:7, vgx4], { z13.h - z16.h }, z8.h
// CHECK-ENCODING: [0xb3,0x69,0x38,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13869b3 <unknown>

bfmlal  za.s[w11, 6:7, vgx4], {z31.h - z2.h}, z15.h  // 11000001-00111111-01101011-11110011
// CHECK-INST: bfmlal  za.s[w11, 6:7, vgx4], { z31.h, z0.h, z1.h, z2.h }, z15.h
// CHECK-ENCODING: [0xf3,0x6b,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13f6bf3 <unknown>

bfmlal  za.s[w11, 6:7], {z31.h - z2.h}, z15.h  // 11000001-00111111-01101011-11110011
// CHECK-INST: bfmlal  za.s[w11, 6:7, vgx4], { z31.h, z0.h, z1.h, z2.h }, z15.h
// CHECK-ENCODING: [0xf3,0x6b,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13f6bf3 <unknown>

bfmlal  za.s[w8, 2:3, vgx4], {z17.h - z20.h}, z0.h  // 11000001-00110000-00001010-00110001
// CHECK-INST: bfmlal  za.s[w8, 2:3, vgx4], { z17.h - z20.h }, z0.h
// CHECK-ENCODING: [0x31,0x0a,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1300a31 <unknown>

bfmlal  za.s[w8, 2:3], {z17.h - z20.h}, z0.h  // 11000001-00110000-00001010-00110001
// CHECK-INST: bfmlal  za.s[w8, 2:3, vgx4], { z17.h - z20.h }, z0.h
// CHECK-ENCODING: [0x31,0x0a,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1300a31 <unknown>

bfmlal  za.s[w8, 2:3, vgx4], {z1.h - z4.h}, z14.h  // 11000001-00111110-00001000-00110001
// CHECK-INST: bfmlal  za.s[w8, 2:3, vgx4], { z1.h - z4.h }, z14.h
// CHECK-ENCODING: [0x31,0x08,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e0831 <unknown>

bfmlal  za.s[w8, 2:3], {z1.h - z4.h}, z14.h  // 11000001-00111110-00001000-00110001
// CHECK-INST: bfmlal  za.s[w8, 2:3, vgx4], { z1.h - z4.h }, z14.h
// CHECK-ENCODING: [0x31,0x08,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e0831 <unknown>

bfmlal  za.s[w10, 0:1, vgx4], {z19.h - z22.h}, z4.h  // 11000001-00110100-01001010-01110000
// CHECK-INST: bfmlal  za.s[w10, 0:1, vgx4], { z19.h - z22.h }, z4.h
// CHECK-ENCODING: [0x70,0x4a,0x34,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1344a70 <unknown>

bfmlal  za.s[w10, 0:1], {z19.h - z22.h}, z4.h  // 11000001-00110100-01001010-01110000
// CHECK-INST: bfmlal  za.s[w10, 0:1, vgx4], { z19.h - z22.h }, z4.h
// CHECK-ENCODING: [0x70,0x4a,0x34,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1344a70 <unknown>

bfmlal  za.s[w8, 0:1, vgx4], {z12.h - z15.h}, z2.h  // 11000001-00110010-00001001-10010000
// CHECK-INST: bfmlal  za.s[w8, 0:1, vgx4], { z12.h - z15.h }, z2.h
// CHECK-ENCODING: [0x90,0x09,0x32,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1320990 <unknown>

bfmlal  za.s[w8, 0:1], {z12.h - z15.h}, z2.h  // 11000001-00110010-00001001-10010000
// CHECK-INST: bfmlal  za.s[w8, 0:1, vgx4], { z12.h - z15.h }, z2.h
// CHECK-ENCODING: [0x90,0x09,0x32,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1320990 <unknown>

bfmlal  za.s[w10, 2:3, vgx4], {z1.h - z4.h}, z10.h  // 11000001-00111010-01001000-00110001
// CHECK-INST: bfmlal  za.s[w10, 2:3, vgx4], { z1.h - z4.h }, z10.h
// CHECK-ENCODING: [0x31,0x48,0x3a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13a4831 <unknown>

bfmlal  za.s[w10, 2:3], {z1.h - z4.h}, z10.h  // 11000001-00111010-01001000-00110001
// CHECK-INST: bfmlal  za.s[w10, 2:3, vgx4], { z1.h - z4.h }, z10.h
// CHECK-ENCODING: [0x31,0x48,0x3a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13a4831 <unknown>

bfmlal  za.s[w8, 2:3, vgx4], {z22.h - z25.h}, z14.h  // 11000001-00111110-00001010-11010001
// CHECK-INST: bfmlal  za.s[w8, 2:3, vgx4], { z22.h - z25.h }, z14.h
// CHECK-ENCODING: [0xd1,0x0a,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e0ad1 <unknown>

bfmlal  za.s[w8, 2:3], {z22.h - z25.h}, z14.h  // 11000001-00111110-00001010-11010001
// CHECK-INST: bfmlal  za.s[w8, 2:3, vgx4], { z22.h - z25.h }, z14.h
// CHECK-ENCODING: [0xd1,0x0a,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e0ad1 <unknown>

bfmlal  za.s[w11, 4:5, vgx4], {z9.h - z12.h}, z1.h  // 11000001-00110001-01101001-00110010
// CHECK-INST: bfmlal  za.s[w11, 4:5, vgx4], { z9.h - z12.h }, z1.h
// CHECK-ENCODING: [0x32,0x69,0x31,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1316932 <unknown>

bfmlal  za.s[w11, 4:5], {z9.h - z12.h}, z1.h  // 11000001-00110001-01101001-00110010
// CHECK-INST: bfmlal  za.s[w11, 4:5, vgx4], { z9.h - z12.h }, z1.h
// CHECK-ENCODING: [0x32,0x69,0x31,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1316932 <unknown>

bfmlal  za.s[w9, 6:7, vgx4], {z12.h - z15.h}, z11.h  // 11000001-00111011-00101001-10010011
// CHECK-INST: bfmlal  za.s[w9, 6:7, vgx4], { z12.h - z15.h }, z11.h
// CHECK-ENCODING: [0x93,0x29,0x3b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13b2993 <unknown>

bfmlal  za.s[w9, 6:7], {z12.h - z15.h}, z11.h  // 11000001-00111011-00101001-10010011
// CHECK-INST: bfmlal  za.s[w9, 6:7, vgx4], { z12.h - z15.h }, z11.h
// CHECK-ENCODING: [0x93,0x29,0x3b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13b2993 <unknown>


bfmlal  za.s[w8, 0:1, vgx4], {z0.h - z3.h}, z0.h[0]  // 11000001-10010000-10010000-00010000
// CHECK-INST: bfmlal  za.s[w8, 0:1, vgx4], { z0.h - z3.h }, z0.h[0]
// CHECK-ENCODING: [0x10,0x90,0x90,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1909010 <unknown>

bfmlal  za.s[w8, 0:1], {z0.h - z3.h}, z0.h[0]  // 11000001-10010000-10010000-00010000
// CHECK-INST: bfmlal  za.s[w8, 0:1, vgx4], { z0.h - z3.h }, z0.h[0]
// CHECK-ENCODING: [0x10,0x90,0x90,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1909010 <unknown>

bfmlal  za.s[w10, 2:3, vgx4], {z8.h - z11.h}, z5.h[3]  // 11000001-10010101-11010101-00010101
// CHECK-INST: bfmlal  za.s[w10, 2:3, vgx4], { z8.h - z11.h }, z5.h[3]
// CHECK-ENCODING: [0x15,0xd5,0x95,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c195d515 <unknown>

bfmlal  za.s[w10, 2:3], {z8.h - z11.h}, z5.h[3]  // 11000001-10010101-11010101-00010101
// CHECK-INST: bfmlal  za.s[w10, 2:3, vgx4], { z8.h - z11.h }, z5.h[3]
// CHECK-ENCODING: [0x15,0xd5,0x95,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c195d515 <unknown>

bfmlal  za.s[w11, 6:7, vgx4], {z12.h - z15.h}, z8.h[7]  // 11000001-10011000-11111101-10010111
// CHECK-INST: bfmlal  za.s[w11, 6:7, vgx4], { z12.h - z15.h }, z8.h[7]
// CHECK-ENCODING: [0x97,0xfd,0x98,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c198fd97 <unknown>

bfmlal  za.s[w11, 6:7], {z12.h - z15.h}, z8.h[7]  // 11000001-10011000-11111101-10010111
// CHECK-INST: bfmlal  za.s[w11, 6:7, vgx4], { z12.h - z15.h }, z8.h[7]
// CHECK-ENCODING: [0x97,0xfd,0x98,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c198fd97 <unknown>

bfmlal  za.s[w11, 6:7, vgx4], {z28.h - z31.h}, z15.h[7]  // 11000001-10011111-11111111-10010111
// CHECK-INST: bfmlal  za.s[w11, 6:7, vgx4], { z28.h - z31.h }, z15.h[7]
// CHECK-ENCODING: [0x97,0xff,0x9f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c19fff97 <unknown>

bfmlal  za.s[w11, 6:7], {z28.h - z31.h}, z15.h[7]  // 11000001-10011111-11111111-10010111
// CHECK-INST: bfmlal  za.s[w11, 6:7, vgx4], { z28.h - z31.h }, z15.h[7]
// CHECK-ENCODING: [0x97,0xff,0x9f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c19fff97 <unknown>

bfmlal  za.s[w8, 2:3, vgx4], {z16.h - z19.h}, z0.h[7]  // 11000001-10010000-10011110-00010101
// CHECK-INST: bfmlal  za.s[w8, 2:3, vgx4], { z16.h - z19.h }, z0.h[7]
// CHECK-ENCODING: [0x15,0x9e,0x90,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1909e15 <unknown>

bfmlal  za.s[w8, 2:3], {z16.h - z19.h}, z0.h[7]  // 11000001-10010000-10011110-00010101
// CHECK-INST: bfmlal  za.s[w8, 2:3, vgx4], { z16.h - z19.h }, z0.h[7]
// CHECK-ENCODING: [0x15,0x9e,0x90,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1909e15 <unknown>

bfmlal  za.s[w8, 2:3, vgx4], {z0.h - z3.h}, z14.h[2]  // 11000001-10011110-10010100-00010001
// CHECK-INST: bfmlal  za.s[w8, 2:3, vgx4], { z0.h - z3.h }, z14.h[2]
// CHECK-ENCODING: [0x11,0x94,0x9e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c19e9411 <unknown>

bfmlal  za.s[w8, 2:3], {z0.h - z3.h}, z14.h[2]  // 11000001-10011110-10010100-00010001
// CHECK-INST: bfmlal  za.s[w8, 2:3, vgx4], { z0.h - z3.h }, z14.h[2]
// CHECK-ENCODING: [0x11,0x94,0x9e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c19e9411 <unknown>

bfmlal  za.s[w10, 0:1, vgx4], {z16.h - z19.h}, z4.h[2]  // 11000001-10010100-11010110-00010000
// CHECK-INST: bfmlal  za.s[w10, 0:1, vgx4], { z16.h - z19.h }, z4.h[2]
// CHECK-ENCODING: [0x10,0xd6,0x94,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c194d610 <unknown>

bfmlal  za.s[w10, 0:1], {z16.h - z19.h}, z4.h[2]  // 11000001-10010100-11010110-00010000
// CHECK-INST: bfmlal  za.s[w10, 0:1, vgx4], { z16.h - z19.h }, z4.h[2]
// CHECK-ENCODING: [0x10,0xd6,0x94,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c194d610 <unknown>

bfmlal  za.s[w8, 0:1, vgx4], {z12.h - z15.h}, z2.h[4]  // 11000001-10010010-10011001-10010000
// CHECK-INST: bfmlal  za.s[w8, 0:1, vgx4], { z12.h - z15.h }, z2.h[4]
// CHECK-ENCODING: [0x90,0x99,0x92,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1929990 <unknown>

bfmlal  za.s[w8, 0:1], {z12.h - z15.h}, z2.h[4]  // 11000001-10010010-10011001-10010000
// CHECK-INST: bfmlal  za.s[w8, 0:1, vgx4], { z12.h - z15.h }, z2.h[4]
// CHECK-ENCODING: [0x90,0x99,0x92,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1929990 <unknown>

bfmlal  za.s[w10, 2:3, vgx4], {z0.h - z3.h}, z10.h[4]  // 11000001-10011010-11011000-00010001
// CHECK-INST: bfmlal  za.s[w10, 2:3, vgx4], { z0.h - z3.h }, z10.h[4]
// CHECK-ENCODING: [0x11,0xd8,0x9a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c19ad811 <unknown>

bfmlal  za.s[w10, 2:3], {z0.h - z3.h}, z10.h[4]  // 11000001-10011010-11011000-00010001
// CHECK-INST: bfmlal  za.s[w10, 2:3, vgx4], { z0.h - z3.h }, z10.h[4]
// CHECK-ENCODING: [0x11,0xd8,0x9a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c19ad811 <unknown>

bfmlal  za.s[w8, 2:3, vgx4], {z20.h - z23.h}, z14.h[5]  // 11000001-10011110-10011010-10010101
// CHECK-INST: bfmlal  za.s[w8, 2:3, vgx4], { z20.h - z23.h }, z14.h[5]
// CHECK-ENCODING: [0x95,0x9a,0x9e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c19e9a95 <unknown>

bfmlal  za.s[w8, 2:3], {z20.h - z23.h}, z14.h[5]  // 11000001-10011110-10011010-10010101
// CHECK-INST: bfmlal  za.s[w8, 2:3, vgx4], { z20.h - z23.h }, z14.h[5]
// CHECK-ENCODING: [0x95,0x9a,0x9e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c19e9a95 <unknown>

bfmlal  za.s[w11, 4:5, vgx4], {z8.h - z11.h}, z1.h[2]  // 11000001-10010001-11110101-00010010
// CHECK-INST: bfmlal  za.s[w11, 4:5, vgx4], { z8.h - z11.h }, z1.h[2]
// CHECK-ENCODING: [0x12,0xf5,0x91,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c191f512 <unknown>

bfmlal  za.s[w11, 4:5], {z8.h - z11.h}, z1.h[2]  // 11000001-10010001-11110101-00010010
// CHECK-INST: bfmlal  za.s[w11, 4:5, vgx4], { z8.h - z11.h }, z1.h[2]
// CHECK-ENCODING: [0x12,0xf5,0x91,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c191f512 <unknown>

bfmlal  za.s[w9, 6:7, vgx4], {z12.h - z15.h}, z11.h[5]  // 11000001-10011011-10111001-10010111
// CHECK-INST: bfmlal  za.s[w9, 6:7, vgx4], { z12.h - z15.h }, z11.h[5]
// CHECK-ENCODING: [0x97,0xb9,0x9b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c19bb997 <unknown>

bfmlal  za.s[w9, 6:7], {z12.h - z15.h}, z11.h[5]  // 11000001-10011011-10111001-10010111
// CHECK-INST: bfmlal  za.s[w9, 6:7, vgx4], { z12.h - z15.h }, z11.h[5]
// CHECK-ENCODING: [0x97,0xb9,0x9b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c19bb997 <unknown>


bfmlal  za.s[w8, 0:1, vgx4], {z0.h - z3.h}, {z0.h - z3.h}  // 11000001-10100001-00001000-00010000
// CHECK-INST: bfmlal  za.s[w8, 0:1, vgx4], { z0.h - z3.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x10,0x08,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a10810 <unknown>

bfmlal  za.s[w8, 0:1], {z0.h - z3.h}, {z0.h - z3.h}  // 11000001-10100001-00001000-00010000
// CHECK-INST: bfmlal  za.s[w8, 0:1, vgx4], { z0.h - z3.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x10,0x08,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a10810 <unknown>

bfmlal  za.s[w10, 2:3, vgx4], {z8.h - z11.h}, {z20.h - z23.h}  // 11000001-10110101-01001001-00010001
// CHECK-INST: bfmlal  za.s[w10, 2:3, vgx4], { z8.h - z11.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x11,0x49,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b54911 <unknown>

bfmlal  za.s[w10, 2:3], {z8.h - z11.h}, {z20.h - z23.h}  // 11000001-10110101-01001001-00010001
// CHECK-INST: bfmlal  za.s[w10, 2:3, vgx4], { z8.h - z11.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x11,0x49,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b54911 <unknown>

bfmlal  za.s[w11, 6:7, vgx4], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-10101001-01101001-10010011
// CHECK-INST: bfmlal  za.s[w11, 6:7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x93,0x69,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a96993 <unknown>

bfmlal  za.s[w11, 6:7], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-10101001-01101001-10010011
// CHECK-INST: bfmlal  za.s[w11, 6:7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x93,0x69,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a96993 <unknown>

bfmlal  za.s[w11, 6:7, vgx4], {z28.h - z31.h}, {z28.h - z31.h}  // 11000001-10111101-01101011-10010011
// CHECK-INST: bfmlal  za.s[w11, 6:7, vgx4], { z28.h - z31.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x93,0x6b,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd6b93 <unknown>

bfmlal  za.s[w11, 6:7], {z28.h - z31.h}, {z28.h - z31.h}  // 11000001-10111101-01101011-10010011
// CHECK-INST: bfmlal  za.s[w11, 6:7, vgx4], { z28.h - z31.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x93,0x6b,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd6b93 <unknown>

bfmlal  za.s[w8, 2:3, vgx4], {z16.h - z19.h}, {z16.h - z19.h}  // 11000001-10110001-00001010-00010001
// CHECK-INST: bfmlal  za.s[w8, 2:3, vgx4], { z16.h - z19.h }, { z16.h - z19.h }
// CHECK-ENCODING: [0x11,0x0a,0xb1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b10a11 <unknown>

bfmlal  za.s[w8, 2:3], {z16.h - z19.h}, {z16.h - z19.h}  // 11000001-10110001-00001010-00010001
// CHECK-INST: bfmlal  za.s[w8, 2:3, vgx4], { z16.h - z19.h }, { z16.h - z19.h }
// CHECK-ENCODING: [0x11,0x0a,0xb1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b10a11 <unknown>

bfmlal  za.s[w8, 2:3, vgx4], {z0.h - z3.h}, {z28.h - z31.h}  // 11000001-10111101-00001000-00010001
// CHECK-INST: bfmlal  za.s[w8, 2:3, vgx4], { z0.h - z3.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x11,0x08,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd0811 <unknown>

bfmlal  za.s[w8, 2:3], {z0.h - z3.h}, {z28.h - z31.h}  // 11000001-10111101-00001000-00010001
// CHECK-INST: bfmlal  za.s[w8, 2:3, vgx4], { z0.h - z3.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x11,0x08,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd0811 <unknown>

bfmlal  za.s[w10, 0:1, vgx4], {z16.h - z19.h}, {z20.h - z23.h}  // 11000001-10110101-01001010-00010000
// CHECK-INST: bfmlal  za.s[w10, 0:1, vgx4], { z16.h - z19.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x10,0x4a,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b54a10 <unknown>

bfmlal  za.s[w10, 0:1], {z16.h - z19.h}, {z20.h - z23.h}  // 11000001-10110101-01001010-00010000
// CHECK-INST: bfmlal  za.s[w10, 0:1, vgx4], { z16.h - z19.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x10,0x4a,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b54a10 <unknown>

bfmlal  za.s[w8, 0:1, vgx4], {z12.h - z15.h}, {z0.h - z3.h}  // 11000001-10100001-00001001-10010000
// CHECK-INST: bfmlal  za.s[w8, 0:1, vgx4], { z12.h - z15.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x90,0x09,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a10990 <unknown>

bfmlal  za.s[w8, 0:1], {z12.h - z15.h}, {z0.h - z3.h}  // 11000001-10100001-00001001-10010000
// CHECK-INST: bfmlal  za.s[w8, 0:1, vgx4], { z12.h - z15.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x90,0x09,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a10990 <unknown>

bfmlal  za.s[w10, 2:3, vgx4], {z0.h - z3.h}, {z24.h - z27.h}  // 11000001-10111001-01001000-00010001
// CHECK-INST: bfmlal  za.s[w10, 2:3, vgx4], { z0.h - z3.h }, { z24.h - z27.h }
// CHECK-ENCODING: [0x11,0x48,0xb9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b94811 <unknown>

bfmlal  za.s[w10, 2:3], {z0.h - z3.h}, {z24.h - z27.h}  // 11000001-10111001-01001000-00010001
// CHECK-INST: bfmlal  za.s[w10, 2:3, vgx4], { z0.h - z3.h }, { z24.h - z27.h }
// CHECK-ENCODING: [0x11,0x48,0xb9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b94811 <unknown>

bfmlal  za.s[w8, 2:3, vgx4], {z20.h - z23.h}, {z28.h - z31.h}  // 11000001-10111101-00001010-10010001
// CHECK-INST: bfmlal  za.s[w8, 2:3, vgx4], { z20.h - z23.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x91,0x0a,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd0a91 <unknown>

bfmlal  za.s[w8, 2:3], {z20.h - z23.h}, {z28.h - z31.h}  // 11000001-10111101-00001010-10010001
// CHECK-INST: bfmlal  za.s[w8, 2:3, vgx4], { z20.h - z23.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x91,0x0a,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd0a91 <unknown>

bfmlal  za.s[w11, 4:5, vgx4], {z8.h - z11.h}, {z0.h - z3.h}  // 11000001-10100001-01101001-00010010
// CHECK-INST: bfmlal  za.s[w11, 4:5, vgx4], { z8.h - z11.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x12,0x69,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a16912 <unknown>

bfmlal  za.s[w11, 4:5], {z8.h - z11.h}, {z0.h - z3.h}  // 11000001-10100001-01101001-00010010
// CHECK-INST: bfmlal  za.s[w11, 4:5, vgx4], { z8.h - z11.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x12,0x69,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a16912 <unknown>

bfmlal  za.s[w9, 6:7, vgx4], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-10101001-00101001-10010011
// CHECK-INST: bfmlal  za.s[w9, 6:7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x93,0x29,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a92993 <unknown>

bfmlal  za.s[w9, 6:7], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-10101001-00101001-10010011
// CHECK-INST: bfmlal  za.s[w9, 6:7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x93,0x29,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a92993 <unknown>

