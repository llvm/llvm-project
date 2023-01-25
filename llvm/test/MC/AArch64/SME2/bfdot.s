// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


bfdot   za.s[w8, 0, vgx2], {z0.h, z1.h}, z0.h  // 11000001-00100000-00010000-00010000
// CHECK-INST: bfdot   za.s[w8, 0, vgx2], { z0.h, z1.h }, z0.h
// CHECK-ENCODING: [0x10,0x10,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1201010 <unknown>

bfdot   za.s[w8, 0], {z0.h, z1.h}, z0.h  // 11000001-00100000-00010000-00010000
// CHECK-INST: bfdot   za.s[w8, 0, vgx2], { z0.h, z1.h }, z0.h
// CHECK-ENCODING: [0x10,0x10,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1201010 <unknown>

bfdot   za.s[w10, 5, vgx2], {z10.h, z11.h}, z5.h  // 11000001-00100101-01010001-01010101
// CHECK-INST: bfdot   za.s[w10, 5, vgx2], { z10.h, z11.h }, z5.h
// CHECK-ENCODING: [0x55,0x51,0x25,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1255155 <unknown>

bfdot   za.s[w10, 5], {z10.h, z11.h}, z5.h  // 11000001-00100101-01010001-01010101
// CHECK-INST: bfdot   za.s[w10, 5, vgx2], { z10.h, z11.h }, z5.h
// CHECK-ENCODING: [0x55,0x51,0x25,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1255155 <unknown>

bfdot   za.s[w11, 7, vgx2], {z13.h, z14.h}, z8.h  // 11000001-00101000-01110001-10110111
// CHECK-INST: bfdot   za.s[w11, 7, vgx2], { z13.h, z14.h }, z8.h
// CHECK-ENCODING: [0xb7,0x71,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12871b7 <unknown>

bfdot   za.s[w11, 7], {z13.h, z14.h}, z8.h  // 11000001-00101000-01110001-10110111
// CHECK-INST: bfdot   za.s[w11, 7, vgx2], { z13.h, z14.h }, z8.h
// CHECK-ENCODING: [0xb7,0x71,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12871b7 <unknown>

bfdot   za.s[w11, 7, vgx2], {z31.h, z0.h}, z15.h  // 11000001-00101111-01110011-11110111
// CHECK-INST: bfdot   za.s[w11, 7, vgx2], { z31.h, z0.h }, z15.h
// CHECK-ENCODING: [0xf7,0x73,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12f73f7 <unknown>

bfdot   za.s[w11, 7], {z31.h, z0.h}, z15.h  // 11000001-00101111-01110011-11110111
// CHECK-INST: bfdot   za.s[w11, 7, vgx2], { z31.h, z0.h }, z15.h
// CHECK-ENCODING: [0xf7,0x73,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12f73f7 <unknown>

bfdot   za.s[w8, 5, vgx2], {z17.h, z18.h}, z0.h  // 11000001-00100000-00010010-00110101
// CHECK-INST: bfdot   za.s[w8, 5, vgx2], { z17.h, z18.h }, z0.h
// CHECK-ENCODING: [0x35,0x12,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1201235 <unknown>

bfdot   za.s[w8, 5], {z17.h, z18.h}, z0.h  // 11000001-00100000-00010010-00110101
// CHECK-INST: bfdot   za.s[w8, 5, vgx2], { z17.h, z18.h }, z0.h
// CHECK-ENCODING: [0x35,0x12,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1201235 <unknown>

bfdot   za.s[w8, 1, vgx2], {z1.h, z2.h}, z14.h  // 11000001-00101110-00010000-00110001
// CHECK-INST: bfdot   za.s[w8, 1, vgx2], { z1.h, z2.h }, z14.h
// CHECK-ENCODING: [0x31,0x10,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e1031 <unknown>

bfdot   za.s[w8, 1], {z1.h, z2.h}, z14.h  // 11000001-00101110-00010000-00110001
// CHECK-INST: bfdot   za.s[w8, 1, vgx2], { z1.h, z2.h }, z14.h
// CHECK-ENCODING: [0x31,0x10,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e1031 <unknown>

bfdot   za.s[w10, 0, vgx2], {z19.h, z20.h}, z4.h  // 11000001-00100100-01010010-01110000
// CHECK-INST: bfdot   za.s[w10, 0, vgx2], { z19.h, z20.h }, z4.h
// CHECK-ENCODING: [0x70,0x52,0x24,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1245270 <unknown>

bfdot   za.s[w10, 0], {z19.h, z20.h}, z4.h  // 11000001-00100100-01010010-01110000
// CHECK-INST: bfdot   za.s[w10, 0, vgx2], { z19.h, z20.h }, z4.h
// CHECK-ENCODING: [0x70,0x52,0x24,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1245270 <unknown>

bfdot   za.s[w8, 0, vgx2], {z12.h, z13.h}, z2.h  // 11000001-00100010-00010001-10010000
// CHECK-INST: bfdot   za.s[w8, 0, vgx2], { z12.h, z13.h }, z2.h
// CHECK-ENCODING: [0x90,0x11,0x22,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1221190 <unknown>

bfdot   za.s[w8, 0], {z12.h, z13.h}, z2.h  // 11000001-00100010-00010001-10010000
// CHECK-INST: bfdot   za.s[w8, 0, vgx2], { z12.h, z13.h }, z2.h
// CHECK-ENCODING: [0x90,0x11,0x22,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1221190 <unknown>

bfdot   za.s[w10, 1, vgx2], {z1.h, z2.h}, z10.h  // 11000001-00101010-01010000-00110001
// CHECK-INST: bfdot   za.s[w10, 1, vgx2], { z1.h, z2.h }, z10.h
// CHECK-ENCODING: [0x31,0x50,0x2a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12a5031 <unknown>

bfdot   za.s[w10, 1], {z1.h, z2.h}, z10.h  // 11000001-00101010-01010000-00110001
// CHECK-INST: bfdot   za.s[w10, 1, vgx2], { z1.h, z2.h }, z10.h
// CHECK-ENCODING: [0x31,0x50,0x2a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12a5031 <unknown>

bfdot   za.s[w8, 5, vgx2], {z22.h, z23.h}, z14.h  // 11000001-00101110-00010010-11010101
// CHECK-INST: bfdot   za.s[w8, 5, vgx2], { z22.h, z23.h }, z14.h
// CHECK-ENCODING: [0xd5,0x12,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e12d5 <unknown>

bfdot   za.s[w8, 5], {z22.h, z23.h}, z14.h  // 11000001-00101110-00010010-11010101
// CHECK-INST: bfdot   za.s[w8, 5, vgx2], { z22.h, z23.h }, z14.h
// CHECK-ENCODING: [0xd5,0x12,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e12d5 <unknown>

bfdot   za.s[w11, 2, vgx2], {z9.h, z10.h}, z1.h  // 11000001-00100001-01110001-00110010
// CHECK-INST: bfdot   za.s[w11, 2, vgx2], { z9.h, z10.h }, z1.h
// CHECK-ENCODING: [0x32,0x71,0x21,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1217132 <unknown>

bfdot   za.s[w11, 2], {z9.h, z10.h}, z1.h  // 11000001-00100001-01110001-00110010
// CHECK-INST: bfdot   za.s[w11, 2, vgx2], { z9.h, z10.h }, z1.h
// CHECK-ENCODING: [0x32,0x71,0x21,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1217132 <unknown>

bfdot   za.s[w9, 7, vgx2], {z12.h, z13.h}, z11.h  // 11000001-00101011-00110001-10010111
// CHECK-INST: bfdot   za.s[w9, 7, vgx2], { z12.h, z13.h }, z11.h
// CHECK-ENCODING: [0x97,0x31,0x2b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12b3197 <unknown>

bfdot   za.s[w9, 7], {z12.h, z13.h}, z11.h  // 11000001-00101011-00110001-10010111
// CHECK-INST: bfdot   za.s[w9, 7, vgx2], { z12.h, z13.h }, z11.h
// CHECK-ENCODING: [0x97,0x31,0x2b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12b3197 <unknown>


bfdot   za.s[w8, 0, vgx2], {z0.h, z1.h}, z0.h[0]  // 11000001-01010000-00010000-00011000
// CHECK-INST: bfdot   za.s[w8, 0, vgx2], { z0.h, z1.h }, z0.h[0]
// CHECK-ENCODING: [0x18,0x10,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1501018 <unknown>

bfdot   za.s[w8, 0], {z0.h, z1.h}, z0.h[0]  // 11000001-01010000-00010000-00011000
// CHECK-INST: bfdot   za.s[w8, 0, vgx2], { z0.h, z1.h }, z0.h[0]
// CHECK-ENCODING: [0x18,0x10,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1501018 <unknown>

bfdot   za.s[w10, 5, vgx2], {z10.h, z11.h}, z5.h[1]  // 11000001-01010101-01010101-01011101
// CHECK-INST: bfdot   za.s[w10, 5, vgx2], { z10.h, z11.h }, z5.h[1]
// CHECK-ENCODING: [0x5d,0x55,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c155555d <unknown>

bfdot   za.s[w10, 5], {z10.h, z11.h}, z5.h[1]  // 11000001-01010101-01010101-01011101
// CHECK-INST: bfdot   za.s[w10, 5, vgx2], { z10.h, z11.h }, z5.h[1]
// CHECK-ENCODING: [0x5d,0x55,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c155555d <unknown>

bfdot   za.s[w11, 7, vgx2], {z12.h, z13.h}, z8.h[3]  // 11000001-01011000-01111101-10011111
// CHECK-INST: bfdot   za.s[w11, 7, vgx2], { z12.h, z13.h }, z8.h[3]
// CHECK-ENCODING: [0x9f,0x7d,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1587d9f <unknown>

bfdot   za.s[w11, 7], {z12.h, z13.h}, z8.h[3]  // 11000001-01011000-01111101-10011111
// CHECK-INST: bfdot   za.s[w11, 7, vgx2], { z12.h, z13.h }, z8.h[3]
// CHECK-ENCODING: [0x9f,0x7d,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1587d9f <unknown>

bfdot   za.s[w11, 7, vgx2], {z30.h, z31.h}, z15.h[3]  // 11000001-01011111-01111111-11011111
// CHECK-INST: bfdot   za.s[w11, 7, vgx2], { z30.h, z31.h }, z15.h[3]
// CHECK-ENCODING: [0xdf,0x7f,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15f7fdf <unknown>

bfdot   za.s[w11, 7], {z30.h, z31.h}, z15.h[3]  // 11000001-01011111-01111111-11011111
// CHECK-INST: bfdot   za.s[w11, 7, vgx2], { z30.h, z31.h }, z15.h[3]
// CHECK-ENCODING: [0xdf,0x7f,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15f7fdf <unknown>

bfdot   za.s[w8, 5, vgx2], {z16.h, z17.h}, z0.h[3]  // 11000001-01010000-00011110-00011101
// CHECK-INST: bfdot   za.s[w8, 5, vgx2], { z16.h, z17.h }, z0.h[3]
// CHECK-ENCODING: [0x1d,0x1e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1501e1d <unknown>

bfdot   za.s[w8, 5], {z16.h, z17.h}, z0.h[3]  // 11000001-01010000-00011110-00011101
// CHECK-INST: bfdot   za.s[w8, 5, vgx2], { z16.h, z17.h }, z0.h[3]
// CHECK-ENCODING: [0x1d,0x1e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1501e1d <unknown>

bfdot   za.s[w8, 1, vgx2], {z0.h, z1.h}, z14.h[1]  // 11000001-01011110-00010100-00011001
// CHECK-INST: bfdot   za.s[w8, 1, vgx2], { z0.h, z1.h }, z14.h[1]
// CHECK-ENCODING: [0x19,0x14,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e1419 <unknown>

bfdot   za.s[w8, 1], {z0.h, z1.h}, z14.h[1]  // 11000001-01011110-00010100-00011001
// CHECK-INST: bfdot   za.s[w8, 1, vgx2], { z0.h, z1.h }, z14.h[1]
// CHECK-ENCODING: [0x19,0x14,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e1419 <unknown>

bfdot   za.s[w10, 0, vgx2], {z18.h, z19.h}, z4.h[1]  // 11000001-01010100-01010110-01011000
// CHECK-INST: bfdot   za.s[w10, 0, vgx2], { z18.h, z19.h }, z4.h[1]
// CHECK-ENCODING: [0x58,0x56,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1545658 <unknown>

bfdot   za.s[w10, 0], {z18.h, z19.h}, z4.h[1]  // 11000001-01010100-01010110-01011000
// CHECK-INST: bfdot   za.s[w10, 0, vgx2], { z18.h, z19.h }, z4.h[1]
// CHECK-ENCODING: [0x58,0x56,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1545658 <unknown>

bfdot   za.s[w8, 0, vgx2], {z12.h, z13.h}, z2.h[2]  // 11000001-01010010-00011001-10011000
// CHECK-INST: bfdot   za.s[w8, 0, vgx2], { z12.h, z13.h }, z2.h[2]
// CHECK-ENCODING: [0x98,0x19,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1521998 <unknown>

bfdot   za.s[w8, 0], {z12.h, z13.h}, z2.h[2]  // 11000001-01010010-00011001-10011000
// CHECK-INST: bfdot   za.s[w8, 0, vgx2], { z12.h, z13.h }, z2.h[2]
// CHECK-ENCODING: [0x98,0x19,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1521998 <unknown>

bfdot   za.s[w10, 1, vgx2], {z0.h, z1.h}, z10.h[2]  // 11000001-01011010-01011000-00011001
// CHECK-INST: bfdot   za.s[w10, 1, vgx2], { z0.h, z1.h }, z10.h[2]
// CHECK-ENCODING: [0x19,0x58,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15a5819 <unknown>

bfdot   za.s[w10, 1], {z0.h, z1.h}, z10.h[2]  // 11000001-01011010-01011000-00011001
// CHECK-INST: bfdot   za.s[w10, 1, vgx2], { z0.h, z1.h }, z10.h[2]
// CHECK-ENCODING: [0x19,0x58,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15a5819 <unknown>

bfdot   za.s[w8, 5, vgx2], {z22.h, z23.h}, z14.h[2]  // 11000001-01011110-00011010-11011101
// CHECK-INST: bfdot   za.s[w8, 5, vgx2], { z22.h, z23.h }, z14.h[2]
// CHECK-ENCODING: [0xdd,0x1a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e1add <unknown>

bfdot   za.s[w8, 5], {z22.h, z23.h}, z14.h[2]  // 11000001-01011110-00011010-11011101
// CHECK-INST: bfdot   za.s[w8, 5, vgx2], { z22.h, z23.h }, z14.h[2]
// CHECK-ENCODING: [0xdd,0x1a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e1add <unknown>

bfdot   za.s[w11, 2, vgx2], {z8.h, z9.h}, z1.h[1]  // 11000001-01010001-01110101-00011010
// CHECK-INST: bfdot   za.s[w11, 2, vgx2], { z8.h, z9.h }, z1.h[1]
// CHECK-ENCODING: [0x1a,0x75,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c151751a <unknown>

bfdot   za.s[w11, 2], {z8.h, z9.h}, z1.h[1]  // 11000001-01010001-01110101-00011010
// CHECK-INST: bfdot   za.s[w11, 2, vgx2], { z8.h, z9.h }, z1.h[1]
// CHECK-ENCODING: [0x1a,0x75,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c151751a <unknown>

bfdot   za.s[w9, 7, vgx2], {z12.h, z13.h}, z11.h[2]  // 11000001-01011011-00111001-10011111
// CHECK-INST: bfdot   za.s[w9, 7, vgx2], { z12.h, z13.h }, z11.h[2]
// CHECK-ENCODING: [0x9f,0x39,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15b399f <unknown>

bfdot   za.s[w9, 7], {z12.h, z13.h}, z11.h[2]  // 11000001-01011011-00111001-10011111
// CHECK-INST: bfdot   za.s[w9, 7, vgx2], { z12.h, z13.h }, z11.h[2]
// CHECK-ENCODING: [0x9f,0x39,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15b399f <unknown>


bfdot   za.s[w8, 0, vgx2], {z0.h, z1.h}, {z0.h, z1.h}  // 11000001-10100000-00010000-00010000
// CHECK-INST: bfdot   za.s[w8, 0, vgx2], { z0.h, z1.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x10,0x10,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a01010 <unknown>

bfdot   za.s[w8, 0], {z0.h, z1.h}, {z0.h, z1.h}  // 11000001-10100000-00010000-00010000
// CHECK-INST: bfdot   za.s[w8, 0, vgx2], { z0.h, z1.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x10,0x10,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a01010 <unknown>

bfdot   za.s[w10, 5, vgx2], {z10.h, z11.h}, {z20.h, z21.h}  // 11000001-10110100-01010001-01010101
// CHECK-INST: bfdot   za.s[w10, 5, vgx2], { z10.h, z11.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x55,0x51,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b45155 <unknown>

bfdot   za.s[w10, 5], {z10.h, z11.h}, {z20.h, z21.h}  // 11000001-10110100-01010001-01010101
// CHECK-INST: bfdot   za.s[w10, 5, vgx2], { z10.h, z11.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x55,0x51,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b45155 <unknown>

bfdot   za.s[w11, 7, vgx2], {z12.h, z13.h}, {z8.h, z9.h}  // 11000001-10101000-01110001-10010111
// CHECK-INST: bfdot   za.s[w11, 7, vgx2], { z12.h, z13.h }, { z8.h, z9.h }
// CHECK-ENCODING: [0x97,0x71,0xa8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a87197 <unknown>

bfdot   za.s[w11, 7], {z12.h, z13.h}, {z8.h, z9.h}  // 11000001-10101000-01110001-10010111
// CHECK-INST: bfdot   za.s[w11, 7, vgx2], { z12.h, z13.h }, { z8.h, z9.h }
// CHECK-ENCODING: [0x97,0x71,0xa8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a87197 <unknown>

bfdot   za.s[w11, 7, vgx2], {z30.h, z31.h}, {z30.h, z31.h}  // 11000001-10111110-01110011-11010111
// CHECK-INST: bfdot   za.s[w11, 7, vgx2], { z30.h, z31.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xd7,0x73,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be73d7 <unknown>

bfdot   za.s[w11, 7], {z30.h, z31.h}, {z30.h, z31.h}  // 11000001-10111110-01110011-11010111
// CHECK-INST: bfdot   za.s[w11, 7, vgx2], { z30.h, z31.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xd7,0x73,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be73d7 <unknown>

bfdot   za.s[w8, 5, vgx2], {z16.h, z17.h}, {z16.h, z17.h}  // 11000001-10110000-00010010-00010101
// CHECK-INST: bfdot   za.s[w8, 5, vgx2], { z16.h, z17.h }, { z16.h, z17.h }
// CHECK-ENCODING: [0x15,0x12,0xb0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b01215 <unknown>

bfdot   za.s[w8, 5], {z16.h, z17.h}, {z16.h, z17.h}  // 11000001-10110000-00010010-00010101
// CHECK-INST: bfdot   za.s[w8, 5, vgx2], { z16.h, z17.h }, { z16.h, z17.h }
// CHECK-ENCODING: [0x15,0x12,0xb0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b01215 <unknown>

bfdot   za.s[w8, 1, vgx2], {z0.h, z1.h}, {z30.h, z31.h}  // 11000001-10111110-00010000-00010001
// CHECK-INST: bfdot   za.s[w8, 1, vgx2], { z0.h, z1.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0x11,0x10,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be1011 <unknown>

bfdot   za.s[w8, 1], {z0.h, z1.h}, {z30.h, z31.h}  // 11000001-10111110-00010000-00010001
// CHECK-INST: bfdot   za.s[w8, 1, vgx2], { z0.h, z1.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0x11,0x10,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be1011 <unknown>

bfdot   za.s[w10, 0, vgx2], {z18.h, z19.h}, {z20.h, z21.h}  // 11000001-10110100-01010010-01010000
// CHECK-INST: bfdot   za.s[w10, 0, vgx2], { z18.h, z19.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x50,0x52,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b45250 <unknown>

bfdot   za.s[w10, 0], {z18.h, z19.h}, {z20.h, z21.h}  // 11000001-10110100-01010010-01010000
// CHECK-INST: bfdot   za.s[w10, 0, vgx2], { z18.h, z19.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x50,0x52,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b45250 <unknown>

bfdot   za.s[w8, 0, vgx2], {z12.h, z13.h}, {z2.h, z3.h}  // 11000001-10100010-00010001-10010000
// CHECK-INST: bfdot   za.s[w8, 0, vgx2], { z12.h, z13.h }, { z2.h, z3.h }
// CHECK-ENCODING: [0x90,0x11,0xa2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a21190 <unknown>

bfdot   za.s[w8, 0], {z12.h, z13.h}, {z2.h, z3.h}  // 11000001-10100010-00010001-10010000
// CHECK-INST: bfdot   za.s[w8, 0, vgx2], { z12.h, z13.h }, { z2.h, z3.h }
// CHECK-ENCODING: [0x90,0x11,0xa2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a21190 <unknown>

bfdot   za.s[w10, 1, vgx2], {z0.h, z1.h}, {z26.h, z27.h}  // 11000001-10111010-01010000-00010001
// CHECK-INST: bfdot   za.s[w10, 1, vgx2], { z0.h, z1.h }, { z26.h, z27.h }
// CHECK-ENCODING: [0x11,0x50,0xba,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ba5011 <unknown>

bfdot   za.s[w10, 1], {z0.h, z1.h}, {z26.h, z27.h}  // 11000001-10111010-01010000-00010001
// CHECK-INST: bfdot   za.s[w10, 1, vgx2], { z0.h, z1.h }, { z26.h, z27.h }
// CHECK-ENCODING: [0x11,0x50,0xba,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ba5011 <unknown>

bfdot   za.s[w8, 5, vgx2], {z22.h, z23.h}, {z30.h, z31.h}  // 11000001-10111110-00010010-11010101
// CHECK-INST: bfdot   za.s[w8, 5, vgx2], { z22.h, z23.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xd5,0x12,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be12d5 <unknown>

bfdot   za.s[w8, 5], {z22.h, z23.h}, {z30.h, z31.h}  // 11000001-10111110-00010010-11010101
// CHECK-INST: bfdot   za.s[w8, 5, vgx2], { z22.h, z23.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xd5,0x12,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be12d5 <unknown>

bfdot   za.s[w11, 2, vgx2], {z8.h, z9.h}, {z0.h, z1.h}  // 11000001-10100000-01110001-00010010
// CHECK-INST: bfdot   za.s[w11, 2, vgx2], { z8.h, z9.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x12,0x71,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a07112 <unknown>

bfdot   za.s[w11, 2], {z8.h, z9.h}, {z0.h, z1.h}  // 11000001-10100000-01110001-00010010
// CHECK-INST: bfdot   za.s[w11, 2, vgx2], { z8.h, z9.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x12,0x71,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a07112 <unknown>

bfdot   za.s[w9, 7, vgx2], {z12.h, z13.h}, {z10.h, z11.h}  // 11000001-10101010-00110001-10010111
// CHECK-INST: bfdot   za.s[w9, 7, vgx2], { z12.h, z13.h }, { z10.h, z11.h }
// CHECK-ENCODING: [0x97,0x31,0xaa,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1aa3197 <unknown>

bfdot   za.s[w9, 7], {z12.h, z13.h}, {z10.h, z11.h}  // 11000001-10101010-00110001-10010111
// CHECK-INST: bfdot   za.s[w9, 7, vgx2], { z12.h, z13.h }, { z10.h, z11.h }
// CHECK-ENCODING: [0x97,0x31,0xaa,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1aa3197 <unknown>


bfdot   za.s[w8, 0, vgx4], {z0.h - z3.h}, z0.h  // 11000001-00110000-00010000-00010000
// CHECK-INST: bfdot   za.s[w8, 0, vgx4], { z0.h - z3.h }, z0.h
// CHECK-ENCODING: [0x10,0x10,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1301010 <unknown>

bfdot   za.s[w8, 0], {z0.h - z3.h}, z0.h  // 11000001-00110000-00010000-00010000
// CHECK-INST: bfdot   za.s[w8, 0, vgx4], { z0.h - z3.h }, z0.h
// CHECK-ENCODING: [0x10,0x10,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1301010 <unknown>

bfdot   za.s[w10, 5, vgx4], {z10.h - z13.h}, z5.h  // 11000001-00110101-01010001-01010101
// CHECK-INST: bfdot   za.s[w10, 5, vgx4], { z10.h - z13.h }, z5.h
// CHECK-ENCODING: [0x55,0x51,0x35,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1355155 <unknown>

bfdot   za.s[w10, 5], {z10.h - z13.h}, z5.h  // 11000001-00110101-01010001-01010101
// CHECK-INST: bfdot   za.s[w10, 5, vgx4], { z10.h - z13.h }, z5.h
// CHECK-ENCODING: [0x55,0x51,0x35,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1355155 <unknown>

bfdot   za.s[w11, 7, vgx4], {z13.h - z16.h}, z8.h  // 11000001-00111000-01110001-10110111
// CHECK-INST: bfdot   za.s[w11, 7, vgx4], { z13.h - z16.h }, z8.h
// CHECK-ENCODING: [0xb7,0x71,0x38,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13871b7 <unknown>

bfdot   za.s[w11, 7], {z13.h - z16.h}, z8.h  // 11000001-00111000-01110001-10110111
// CHECK-INST: bfdot   za.s[w11, 7, vgx4], { z13.h - z16.h }, z8.h
// CHECK-ENCODING: [0xb7,0x71,0x38,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13871b7 <unknown>

bfdot   za.s[w11, 7, vgx4], {z31.h - z2.h}, z15.h  // 11000001-00111111-01110011-11110111
// CHECK-INST: bfdot   za.s[w11, 7, vgx4], {  z31.h, z0.h, z1.h, z2.h }, z15.h
// CHECK-ENCODING: [0xf7,0x73,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13f73f7 <unknown>

bfdot   za.s[w11, 7], {z31.h - z2.h}, z15.h  // 11000001-00111111-01110011-11110111
// CHECK-INST: bfdot   za.s[w11, 7, vgx4], {  z31.h, z0.h, z1.h, z2.h }, z15.h
// CHECK-ENCODING: [0xf7,0x73,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13f73f7 <unknown>

bfdot   za.s[w8, 5, vgx4], {z17.h - z20.h}, z0.h  // 11000001-00110000-00010010-00110101
// CHECK-INST: bfdot   za.s[w8, 5, vgx4], { z17.h - z20.h }, z0.h
// CHECK-ENCODING: [0x35,0x12,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1301235 <unknown>

bfdot   za.s[w8, 5], {z17.h - z20.h}, z0.h  // 11000001-00110000-00010010-00110101
// CHECK-INST: bfdot   za.s[w8, 5, vgx4], { z17.h - z20.h }, z0.h
// CHECK-ENCODING: [0x35,0x12,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1301235 <unknown>

bfdot   za.s[w8, 1, vgx4], {z1.h - z4.h}, z14.h  // 11000001-00111110-00010000-00110001
// CHECK-INST: bfdot   za.s[w8, 1, vgx4], { z1.h - z4.h }, z14.h
// CHECK-ENCODING: [0x31,0x10,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e1031 <unknown>

bfdot   za.s[w8, 1], {z1.h - z4.h}, z14.h  // 11000001-00111110-00010000-00110001
// CHECK-INST: bfdot   za.s[w8, 1, vgx4], { z1.h - z4.h }, z14.h
// CHECK-ENCODING: [0x31,0x10,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e1031 <unknown>

bfdot   za.s[w10, 0, vgx4], {z19.h - z22.h}, z4.h  // 11000001-00110100-01010010-01110000
// CHECK-INST: bfdot   za.s[w10, 0, vgx4], { z19.h - z22.h }, z4.h
// CHECK-ENCODING: [0x70,0x52,0x34,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1345270 <unknown>

bfdot   za.s[w10, 0], {z19.h - z22.h}, z4.h  // 11000001-00110100-01010010-01110000
// CHECK-INST: bfdot   za.s[w10, 0, vgx4], { z19.h - z22.h }, z4.h
// CHECK-ENCODING: [0x70,0x52,0x34,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1345270 <unknown>

bfdot   za.s[w8, 0, vgx4], {z12.h - z15.h}, z2.h  // 11000001-00110010-00010001-10010000
// CHECK-INST: bfdot   za.s[w8, 0, vgx4], { z12.h - z15.h }, z2.h
// CHECK-ENCODING: [0x90,0x11,0x32,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1321190 <unknown>

bfdot   za.s[w8, 0], {z12.h - z15.h}, z2.h  // 11000001-00110010-00010001-10010000
// CHECK-INST: bfdot   za.s[w8, 0, vgx4], { z12.h - z15.h }, z2.h
// CHECK-ENCODING: [0x90,0x11,0x32,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1321190 <unknown>

bfdot   za.s[w10, 1, vgx4], {z1.h - z4.h}, z10.h  // 11000001-00111010-01010000-00110001
// CHECK-INST: bfdot   za.s[w10, 1, vgx4], { z1.h - z4.h }, z10.h
// CHECK-ENCODING: [0x31,0x50,0x3a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13a5031 <unknown>

bfdot   za.s[w10, 1], {z1.h - z4.h}, z10.h  // 11000001-00111010-01010000-00110001
// CHECK-INST: bfdot   za.s[w10, 1, vgx4], { z1.h - z4.h }, z10.h
// CHECK-ENCODING: [0x31,0x50,0x3a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13a5031 <unknown>

bfdot   za.s[w8, 5, vgx4], {z22.h - z25.h}, z14.h  // 11000001-00111110-00010010-11010101
// CHECK-INST: bfdot   za.s[w8, 5, vgx4], { z22.h - z25.h }, z14.h
// CHECK-ENCODING: [0xd5,0x12,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e12d5 <unknown>

bfdot   za.s[w8, 5], {z22.h - z25.h}, z14.h  // 11000001-00111110-00010010-11010101
// CHECK-INST: bfdot   za.s[w8, 5, vgx4], { z22.h - z25.h }, z14.h
// CHECK-ENCODING: [0xd5,0x12,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e12d5 <unknown>

bfdot   za.s[w11, 2, vgx4], {z9.h - z12.h}, z1.h  // 11000001-00110001-01110001-00110010
// CHECK-INST: bfdot   za.s[w11, 2, vgx4], { z9.h - z12.h }, z1.h
// CHECK-ENCODING: [0x32,0x71,0x31,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1317132 <unknown>

bfdot   za.s[w11, 2], {z9.h - z12.h}, z1.h  // 11000001-00110001-01110001-00110010
// CHECK-INST: bfdot   za.s[w11, 2, vgx4], { z9.h - z12.h }, z1.h
// CHECK-ENCODING: [0x32,0x71,0x31,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1317132 <unknown>

bfdot   za.s[w9, 7, vgx4], {z12.h - z15.h}, z11.h  // 11000001-00111011-00110001-10010111
// CHECK-INST: bfdot   za.s[w9, 7, vgx4], { z12.h - z15.h }, z11.h
// CHECK-ENCODING: [0x97,0x31,0x3b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13b3197 <unknown>

bfdot   za.s[w9, 7], {z12.h - z15.h}, z11.h  // 11000001-00111011-00110001-10010111
// CHECK-INST: bfdot   za.s[w9, 7, vgx4], { z12.h - z15.h }, z11.h
// CHECK-ENCODING: [0x97,0x31,0x3b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13b3197 <unknown>


bfdot   za.s[w8, 0, vgx4], {z0.h - z3.h}, z0.h[0]  // 11000001-01010000-10010000-00011000
// CHECK-INST: bfdot   za.s[w8, 0, vgx4], { z0.h - z3.h }, z0.h[0]
// CHECK-ENCODING: [0x18,0x90,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1509018 <unknown>

bfdot   za.s[w8, 0], {z0.h - z3.h}, z0.h[0]  // 11000001-01010000-10010000-00011000
// CHECK-INST: bfdot   za.s[w8, 0, vgx4], { z0.h - z3.h }, z0.h[0]
// CHECK-ENCODING: [0x18,0x90,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1509018 <unknown>

bfdot   za.s[w10, 5, vgx4], {z8.h - z11.h}, z5.h[1]  // 11000001-01010101-11010101-00011101
// CHECK-INST: bfdot   za.s[w10, 5, vgx4], { z8.h - z11.h }, z5.h[1]
// CHECK-ENCODING: [0x1d,0xd5,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c155d51d <unknown>

bfdot   za.s[w10, 5], {z8.h - z11.h}, z5.h[1]  // 11000001-01010101-11010101-00011101
// CHECK-INST: bfdot   za.s[w10, 5, vgx4], { z8.h - z11.h }, z5.h[1]
// CHECK-ENCODING: [0x1d,0xd5,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c155d51d <unknown>

bfdot   za.s[w11, 7, vgx4], {z12.h - z15.h}, z8.h[3]  // 11000001-01011000-11111101-10011111
// CHECK-INST: bfdot   za.s[w11, 7, vgx4], { z12.h - z15.h }, z8.h[3]
// CHECK-ENCODING: [0x9f,0xfd,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c158fd9f <unknown>

bfdot   za.s[w11, 7], {z12.h - z15.h}, z8.h[3]  // 11000001-01011000-11111101-10011111
// CHECK-INST: bfdot   za.s[w11, 7, vgx4], { z12.h - z15.h }, z8.h[3]
// CHECK-ENCODING: [0x9f,0xfd,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c158fd9f <unknown>

bfdot   za.s[w11, 7, vgx4], {z28.h - z31.h}, z15.h[3]  // 11000001-01011111-11111111-10011111
// CHECK-INST: bfdot   za.s[w11, 7, vgx4], { z28.h - z31.h }, z15.h[3]
// CHECK-ENCODING: [0x9f,0xff,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15fff9f <unknown>

bfdot   za.s[w11, 7], {z28.h - z31.h}, z15.h[3]  // 11000001-01011111-11111111-10011111
// CHECK-INST: bfdot   za.s[w11, 7, vgx4], { z28.h - z31.h }, z15.h[3]
// CHECK-ENCODING: [0x9f,0xff,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15fff9f <unknown>

bfdot   za.s[w8, 5, vgx4], {z16.h - z19.h}, z0.h[3]  // 11000001-01010000-10011110-00011101
// CHECK-INST: bfdot   za.s[w8, 5, vgx4], { z16.h - z19.h }, z0.h[3]
// CHECK-ENCODING: [0x1d,0x9e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1509e1d <unknown>

bfdot   za.s[w8, 5], {z16.h - z19.h}, z0.h[3]  // 11000001-01010000-10011110-00011101
// CHECK-INST: bfdot   za.s[w8, 5, vgx4], { z16.h - z19.h }, z0.h[3]
// CHECK-ENCODING: [0x1d,0x9e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1509e1d <unknown>

bfdot   za.s[w8, 1, vgx4], {z0.h - z3.h}, z14.h[1]  // 11000001-01011110-10010100-00011001
// CHECK-INST: bfdot   za.s[w8, 1, vgx4], { z0.h - z3.h }, z14.h[1]
// CHECK-ENCODING: [0x19,0x94,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e9419 <unknown>

bfdot   za.s[w8, 1], {z0.h - z3.h}, z14.h[1]  // 11000001-01011110-10010100-00011001
// CHECK-INST: bfdot   za.s[w8, 1, vgx4], { z0.h - z3.h }, z14.h[1]
// CHECK-ENCODING: [0x19,0x94,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e9419 <unknown>

bfdot   za.s[w10, 0, vgx4], {z16.h - z19.h}, z4.h[1]  // 11000001-01010100-11010110-00011000
// CHECK-INST: bfdot   za.s[w10, 0, vgx4], { z16.h - z19.h }, z4.h[1]
// CHECK-ENCODING: [0x18,0xd6,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c154d618 <unknown>

bfdot   za.s[w10, 0], {z16.h - z19.h}, z4.h[1]  // 11000001-01010100-11010110-00011000
// CHECK-INST: bfdot   za.s[w10, 0, vgx4], { z16.h - z19.h }, z4.h[1]
// CHECK-ENCODING: [0x18,0xd6,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c154d618 <unknown>

bfdot   za.s[w8, 0, vgx4], {z12.h - z15.h}, z2.h[2]  // 11000001-01010010-10011001-10011000
// CHECK-INST: bfdot   za.s[w8, 0, vgx4], { z12.h - z15.h }, z2.h[2]
// CHECK-ENCODING: [0x98,0x99,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1529998 <unknown>

bfdot   za.s[w8, 0], {z12.h - z15.h}, z2.h[2]  // 11000001-01010010-10011001-10011000
// CHECK-INST: bfdot   za.s[w8, 0, vgx4], { z12.h - z15.h }, z2.h[2]
// CHECK-ENCODING: [0x98,0x99,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1529998 <unknown>

bfdot   za.s[w10, 1, vgx4], {z0.h - z3.h}, z10.h[2]  // 11000001-01011010-11011000-00011001
// CHECK-INST: bfdot   za.s[w10, 1, vgx4], { z0.h - z3.h }, z10.h[2]
// CHECK-ENCODING: [0x19,0xd8,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15ad819 <unknown>

bfdot   za.s[w10, 1], {z0.h - z3.h}, z10.h[2]  // 11000001-01011010-11011000-00011001
// CHECK-INST: bfdot   za.s[w10, 1, vgx4], { z0.h - z3.h }, z10.h[2]
// CHECK-ENCODING: [0x19,0xd8,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15ad819 <unknown>

bfdot   za.s[w8, 5, vgx4], {z20.h - z23.h}, z14.h[2]  // 11000001-01011110-10011010-10011101
// CHECK-INST: bfdot   za.s[w8, 5, vgx4], { z20.h - z23.h }, z14.h[2]
// CHECK-ENCODING: [0x9d,0x9a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e9a9d <unknown>

bfdot   za.s[w8, 5], {z20.h - z23.h}, z14.h[2]  // 11000001-01011110-10011010-10011101
// CHECK-INST: bfdot   za.s[w8, 5, vgx4], { z20.h - z23.h }, z14.h[2]
// CHECK-ENCODING: [0x9d,0x9a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e9a9d <unknown>

bfdot   za.s[w11, 2, vgx4], {z8.h - z11.h}, z1.h[1]  // 11000001-01010001-11110101-00011010
// CHECK-INST: bfdot   za.s[w11, 2, vgx4], { z8.h - z11.h }, z1.h[1]
// CHECK-ENCODING: [0x1a,0xf5,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c151f51a <unknown>

bfdot   za.s[w11, 2], {z8.h - z11.h}, z1.h[1]  // 11000001-01010001-11110101-00011010
// CHECK-INST: bfdot   za.s[w11, 2, vgx4], { z8.h - z11.h }, z1.h[1]
// CHECK-ENCODING: [0x1a,0xf5,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c151f51a <unknown>

bfdot   za.s[w9, 7, vgx4], {z12.h - z15.h}, z11.h[2]  // 11000001-01011011-10111001-10011111
// CHECK-INST: bfdot   za.s[w9, 7, vgx4], { z12.h - z15.h }, z11.h[2]
// CHECK-ENCODING: [0x9f,0xb9,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15bb99f <unknown>

bfdot   za.s[w9, 7], {z12.h - z15.h}, z11.h[2]  // 11000001-01011011-10111001-10011111
// CHECK-INST: bfdot   za.s[w9, 7, vgx4], { z12.h - z15.h }, z11.h[2]
// CHECK-ENCODING: [0x9f,0xb9,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15bb99f <unknown>


bfdot   za.s[w8, 0, vgx4], {z0.h - z3.h}, {z0.h - z3.h}  // 11000001-10100001-00010000-00010000
// CHECK-INST: bfdot   za.s[w8, 0, vgx4], { z0.h - z3.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x10,0x10,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11010 <unknown>

bfdot   za.s[w8, 0], {z0.h - z3.h}, {z0.h - z3.h}  // 11000001-10100001-00010000-00010000
// CHECK-INST: bfdot   za.s[w8, 0, vgx4], { z0.h - z3.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x10,0x10,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11010 <unknown>

bfdot   za.s[w10, 5, vgx4], {z8.h - z11.h}, {z20.h - z23.h}  // 11000001-10110101-01010001-00010101
// CHECK-INST: bfdot   za.s[w10, 5, vgx4], { z8.h - z11.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x15,0x51,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b55115 <unknown>

bfdot   za.s[w10, 5], {z8.h - z11.h}, {z20.h - z23.h}  // 11000001-10110101-01010001-00010101
// CHECK-INST: bfdot   za.s[w10, 5, vgx4], { z8.h - z11.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x15,0x51,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b55115 <unknown>

bfdot   za.s[w11, 7, vgx4], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-10101001-01110001-10010111
// CHECK-INST: bfdot   za.s[w11, 7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x97,0x71,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a97197 <unknown>

bfdot   za.s[w11, 7], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-10101001-01110001-10010111
// CHECK-INST: bfdot   za.s[w11, 7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x97,0x71,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a97197 <unknown>

bfdot   za.s[w11, 7, vgx4], {z28.h - z31.h}, {z28.h - z31.h}  // 11000001-10111101-01110011-10010111
// CHECK-INST: bfdot   za.s[w11, 7, vgx4], { z28.h - z31.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x97,0x73,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd7397 <unknown>

bfdot   za.s[w11, 7], {z28.h - z31.h}, {z28.h - z31.h}  // 11000001-10111101-01110011-10010111
// CHECK-INST: bfdot   za.s[w11, 7, vgx4], { z28.h - z31.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x97,0x73,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd7397 <unknown>

bfdot   za.s[w8, 5, vgx4], {z16.h - z19.h}, {z16.h - z19.h}  // 11000001-10110001-00010010-00010101
// CHECK-INST: bfdot   za.s[w8, 5, vgx4], { z16.h - z19.h }, { z16.h - z19.h }
// CHECK-ENCODING: [0x15,0x12,0xb1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b11215 <unknown>

bfdot   za.s[w8, 5], {z16.h - z19.h}, {z16.h - z19.h}  // 11000001-10110001-00010010-00010101
// CHECK-INST: bfdot   za.s[w8, 5, vgx4], { z16.h - z19.h }, { z16.h - z19.h }
// CHECK-ENCODING: [0x15,0x12,0xb1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b11215 <unknown>

bfdot   za.s[w8, 1, vgx4], {z0.h - z3.h}, {z28.h - z31.h}  // 11000001-10111101-00010000-00010001
// CHECK-INST: bfdot   za.s[w8, 1, vgx4], { z0.h - z3.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x11,0x10,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd1011 <unknown>

bfdot   za.s[w8, 1], {z0.h - z3.h}, {z28.h - z31.h}  // 11000001-10111101-00010000-00010001
// CHECK-INST: bfdot   za.s[w8, 1, vgx4], { z0.h - z3.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x11,0x10,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd1011 <unknown>

bfdot   za.s[w10, 0, vgx4], {z16.h - z19.h}, {z20.h - z23.h}  // 11000001-10110101-01010010-00010000
// CHECK-INST: bfdot   za.s[w10, 0, vgx4], { z16.h - z19.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x10,0x52,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b55210 <unknown>

bfdot   za.s[w10, 0], {z16.h - z19.h}, {z20.h - z23.h}  // 11000001-10110101-01010010-00010000
// CHECK-INST: bfdot   za.s[w10, 0, vgx4], { z16.h - z19.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x10,0x52,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b55210 <unknown>

bfdot   za.s[w8, 0, vgx4], {z12.h - z15.h}, {z0.h - z3.h}  // 11000001-10100001-00010001-10010000
// CHECK-INST: bfdot   za.s[w8, 0, vgx4], { z12.h - z15.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x90,0x11,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11190 <unknown>

bfdot   za.s[w8, 0], {z12.h - z15.h}, {z0.h - z3.h}  // 11000001-10100001-00010001-10010000
// CHECK-INST: bfdot   za.s[w8, 0, vgx4], { z12.h - z15.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x90,0x11,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11190 <unknown>

bfdot   za.s[w10, 1, vgx4], {z0.h - z3.h}, {z24.h - z27.h}  // 11000001-10111001-01010000-00010001
// CHECK-INST: bfdot   za.s[w10, 1, vgx4], { z0.h - z3.h }, { z24.h - z27.h }
// CHECK-ENCODING: [0x11,0x50,0xb9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b95011 <unknown>

bfdot   za.s[w10, 1], {z0.h - z3.h}, {z24.h - z27.h}  // 11000001-10111001-01010000-00010001
// CHECK-INST: bfdot   za.s[w10, 1, vgx4], { z0.h - z3.h }, { z24.h - z27.h }
// CHECK-ENCODING: [0x11,0x50,0xb9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b95011 <unknown>

bfdot   za.s[w8, 5, vgx4], {z20.h - z23.h}, {z28.h - z31.h}  // 11000001-10111101-00010010-10010101
// CHECK-INST: bfdot   za.s[w8, 5, vgx4], { z20.h - z23.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x95,0x12,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd1295 <unknown>

bfdot   za.s[w8, 5], {z20.h - z23.h}, {z28.h - z31.h}  // 11000001-10111101-00010010-10010101
// CHECK-INST: bfdot   za.s[w8, 5, vgx4], { z20.h - z23.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x95,0x12,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd1295 <unknown>

bfdot   za.s[w11, 2, vgx4], {z8.h - z11.h}, {z0.h - z3.h}  // 11000001-10100001-01110001-00010010
// CHECK-INST: bfdot   za.s[w11, 2, vgx4], { z8.h - z11.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x12,0x71,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a17112 <unknown>

bfdot   za.s[w11, 2], {z8.h - z11.h}, {z0.h - z3.h}  // 11000001-10100001-01110001-00010010
// CHECK-INST: bfdot   za.s[w11, 2, vgx4], { z8.h - z11.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x12,0x71,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a17112 <unknown>

bfdot   za.s[w9, 7, vgx4], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-10101001-00110001-10010111
// CHECK-INST: bfdot   za.s[w9, 7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x97,0x31,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a93197 <unknown>

bfdot   za.s[w9, 7], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-10101001-00110001-10010111
// CHECK-INST: bfdot   za.s[w9, 7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x97,0x31,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a93197 <unknown>

