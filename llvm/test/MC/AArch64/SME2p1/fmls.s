// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p1,+sme-f16f16 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p1,+sme-f16f16 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2p1,+sme-f16f16 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p1,+sme-f16f16 < %s \
// RUN:        | llvm-objdump -d  --mattr=-sme2p1 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p1,+sme-f16f16 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2p1,+sme-f16f16 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

fmls    za.h[w8, 0, vgx2], {z0.h, z1.h}, z0.h  // 11000001-00100000-00011100-00001000
// CHECK-INST: fmls    za.h[w8, 0, vgx2], { z0.h, z1.h }, z0.h
// CHECK-ENCODING: [0x08,0x1c,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1201c08 <unknown>

fmls    za.h[w8, 0], {z0.h - z1.h}, z0.h  // 11000001-00100000-00011100-00001000
// CHECK-INST: fmls    za.h[w8, 0, vgx2], { z0.h, z1.h }, z0.h
// CHECK-ENCODING: [0x08,0x1c,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1201c08 <unknown>

fmls    za.h[w10, 5, vgx2], {z10.h, z11.h}, z5.h  // 11000001-00100101-01011101-01001101
// CHECK-INST: fmls    za.h[w10, 5, vgx2], { z10.h, z11.h }, z5.h
// CHECK-ENCODING: [0x4d,0x5d,0x25,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1255d4d <unknown>

fmls    za.h[w10, 5], {z10.h - z11.h}, z5.h  // 11000001-00100101-01011101-01001101
// CHECK-INST: fmls    za.h[w10, 5, vgx2], { z10.h, z11.h }, z5.h
// CHECK-ENCODING: [0x4d,0x5d,0x25,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1255d4d <unknown>

fmls    za.h[w11, 7, vgx2], {z13.h, z14.h}, z8.h  // 11000001-00101000-01111101-10101111
// CHECK-INST: fmls    za.h[w11, 7, vgx2], { z13.h, z14.h }, z8.h
// CHECK-ENCODING: [0xaf,0x7d,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1287daf <unknown>

fmls    za.h[w11, 7], {z13.h - z14.h}, z8.h  // 11000001-00101000-01111101-10101111
// CHECK-INST: fmls    za.h[w11, 7, vgx2], { z13.h, z14.h }, z8.h
// CHECK-ENCODING: [0xaf,0x7d,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1287daf <unknown>

fmls    za.h[w11, 7, vgx2], {z31.h, z0.h}, z15.h  // 11000001-00101111-01111111-11101111
// CHECK-INST: fmls    za.h[w11, 7, vgx2], { z31.h, z0.h }, z15.h
// CHECK-ENCODING: [0xef,0x7f,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c12f7fef <unknown>

fmls    za.h[w11, 7], {z31.h - z0.h}, z15.h  // 11000001-00101111-01111111-11101111
// CHECK-INST: fmls    za.h[w11, 7, vgx2], { z31.h, z0.h }, z15.h
// CHECK-ENCODING: [0xef,0x7f,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c12f7fef <unknown>

fmls    za.h[w8, 5, vgx2], {z17.h, z18.h}, z0.h  // 11000001-00100000-00011110-00101101
// CHECK-INST: fmls    za.h[w8, 5, vgx2], { z17.h, z18.h }, z0.h
// CHECK-ENCODING: [0x2d,0x1e,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1201e2d <unknown>

fmls    za.h[w8, 5], {z17.h - z18.h}, z0.h  // 11000001-00100000-00011110-00101101
// CHECK-INST: fmls    za.h[w8, 5, vgx2], { z17.h, z18.h }, z0.h
// CHECK-ENCODING: [0x2d,0x1e,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1201e2d <unknown>

fmls    za.h[w8, 1, vgx2], {z1.h, z2.h}, z14.h  // 11000001-00101110-00011100-00101001
// CHECK-INST: fmls    za.h[w8, 1, vgx2], { z1.h, z2.h }, z14.h
// CHECK-ENCODING: [0x29,0x1c,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c12e1c29 <unknown>

fmls    za.h[w8, 1], {z1.h - z2.h}, z14.h  // 11000001-00101110-00011100-00101001
// CHECK-INST: fmls    za.h[w8, 1, vgx2], { z1.h, z2.h }, z14.h
// CHECK-ENCODING: [0x29,0x1c,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c12e1c29 <unknown>

fmls    za.h[w10, 0, vgx2], {z19.h, z20.h}, z4.h  // 11000001-00100100-01011110-01101000
// CHECK-INST: fmls    za.h[w10, 0, vgx2], { z19.h, z20.h }, z4.h
// CHECK-ENCODING: [0x68,0x5e,0x24,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1245e68 <unknown>

fmls    za.h[w10, 0], {z19.h - z20.h}, z4.h  // 11000001-00100100-01011110-01101000
// CHECK-INST: fmls    za.h[w10, 0, vgx2], { z19.h, z20.h }, z4.h
// CHECK-ENCODING: [0x68,0x5e,0x24,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1245e68 <unknown>

fmls    za.h[w8, 0, vgx2], {z12.h, z13.h}, z2.h  // 11000001-00100010-00011101-10001000
// CHECK-INST: fmls    za.h[w8, 0, vgx2], { z12.h, z13.h }, z2.h
// CHECK-ENCODING: [0x88,0x1d,0x22,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1221d88 <unknown>

fmls    za.h[w8, 0], {z12.h - z13.h}, z2.h  // 11000001-00100010-00011101-10001000
// CHECK-INST: fmls    za.h[w8, 0, vgx2], { z12.h, z13.h }, z2.h
// CHECK-ENCODING: [0x88,0x1d,0x22,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1221d88 <unknown>

fmls    za.h[w10, 1, vgx2], {z1.h, z2.h}, z10.h  // 11000001-00101010-01011100-00101001
// CHECK-INST: fmls    za.h[w10, 1, vgx2], { z1.h, z2.h }, z10.h
// CHECK-ENCODING: [0x29,0x5c,0x2a,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c12a5c29 <unknown>

fmls    za.h[w10, 1], {z1.h - z2.h}, z10.h  // 11000001-00101010-01011100-00101001
// CHECK-INST: fmls    za.h[w10, 1, vgx2], { z1.h, z2.h }, z10.h
// CHECK-ENCODING: [0x29,0x5c,0x2a,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c12a5c29 <unknown>

fmls    za.h[w8, 5, vgx2], {z22.h, z23.h}, z14.h  // 11000001-00101110-00011110-11001101
// CHECK-INST: fmls    za.h[w8, 5, vgx2], { z22.h, z23.h }, z14.h
// CHECK-ENCODING: [0xcd,0x1e,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c12e1ecd <unknown>

fmls    za.h[w8, 5], {z22.h - z23.h}, z14.h  // 11000001-00101110-00011110-11001101
// CHECK-INST: fmls    za.h[w8, 5, vgx2], { z22.h, z23.h }, z14.h
// CHECK-ENCODING: [0xcd,0x1e,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c12e1ecd <unknown>

fmls    za.h[w11, 2, vgx2], {z9.h, z10.h}, z1.h  // 11000001-00100001-01111101-00101010
// CHECK-INST: fmls    za.h[w11, 2, vgx2], { z9.h, z10.h }, z1.h
// CHECK-ENCODING: [0x2a,0x7d,0x21,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1217d2a <unknown>

fmls    za.h[w11, 2], {z9.h - z10.h}, z1.h  // 11000001-00100001-01111101-00101010
// CHECK-INST: fmls    za.h[w11, 2, vgx2], { z9.h, z10.h }, z1.h
// CHECK-ENCODING: [0x2a,0x7d,0x21,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1217d2a <unknown>

fmls    za.h[w9, 7, vgx2], {z12.h, z13.h}, z11.h  // 11000001-00101011-00111101-10001111
// CHECK-INST: fmls    za.h[w9, 7, vgx2], { z12.h, z13.h }, z11.h
// CHECK-ENCODING: [0x8f,0x3d,0x2b,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c12b3d8f <unknown>

fmls    za.h[w9, 7], {z12.h - z13.h}, z11.h  // 11000001-00101011-00111101-10001111
// CHECK-INST: fmls    za.h[w9, 7, vgx2], { z12.h, z13.h }, z11.h
// CHECK-ENCODING: [0x8f,0x3d,0x2b,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c12b3d8f <unknown>


fmls    za.h[w8, 0, vgx2], {z0.h, z1.h}, z0.h[0]  // 11000001-00010000-00010000-00010000
// CHECK-INST: fmls    za.h[w8, 0, vgx2], { z0.h, z1.h }, z0.h[0]
// CHECK-ENCODING: [0x10,0x10,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1101010 <unknown>

fmls    za.h[w8, 0], {z0.h - z1.h}, z0.h[0]  // 11000001-00010000-00010000-00010000
// CHECK-INST: fmls    za.h[w8, 0, vgx2], { z0.h, z1.h }, z0.h[0]
// CHECK-ENCODING: [0x10,0x10,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1101010 <unknown>

fmls    za.h[w10, 5, vgx2], {z10.h, z11.h}, z5.h[2]  // 11000001-00010101-01010101-01010101
// CHECK-INST: fmls    za.h[w10, 5, vgx2], { z10.h, z11.h }, z5.h[2]
// CHECK-ENCODING: [0x55,0x55,0x15,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1155555 <unknown>

fmls    za.h[w10, 5], {z10.h - z11.h}, z5.h[2]  // 11000001-00010101-01010101-01010101
// CHECK-INST: fmls    za.h[w10, 5, vgx2], { z10.h, z11.h }, z5.h[2]
// CHECK-ENCODING: [0x55,0x55,0x15,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1155555 <unknown>

fmls    za.h[w11, 7, vgx2], {z12.h, z13.h}, z8.h[6]  // 11000001-00011000-01111101-10010111
// CHECK-INST: fmls    za.h[w11, 7, vgx2], { z12.h, z13.h }, z8.h[6]
// CHECK-ENCODING: [0x97,0x7d,0x18,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1187d97 <unknown>

fmls    za.h[w11, 7], {z12.h - z13.h}, z8.h[6]  // 11000001-00011000-01111101-10010111
// CHECK-INST: fmls    za.h[w11, 7, vgx2], { z12.h, z13.h }, z8.h[6]
// CHECK-ENCODING: [0x97,0x7d,0x18,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1187d97 <unknown>

fmls    za.h[w11, 7, vgx2], {z30.h, z31.h}, z15.h[7]  // 11000001-00011111-01111111-11011111
// CHECK-INST: fmls    za.h[w11, 7, vgx2], { z30.h, z31.h }, z15.h[7]
// CHECK-ENCODING: [0xdf,0x7f,0x1f,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c11f7fdf <unknown>

fmls    za.h[w11, 7], {z30.h - z31.h}, z15.h[7]  // 11000001-00011111-01111111-11011111
// CHECK-INST: fmls    za.h[w11, 7, vgx2], { z30.h, z31.h }, z15.h[7]
// CHECK-ENCODING: [0xdf,0x7f,0x1f,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c11f7fdf <unknown>

fmls    za.h[w8, 5, vgx2], {z16.h, z17.h}, z0.h[6]  // 11000001-00010000-00011110-00010101
// CHECK-INST: fmls    za.h[w8, 5, vgx2], { z16.h, z17.h }, z0.h[6]
// CHECK-ENCODING: [0x15,0x1e,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1101e15 <unknown>

fmls    za.h[w8, 5], {z16.h - z17.h}, z0.h[6]  // 11000001-00010000-00011110-00010101
// CHECK-INST: fmls    za.h[w8, 5, vgx2], { z16.h, z17.h }, z0.h[6]
// CHECK-ENCODING: [0x15,0x1e,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1101e15 <unknown>

fmls    za.h[w8, 1, vgx2], {z0.h, z1.h}, z14.h[2]  // 11000001-00011110-00010100-00010001
// CHECK-INST: fmls    za.h[w8, 1, vgx2], { z0.h, z1.h }, z14.h[2]
// CHECK-ENCODING: [0x11,0x14,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c11e1411 <unknown>

fmls    za.h[w8, 1], {z0.h - z1.h}, z14.h[2]  // 11000001-00011110-00010100-00010001
// CHECK-INST: fmls    za.h[w8, 1, vgx2], { z0.h, z1.h }, z14.h[2]
// CHECK-ENCODING: [0x11,0x14,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c11e1411 <unknown>

fmls    za.h[w10, 0, vgx2], {z18.h, z19.h}, z4.h[3]  // 11000001-00010100-01010110-01011000
// CHECK-INST: fmls    za.h[w10, 0, vgx2], { z18.h, z19.h }, z4.h[3]
// CHECK-ENCODING: [0x58,0x56,0x14,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1145658 <unknown>

fmls    za.h[w10, 0], {z18.h - z19.h}, z4.h[3]  // 11000001-00010100-01010110-01011000
// CHECK-INST: fmls    za.h[w10, 0, vgx2], { z18.h, z19.h }, z4.h[3]
// CHECK-ENCODING: [0x58,0x56,0x14,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1145658 <unknown>

fmls    za.h[w8, 0, vgx2], {z12.h, z13.h}, z2.h[4]  // 11000001-00010010-00011001-10010000
// CHECK-INST: fmls    za.h[w8, 0, vgx2], { z12.h, z13.h }, z2.h[4]
// CHECK-ENCODING: [0x90,0x19,0x12,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1121990 <unknown>

fmls    za.h[w8, 0], {z12.h - z13.h}, z2.h[4]  // 11000001-00010010-00011001-10010000
// CHECK-INST: fmls    za.h[w8, 0, vgx2], { z12.h, z13.h }, z2.h[4]
// CHECK-ENCODING: [0x90,0x19,0x12,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1121990 <unknown>

fmls    za.h[w10, 1, vgx2], {z0.h, z1.h}, z10.h[4]  // 11000001-00011010-01011000-00010001
// CHECK-INST: fmls    za.h[w10, 1, vgx2], { z0.h, z1.h }, z10.h[4]
// CHECK-ENCODING: [0x11,0x58,0x1a,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c11a5811 <unknown>

fmls    za.h[w10, 1], {z0.h - z1.h}, z10.h[4]  // 11000001-00011010-01011000-00010001
// CHECK-INST: fmls    za.h[w10, 1, vgx2], { z0.h, z1.h }, z10.h[4]
// CHECK-ENCODING: [0x11,0x58,0x1a,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c11a5811 <unknown>

fmls    za.h[w8, 5, vgx2], {z22.h, z23.h}, z14.h[5]  // 11000001-00011110-00011010-11011101
// CHECK-INST: fmls    za.h[w8, 5, vgx2], { z22.h, z23.h }, z14.h[5]
// CHECK-ENCODING: [0xdd,0x1a,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c11e1add <unknown>

fmls    za.h[w8, 5], {z22.h - z23.h}, z14.h[5]  // 11000001-00011110-00011010-11011101
// CHECK-INST: fmls    za.h[w8, 5, vgx2], { z22.h, z23.h }, z14.h[5]
// CHECK-ENCODING: [0xdd,0x1a,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c11e1add <unknown>

fmls    za.h[w11, 2, vgx2], {z8.h, z9.h}, z1.h[2]  // 11000001-00010001-01110101-00010010
// CHECK-INST: fmls    za.h[w11, 2, vgx2], { z8.h, z9.h }, z1.h[2]
// CHECK-ENCODING: [0x12,0x75,0x11,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1117512 <unknown>

fmls    za.h[w11, 2], {z8.h - z9.h}, z1.h[2]  // 11000001-00010001-01110101-00010010
// CHECK-INST: fmls    za.h[w11, 2, vgx2], { z8.h, z9.h }, z1.h[2]
// CHECK-ENCODING: [0x12,0x75,0x11,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1117512 <unknown>

fmls    za.h[w9, 7, vgx2], {z12.h, z13.h}, z11.h[4]  // 11000001-00011011-00111001-10010111
// CHECK-INST: fmls    za.h[w9, 7, vgx2], { z12.h, z13.h }, z11.h[4]
// CHECK-ENCODING: [0x97,0x39,0x1b,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c11b3997 <unknown>

fmls    za.h[w9, 7], {z12.h - z13.h}, z11.h[4]  // 11000001-00011011-00111001-10010111
// CHECK-INST: fmls    za.h[w9, 7, vgx2], { z12.h, z13.h }, z11.h[4]
// CHECK-ENCODING: [0x97,0x39,0x1b,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c11b3997 <unknown>


fmls    za.h[w8, 0, vgx2], {z0.h, z1.h}, {z0.h, z1.h}  // 11000001-10100000-00010000-00011000
// CHECK-INST: fmls    za.h[w8, 0, vgx2], { z0.h, z1.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x18,0x10,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1a01018 <unknown>

fmls    za.h[w8, 0], {z0.h - z1.h}, {z0.h - z1.h}  // 11000001-10100000-00010000-00011000
// CHECK-INST: fmls    za.h[w8, 0, vgx2], { z0.h, z1.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x18,0x10,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1a01018 <unknown>

fmls    za.h[w10, 5, vgx2], {z10.h, z11.h}, {z20.h, z21.h}  // 11000001-10110100-01010001-01011101
// CHECK-INST: fmls    za.h[w10, 5, vgx2], { z10.h, z11.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x5d,0x51,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1b4515d <unknown>

fmls    za.h[w10, 5], {z10.h - z11.h}, {z20.h - z21.h}  // 11000001-10110100-01010001-01011101
// CHECK-INST: fmls    za.h[w10, 5, vgx2], { z10.h, z11.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x5d,0x51,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1b4515d <unknown>

fmls    za.h[w11, 7, vgx2], {z12.h, z13.h}, {z8.h, z9.h}  // 11000001-10101000-01110001-10011111
// CHECK-INST: fmls    za.h[w11, 7, vgx2], { z12.h, z13.h }, { z8.h, z9.h }
// CHECK-ENCODING: [0x9f,0x71,0xa8,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1a8719f <unknown>

fmls    za.h[w11, 7], {z12.h - z13.h}, {z8.h - z9.h}  // 11000001-10101000-01110001-10011111
// CHECK-INST: fmls    za.h[w11, 7, vgx2], { z12.h, z13.h }, { z8.h, z9.h }
// CHECK-ENCODING: [0x9f,0x71,0xa8,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1a8719f <unknown>

fmls    za.h[w11, 7, vgx2], {z30.h, z31.h}, {z30.h, z31.h}  // 11000001-10111110-01110011-11011111
// CHECK-INST: fmls    za.h[w11, 7, vgx2], { z30.h, z31.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xdf,0x73,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1be73df <unknown>

fmls    za.h[w11, 7], {z30.h - z31.h}, {z30.h - z31.h}  // 11000001-10111110-01110011-11011111
// CHECK-INST: fmls    za.h[w11, 7, vgx2], { z30.h, z31.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xdf,0x73,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1be73df <unknown>

fmls    za.h[w8, 5, vgx2], {z16.h, z17.h}, {z16.h, z17.h}  // 11000001-10110000-00010010-00011101
// CHECK-INST: fmls    za.h[w8, 5, vgx2], { z16.h, z17.h }, { z16.h, z17.h }
// CHECK-ENCODING: [0x1d,0x12,0xb0,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1b0121d <unknown>

fmls    za.h[w8, 5], {z16.h - z17.h}, {z16.h - z17.h}  // 11000001-10110000-00010010-00011101
// CHECK-INST: fmls    za.h[w8, 5, vgx2], { z16.h, z17.h }, { z16.h, z17.h }
// CHECK-ENCODING: [0x1d,0x12,0xb0,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1b0121d <unknown>

fmls    za.h[w8, 1, vgx2], {z0.h, z1.h}, {z30.h, z31.h}  // 11000001-10111110-00010000-00011001
// CHECK-INST: fmls    za.h[w8, 1, vgx2], { z0.h, z1.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0x19,0x10,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1be1019 <unknown>

fmls    za.h[w8, 1], {z0.h - z1.h}, {z30.h - z31.h}  // 11000001-10111110-00010000-00011001
// CHECK-INST: fmls    za.h[w8, 1, vgx2], { z0.h, z1.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0x19,0x10,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1be1019 <unknown>

fmls    za.h[w10, 0, vgx2], {z18.h, z19.h}, {z20.h, z21.h}  // 11000001-10110100-01010010-01011000
// CHECK-INST: fmls    za.h[w10, 0, vgx2], { z18.h, z19.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x58,0x52,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1b45258 <unknown>

fmls    za.h[w10, 0], {z18.h - z19.h}, {z20.h - z21.h}  // 11000001-10110100-01010010-01011000
// CHECK-INST: fmls    za.h[w10, 0, vgx2], { z18.h, z19.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x58,0x52,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1b45258 <unknown>

fmls    za.h[w8, 0, vgx2], {z12.h, z13.h}, {z2.h, z3.h}  // 11000001-10100010-00010001-10011000
// CHECK-INST: fmls    za.h[w8, 0, vgx2], { z12.h, z13.h }, { z2.h, z3.h }
// CHECK-ENCODING: [0x98,0x11,0xa2,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1a21198 <unknown>

fmls    za.h[w8, 0], {z12.h - z13.h}, {z2.h - z3.h}  // 11000001-10100010-00010001-10011000
// CHECK-INST: fmls    za.h[w8, 0, vgx2], { z12.h, z13.h }, { z2.h, z3.h }
// CHECK-ENCODING: [0x98,0x11,0xa2,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1a21198 <unknown>

fmls    za.h[w10, 1, vgx2], {z0.h, z1.h}, {z26.h, z27.h}  // 11000001-10111010-01010000-00011001
// CHECK-INST: fmls    za.h[w10, 1, vgx2], { z0.h, z1.h }, { z26.h, z27.h }
// CHECK-ENCODING: [0x19,0x50,0xba,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1ba5019 <unknown>

fmls    za.h[w10, 1], {z0.h - z1.h}, {z26.h - z27.h}  // 11000001-10111010-01010000-00011001
// CHECK-INST: fmls    za.h[w10, 1, vgx2], { z0.h, z1.h }, { z26.h, z27.h }
// CHECK-ENCODING: [0x19,0x50,0xba,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1ba5019 <unknown>

fmls    za.h[w8, 5, vgx2], {z22.h, z23.h}, {z30.h, z31.h}  // 11000001-10111110-00010010-11011101
// CHECK-INST: fmls    za.h[w8, 5, vgx2], { z22.h, z23.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xdd,0x12,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1be12dd <unknown>

fmls    za.h[w8, 5], {z22.h - z23.h}, {z30.h - z31.h}  // 11000001-10111110-00010010-11011101
// CHECK-INST: fmls    za.h[w8, 5, vgx2], { z22.h, z23.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xdd,0x12,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1be12dd <unknown>

fmls    za.h[w11, 2, vgx2], {z8.h, z9.h}, {z0.h, z1.h}  // 11000001-10100000-01110001-00011010
// CHECK-INST: fmls    za.h[w11, 2, vgx2], { z8.h, z9.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x1a,0x71,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1a0711a <unknown>

fmls    za.h[w11, 2], {z8.h - z9.h}, {z0.h - z1.h}  // 11000001-10100000-01110001-00011010
// CHECK-INST: fmls    za.h[w11, 2, vgx2], { z8.h, z9.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x1a,0x71,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1a0711a <unknown>

fmls    za.h[w9, 7, vgx2], {z12.h, z13.h}, {z10.h, z11.h}  // 11000001-10101010-00110001-10011111
// CHECK-INST: fmls    za.h[w9, 7, vgx2], { z12.h, z13.h }, { z10.h, z11.h }
// CHECK-ENCODING: [0x9f,0x31,0xaa,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1aa319f <unknown>

fmls    za.h[w9, 7], {z12.h - z13.h}, {z10.h - z11.h}  // 11000001-10101010-00110001-10011111
// CHECK-INST: fmls    za.h[w9, 7, vgx2], { z12.h, z13.h }, { z10.h, z11.h }
// CHECK-ENCODING: [0x9f,0x31,0xaa,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1aa319f <unknown>

fmls    za.h[w8, 0, vgx4], {z0.h - z3.h}, z0.h  // 11000001-00110000-00011100-00001000
// CHECK-INST: fmls    za.h[w8, 0, vgx4], { z0.h - z3.h }, z0.h
// CHECK-ENCODING: [0x08,0x1c,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1301c08 <unknown>

fmls    za.h[w8, 0], {z0.h - z3.h}, z0.h  // 11000001-00110000-00011100-00001000
// CHECK-INST: fmls    za.h[w8, 0, vgx4], { z0.h - z3.h }, z0.h
// CHECK-ENCODING: [0x08,0x1c,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1301c08 <unknown>

fmls    za.h[w10, 5, vgx4], {z10.h - z13.h}, z5.h  // 11000001-00110101-01011101-01001101
// CHECK-INST: fmls    za.h[w10, 5, vgx4], { z10.h - z13.h }, z5.h
// CHECK-ENCODING: [0x4d,0x5d,0x35,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1355d4d <unknown>

fmls    za.h[w10, 5], {z10.h - z13.h}, z5.h  // 11000001-00110101-01011101-01001101
// CHECK-INST: fmls    za.h[w10, 5, vgx4], { z10.h - z13.h }, z5.h
// CHECK-ENCODING: [0x4d,0x5d,0x35,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1355d4d <unknown>

fmls    za.h[w11, 7, vgx4], {z13.h - z16.h}, z8.h  // 11000001-00111000-01111101-10101111
// CHECK-INST: fmls    za.h[w11, 7, vgx4], { z13.h - z16.h }, z8.h
// CHECK-ENCODING: [0xaf,0x7d,0x38,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1387daf <unknown>

fmls    za.h[w11, 7], {z13.h - z16.h}, z8.h  // 11000001-00111000-01111101-10101111
// CHECK-INST: fmls    za.h[w11, 7, vgx4], { z13.h - z16.h }, z8.h
// CHECK-ENCODING: [0xaf,0x7d,0x38,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1387daf <unknown>

fmls    za.h[w11, 7, vgx4], {z31.h, z0.h, z1.h, z2.h}, z15.h  // 11000001-00111111-01111111-11101111
// CHECK-INST: fmls    za.h[w11, 7, vgx4], { z31.h, z0.h, z1.h, z2.h }, z15.h
// CHECK-ENCODING: [0xef,0x7f,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c13f7fef <unknown>

fmls    za.h[w11, 7], {z31.h, z0.h, z1.h, z2.h}, z15.h  // 11000001-00111111-01111111-11101111
// CHECK-INST: fmls    za.h[w11, 7, vgx4], { z31.h, z0.h, z1.h, z2.h }, z15.h
// CHECK-ENCODING: [0xef,0x7f,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c13f7fef <unknown>

fmls    za.h[w8, 5, vgx4], {z17.h - z20.h}, z0.h  // 11000001-00110000-00011110-00101101
// CHECK-INST: fmls    za.h[w8, 5, vgx4], { z17.h - z20.h }, z0.h
// CHECK-ENCODING: [0x2d,0x1e,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1301e2d <unknown>

fmls    za.h[w8, 5], {z17.h - z20.h}, z0.h  // 11000001-00110000-00011110-00101101
// CHECK-INST: fmls    za.h[w8, 5, vgx4], { z17.h - z20.h }, z0.h
// CHECK-ENCODING: [0x2d,0x1e,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1301e2d <unknown>

fmls    za.h[w8, 1, vgx4], {z1.h - z4.h}, z14.h  // 11000001-00111110-00011100-00101001
// CHECK-INST: fmls    za.h[w8, 1, vgx4], { z1.h - z4.h }, z14.h
// CHECK-ENCODING: [0x29,0x1c,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c13e1c29 <unknown>

fmls    za.h[w8, 1], {z1.h - z4.h}, z14.h  // 11000001-00111110-00011100-00101001
// CHECK-INST: fmls    za.h[w8, 1, vgx4], { z1.h - z4.h }, z14.h
// CHECK-ENCODING: [0x29,0x1c,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c13e1c29 <unknown>

fmls    za.h[w10, 0, vgx4], {z19.h - z22.h}, z4.h  // 11000001-00110100-01011110-01101000
// CHECK-INST: fmls    za.h[w10, 0, vgx4], { z19.h - z22.h }, z4.h
// CHECK-ENCODING: [0x68,0x5e,0x34,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1345e68 <unknown>

fmls    za.h[w10, 0], {z19.h - z22.h}, z4.h  // 11000001-00110100-01011110-01101000
// CHECK-INST: fmls    za.h[w10, 0, vgx4], { z19.h - z22.h }, z4.h
// CHECK-ENCODING: [0x68,0x5e,0x34,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1345e68 <unknown>

fmls    za.h[w8, 0, vgx4], {z12.h - z15.h}, z2.h  // 11000001-00110010-00011101-10001000
// CHECK-INST: fmls    za.h[w8, 0, vgx4], { z12.h - z15.h }, z2.h
// CHECK-ENCODING: [0x88,0x1d,0x32,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1321d88 <unknown>

fmls    za.h[w8, 0], {z12.h - z15.h}, z2.h  // 11000001-00110010-00011101-10001000
// CHECK-INST: fmls    za.h[w8, 0, vgx4], { z12.h - z15.h }, z2.h
// CHECK-ENCODING: [0x88,0x1d,0x32,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1321d88 <unknown>

fmls    za.h[w10, 1, vgx4], {z1.h - z4.h}, z10.h  // 11000001-00111010-01011100-00101001
// CHECK-INST: fmls    za.h[w10, 1, vgx4], { z1.h - z4.h }, z10.h
// CHECK-ENCODING: [0x29,0x5c,0x3a,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c13a5c29 <unknown>

fmls    za.h[w10, 1], {z1.h - z4.h}, z10.h  // 11000001-00111010-01011100-00101001
// CHECK-INST: fmls    za.h[w10, 1, vgx4], { z1.h - z4.h }, z10.h
// CHECK-ENCODING: [0x29,0x5c,0x3a,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c13a5c29 <unknown>

fmls    za.h[w8, 5, vgx4], {z22.h - z25.h}, z14.h  // 11000001-00111110-00011110-11001101
// CHECK-INST: fmls    za.h[w8, 5, vgx4], { z22.h - z25.h }, z14.h
// CHECK-ENCODING: [0xcd,0x1e,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c13e1ecd <unknown>

fmls    za.h[w8, 5], {z22.h - z25.h}, z14.h  // 11000001-00111110-00011110-11001101
// CHECK-INST: fmls    za.h[w8, 5, vgx4], { z22.h - z25.h }, z14.h
// CHECK-ENCODING: [0xcd,0x1e,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c13e1ecd <unknown>

fmls    za.h[w11, 2, vgx4], {z9.h - z12.h}, z1.h  // 11000001-00110001-01111101-00101010
// CHECK-INST: fmls    za.h[w11, 2, vgx4], { z9.h - z12.h }, z1.h
// CHECK-ENCODING: [0x2a,0x7d,0x31,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1317d2a <unknown>

fmls    za.h[w11, 2], {z9.h - z12.h}, z1.h  // 11000001-00110001-01111101-00101010
// CHECK-INST: fmls    za.h[w11, 2, vgx4], { z9.h - z12.h }, z1.h
// CHECK-ENCODING: [0x2a,0x7d,0x31,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1317d2a <unknown>

fmls    za.h[w9, 7, vgx4], {z12.h - z15.h}, z11.h  // 11000001-00111011-00111101-10001111
// CHECK-INST: fmls    za.h[w9, 7, vgx4], { z12.h - z15.h }, z11.h
// CHECK-ENCODING: [0x8f,0x3d,0x3b,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c13b3d8f <unknown>

fmls    za.h[w9, 7], {z12.h - z15.h}, z11.h  // 11000001-00111011-00111101-10001111
// CHECK-INST: fmls    za.h[w9, 7, vgx4], { z12.h - z15.h }, z11.h
// CHECK-ENCODING: [0x8f,0x3d,0x3b,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c13b3d8f <unknown>

fmls    za.h[w8, 0, vgx4], {z0.h - z3.h}, z0.h[0]  // 11000001-00010000-10010000-00010000
// CHECK-INST: fmls    za.h[w8, 0, vgx4], { z0.h - z3.h }, z0.h[0]
// CHECK-ENCODING: [0x10,0x90,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1109010 <unknown>

fmls    za.h[w8, 0], {z0.h - z3.h}, z0.h[0]  // 11000001-00010000-10010000-00010000
// CHECK-INST: fmls    za.h[w8, 0, vgx4], { z0.h - z3.h }, z0.h[0]
// CHECK-ENCODING: [0x10,0x90,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1109010 <unknown>

fmls    za.h[w10, 5, vgx4], {z8.h - z11.h}, z5.h[2]  // 11000001-00010101-11010101-00010101
// CHECK-INST: fmls    za.h[w10, 5, vgx4], { z8.h - z11.h }, z5.h[2]
// CHECK-ENCODING: [0x15,0xd5,0x15,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c115d515 <unknown>

fmls    za.h[w10, 5], {z8.h - z11.h}, z5.h[2]  // 11000001-00010101-11010101-00010101
// CHECK-INST: fmls    za.h[w10, 5, vgx4], { z8.h - z11.h }, z5.h[2]
// CHECK-ENCODING: [0x15,0xd5,0x15,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c115d515 <unknown>

fmls    za.h[w11, 7, vgx4], {z12.h - z15.h}, z8.h[6]  // 11000001-00011000-11111101-10010111
// CHECK-INST: fmls    za.h[w11, 7, vgx4], { z12.h - z15.h }, z8.h[6]
// CHECK-ENCODING: [0x97,0xfd,0x18,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c118fd97 <unknown>

fmls    za.h[w11, 7], {z12.h - z15.h}, z8.h[6]  // 11000001-00011000-11111101-10010111
// CHECK-INST: fmls    za.h[w11, 7, vgx4], { z12.h - z15.h }, z8.h[6]
// CHECK-ENCODING: [0x97,0xfd,0x18,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c118fd97 <unknown>

fmls    za.h[w11, 7, vgx4], {z28.h - z31.h}, z15.h[7]  // 11000001-00011111-11111111-10011111
// CHECK-INST: fmls    za.h[w11, 7, vgx4], { z28.h - z31.h }, z15.h[7]
// CHECK-ENCODING: [0x9f,0xff,0x1f,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c11fff9f <unknown>

fmls    za.h[w11, 7], {z28.h - z31.h}, z15.h[7]  // 11000001-00011111-11111111-10011111
// CHECK-INST: fmls    za.h[w11, 7, vgx4], { z28.h - z31.h }, z15.h[7]
// CHECK-ENCODING: [0x9f,0xff,0x1f,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c11fff9f <unknown>

fmls    za.h[w8, 5, vgx4], {z16.h - z19.h}, z0.h[6]  // 11000001-00010000-10011110-00010101
// CHECK-INST: fmls    za.h[w8, 5, vgx4], { z16.h - z19.h }, z0.h[6]
// CHECK-ENCODING: [0x15,0x9e,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1109e15 <unknown>

fmls    za.h[w8, 5], {z16.h - z19.h}, z0.h[6]  // 11000001-00010000-10011110-00010101
// CHECK-INST: fmls    za.h[w8, 5, vgx4], { z16.h - z19.h }, z0.h[6]
// CHECK-ENCODING: [0x15,0x9e,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1109e15 <unknown>

fmls    za.h[w8, 1, vgx4], {z0.h - z3.h}, z14.h[2]  // 11000001-00011110-10010100-00010001
// CHECK-INST: fmls    za.h[w8, 1, vgx4], { z0.h - z3.h }, z14.h[2]
// CHECK-ENCODING: [0x11,0x94,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c11e9411 <unknown>

fmls    za.h[w8, 1], {z0.h - z3.h}, z14.h[2]  // 11000001-00011110-10010100-00010001
// CHECK-INST: fmls    za.h[w8, 1, vgx4], { z0.h - z3.h }, z14.h[2]
// CHECK-ENCODING: [0x11,0x94,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c11e9411 <unknown>

fmls    za.h[w10, 0, vgx4], {z16.h - z19.h}, z4.h[3]  // 11000001-00010100-11010110-00011000
// CHECK-INST: fmls    za.h[w10, 0, vgx4], { z16.h - z19.h }, z4.h[3]
// CHECK-ENCODING: [0x18,0xd6,0x14,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c114d618 <unknown>

fmls    za.h[w10, 0], {z16.h - z19.h}, z4.h[3]  // 11000001-00010100-11010110-00011000
// CHECK-INST: fmls    za.h[w10, 0, vgx4], { z16.h - z19.h }, z4.h[3]
// CHECK-ENCODING: [0x18,0xd6,0x14,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c114d618 <unknown>

fmls    za.h[w8, 0, vgx4], {z12.h - z15.h}, z2.h[4]  // 11000001-00010010-10011001-10010000
// CHECK-INST: fmls    za.h[w8, 0, vgx4], { z12.h - z15.h }, z2.h[4]
// CHECK-ENCODING: [0x90,0x99,0x12,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1129990 <unknown>

fmls    za.h[w8, 0], {z12.h - z15.h}, z2.h[4]  // 11000001-00010010-10011001-10010000
// CHECK-INST: fmls    za.h[w8, 0, vgx4], { z12.h - z15.h }, z2.h[4]
// CHECK-ENCODING: [0x90,0x99,0x12,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1129990 <unknown>

fmls    za.h[w10, 1, vgx4], {z0.h - z3.h}, z10.h[4]  // 11000001-00011010-11011000-00010001
// CHECK-INST: fmls    za.h[w10, 1, vgx4], { z0.h - z3.h }, z10.h[4]
// CHECK-ENCODING: [0x11,0xd8,0x1a,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c11ad811 <unknown>

fmls    za.h[w10, 1], {z0.h - z3.h}, z10.h[4]  // 11000001-00011010-11011000-00010001
// CHECK-INST: fmls    za.h[w10, 1, vgx4], { z0.h - z3.h }, z10.h[4]
// CHECK-ENCODING: [0x11,0xd8,0x1a,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c11ad811 <unknown>

fmls    za.h[w8, 5, vgx4], {z20.h - z23.h}, z14.h[5]  // 11000001-00011110-10011010-10011101
// CHECK-INST: fmls    za.h[w8, 5, vgx4], { z20.h - z23.h }, z14.h[5]
// CHECK-ENCODING: [0x9d,0x9a,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c11e9a9d <unknown>

fmls    za.h[w8, 5], {z20.h - z23.h}, z14.h[5]  // 11000001-00011110-10011010-10011101
// CHECK-INST: fmls    za.h[w8, 5, vgx4], { z20.h - z23.h }, z14.h[5]
// CHECK-ENCODING: [0x9d,0x9a,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c11e9a9d <unknown>

fmls    za.h[w11, 2, vgx4], {z8.h - z11.h}, z1.h[2]  // 11000001-00010001-11110101-00010010
// CHECK-INST: fmls    za.h[w11, 2, vgx4], { z8.h - z11.h }, z1.h[2]
// CHECK-ENCODING: [0x12,0xf5,0x11,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c111f512 <unknown>

fmls    za.h[w11, 2], {z8.h - z11.h}, z1.h[2]  // 11000001-00010001-11110101-00010010
// CHECK-INST: fmls    za.h[w11, 2, vgx4], { z8.h - z11.h }, z1.h[2]
// CHECK-ENCODING: [0x12,0xf5,0x11,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c111f512 <unknown>

fmls    za.h[w9, 7, vgx4], {z12.h - z15.h}, z11.h[4]  // 11000001-00011011-10111001-10010111
// CHECK-INST: fmls    za.h[w9, 7, vgx4], { z12.h - z15.h }, z11.h[4]
// CHECK-ENCODING: [0x97,0xb9,0x1b,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c11bb997 <unknown>

fmls    za.h[w9, 7], {z12.h - z15.h}, z11.h[4]  // 11000001-00011011-10111001-10010111
// CHECK-INST: fmls    za.h[w9, 7, vgx4], { z12.h - z15.h }, z11.h[4]
// CHECK-ENCODING: [0x97,0xb9,0x1b,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c11bb997 <unknown>

fmls    za.h[w8, 0, vgx4], {z0.h - z3.h}, {z0.h - z3.h}  // 11000001-10100001-00010000-00011000
// CHECK-INST: fmls    za.h[w8, 0, vgx4], { z0.h - z3.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x18,0x10,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1a11018 <unknown>

fmls    za.h[w8, 0], {z0.h - z3.h}, {z0.h - z3.h}  // 11000001-10100001-00010000-00011000
// CHECK-INST: fmls    za.h[w8, 0, vgx4], { z0.h - z3.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x18,0x10,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1a11018 <unknown>

fmls    za.h[w10, 5, vgx4], {z8.h - z11.h}, {z20.h - z23.h}  // 11000001-10110101-01010001-00011101
// CHECK-INST: fmls    za.h[w10, 5, vgx4], { z8.h - z11.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x1d,0x51,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1b5511d <unknown>

fmls    za.h[w10, 5], {z8.h - z11.h}, {z20.h - z23.h}  // 11000001-10110101-01010001-00011101
// CHECK-INST: fmls    za.h[w10, 5, vgx4], { z8.h - z11.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x1d,0x51,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1b5511d <unknown>

fmls    za.h[w11, 7, vgx4], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-10101001-01110001-10011111
// CHECK-INST: fmls    za.h[w11, 7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x9f,0x71,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1a9719f <unknown>

fmls    za.h[w11, 7], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-10101001-01110001-10011111
// CHECK-INST: fmls    za.h[w11, 7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x9f,0x71,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1a9719f <unknown>

fmls    za.h[w11, 7, vgx4], {z28.h - z31.h}, {z28.h - z31.h}  // 11000001-10111101-01110011-10011111
// CHECK-INST: fmls    za.h[w11, 7, vgx4], { z28.h - z31.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x9f,0x73,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1bd739f <unknown>

fmls    za.h[w11, 7], {z28.h - z31.h}, {z28.h - z31.h}  // 11000001-10111101-01110011-10011111
// CHECK-INST: fmls    za.h[w11, 7, vgx4], { z28.h - z31.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x9f,0x73,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1bd739f <unknown>

fmls    za.h[w8, 5, vgx4], {z16.h - z19.h}, {z16.h - z19.h}  // 11000001-10110001-00010010-00011101
// CHECK-INST: fmls    za.h[w8, 5, vgx4], { z16.h - z19.h }, { z16.h - z19.h }
// CHECK-ENCODING: [0x1d,0x12,0xb1,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1b1121d <unknown>

fmls    za.h[w8, 5], {z16.h - z19.h}, {z16.h - z19.h}  // 11000001-10110001-00010010-00011101
// CHECK-INST: fmls    za.h[w8, 5, vgx4], { z16.h - z19.h }, { z16.h - z19.h }
// CHECK-ENCODING: [0x1d,0x12,0xb1,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1b1121d <unknown>

fmls    za.h[w8, 1, vgx4], {z0.h - z3.h}, {z28.h - z31.h}  // 11000001-10111101-00010000-00011001
// CHECK-INST: fmls    za.h[w8, 1, vgx4], { z0.h - z3.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x19,0x10,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1bd1019 <unknown>

fmls    za.h[w8, 1], {z0.h - z3.h}, {z28.h - z31.h}  // 11000001-10111101-00010000-00011001
// CHECK-INST: fmls    za.h[w8, 1, vgx4], { z0.h - z3.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x19,0x10,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1bd1019 <unknown>

fmls    za.h[w10, 0, vgx4], {z16.h - z19.h}, {z20.h - z23.h}  // 11000001-10110101-01010010-00011000
// CHECK-INST: fmls    za.h[w10, 0, vgx4], { z16.h - z19.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x18,0x52,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1b55218 <unknown>

fmls    za.h[w10, 0], {z16.h - z19.h}, {z20.h - z23.h}  // 11000001-10110101-01010010-00011000
// CHECK-INST: fmls    za.h[w10, 0, vgx4], { z16.h - z19.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x18,0x52,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1b55218 <unknown>

fmls    za.h[w8, 0, vgx4], {z12.h - z15.h}, {z0.h - z3.h}  // 11000001-10100001-00010001-10011000
// CHECK-INST: fmls    za.h[w8, 0, vgx4], { z12.h - z15.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x98,0x11,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1a11198 <unknown>

fmls    za.h[w8, 0], {z12.h - z15.h}, {z0.h - z3.h}  // 11000001-10100001-00010001-10011000
// CHECK-INST: fmls    za.h[w8, 0, vgx4], { z12.h - z15.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x98,0x11,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1a11198 <unknown>

fmls    za.h[w10, 1, vgx4], {z0.h - z3.h}, {z24.h - z27.h}  // 11000001-10111001-01010000-00011001
// CHECK-INST: fmls    za.h[w10, 1, vgx4], { z0.h - z3.h }, { z24.h - z27.h }
// CHECK-ENCODING: [0x19,0x50,0xb9,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1b95019 <unknown>

fmls    za.h[w10, 1], {z0.h - z3.h}, {z24.h - z27.h}  // 11000001-10111001-01010000-00011001
// CHECK-INST: fmls    za.h[w10, 1, vgx4], { z0.h - z3.h }, { z24.h - z27.h }
// CHECK-ENCODING: [0x19,0x50,0xb9,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1b95019 <unknown>

fmls    za.h[w8, 5, vgx4], {z20.h - z23.h}, {z28.h - z31.h}  // 11000001-10111101-00010010-10011101
// CHECK-INST: fmls    za.h[w8, 5, vgx4], { z20.h - z23.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x9d,0x12,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1bd129d <unknown>

fmls    za.h[w8, 5], {z20.h - z23.h}, {z28.h - z31.h}  // 11000001-10111101-00010010-10011101
// CHECK-INST: fmls    za.h[w8, 5, vgx4], { z20.h - z23.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x9d,0x12,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1bd129d <unknown>

fmls    za.h[w11, 2, vgx4], {z8.h - z11.h}, {z0.h - z3.h}  // 11000001-10100001-01110001-00011010
// CHECK-INST: fmls    za.h[w11, 2, vgx4], { z8.h - z11.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x1a,0x71,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1a1711a <unknown>

fmls    za.h[w11, 2], {z8.h - z11.h}, {z0.h - z3.h}  // 11000001-10100001-01110001-00011010
// CHECK-INST: fmls    za.h[w11, 2, vgx4], { z8.h - z11.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x1a,0x71,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1a1711a <unknown>

fmls    za.h[w9, 7, vgx4], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-10101001-00110001-10011111
// CHECK-INST: fmls    za.h[w9, 7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x9f,0x31,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1a9319f <unknown>

fmls    za.h[w9, 7], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-10101001-00110001-10011111
// CHECK-INST: fmls    za.h[w9, 7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x9f,0x31,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1a9319f <unknown>
