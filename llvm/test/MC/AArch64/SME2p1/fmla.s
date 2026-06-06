// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme-f16f16 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme-f16f16 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme-f16f16 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme-f16f16 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2p1 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme-f16f16 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme-f16f16 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

fmla    za.h[w8, 0, vgx2], {z0.h, z1.h}, z0.h  // 11000001-00100000-00011100-00000000
// CHECK-INST: fmla    za.h[w8, 0, vgx2], { z0.h, z1.h }, z0.h
// CHECK-ENCODING: [0x00,0x1c,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1201c00 <unknown>

fmla    za.h[w8, 0], {z0.h - z1.h}, z0.h  // 11000001-00100000-00011100-00000000
// CHECK-INST: fmla    za.h[w8, 0, vgx2], { z0.h, z1.h }, z0.h
// CHECK-ENCODING: [0x00,0x1c,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1201c00 <unknown>

fmla    za.h[w10, 5, vgx2], {z10.h, z11.h}, z5.h  // 11000001-00100101-01011101-01000101
// CHECK-INST: fmla    za.h[w10, 5, vgx2], { z10.h, z11.h }, z5.h
// CHECK-ENCODING: [0x45,0x5d,0x25,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1255d45 <unknown>

fmla    za.h[w10, 5], {z10.h - z11.h}, z5.h  // 11000001-00100101-01011101-01000101
// CHECK-INST: fmla    za.h[w10, 5, vgx2], { z10.h, z11.h }, z5.h
// CHECK-ENCODING: [0x45,0x5d,0x25,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1255d45 <unknown>

fmla    za.h[w11, 7, vgx2], {z13.h, z14.h}, z8.h  // 11000001-00101000-01111101-10100111
// CHECK-INST: fmla    za.h[w11, 7, vgx2], { z13.h, z14.h }, z8.h
// CHECK-ENCODING: [0xa7,0x7d,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1287da7 <unknown>

fmla    za.h[w11, 7], {z13.h - z14.h}, z8.h  // 11000001-00101000-01111101-10100111
// CHECK-INST: fmla    za.h[w11, 7, vgx2], { z13.h, z14.h }, z8.h
// CHECK-ENCODING: [0xa7,0x7d,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1287da7 <unknown>

fmla    za.h[w11, 7, vgx2], {z31.h, z0.h}, z15.h  // 11000001-00101111-01111111-11100111
// CHECK-INST: fmla    za.h[w11, 7, vgx2], { z31.h, z0.h }, z15.h
// CHECK-ENCODING: [0xe7,0x7f,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c12f7fe7 <unknown>

fmla    za.h[w11, 7], {z31.h - z0.h}, z15.h  // 11000001-00101111-01111111-11100111
// CHECK-INST: fmla    za.h[w11, 7, vgx2], { z31.h, z0.h }, z15.h
// CHECK-ENCODING: [0xe7,0x7f,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c12f7fe7 <unknown>

fmla    za.h[w8, 5, vgx2], {z17.h, z18.h}, z0.h  // 11000001-00100000-00011110-00100101
// CHECK-INST: fmla    za.h[w8, 5, vgx2], { z17.h, z18.h }, z0.h
// CHECK-ENCODING: [0x25,0x1e,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1201e25 <unknown>

fmla    za.h[w8, 5], {z17.h - z18.h}, z0.h  // 11000001-00100000-00011110-00100101
// CHECK-INST: fmla    za.h[w8, 5, vgx2], { z17.h, z18.h }, z0.h
// CHECK-ENCODING: [0x25,0x1e,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1201e25 <unknown>

fmla    za.h[w8, 1, vgx2], {z1.h, z2.h}, z14.h  // 11000001-00101110-00011100-00100001
// CHECK-INST: fmla    za.h[w8, 1, vgx2], { z1.h, z2.h }, z14.h
// CHECK-ENCODING: [0x21,0x1c,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c12e1c21 <unknown>

fmla    za.h[w8, 1], {z1.h - z2.h}, z14.h  // 11000001-00101110-00011100-00100001
// CHECK-INST: fmla    za.h[w8, 1, vgx2], { z1.h, z2.h }, z14.h
// CHECK-ENCODING: [0x21,0x1c,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c12e1c21 <unknown>

fmla    za.h[w10, 0, vgx2], {z19.h, z20.h}, z4.h  // 11000001-00100100-01011110-01100000
// CHECK-INST: fmla    za.h[w10, 0, vgx2], { z19.h, z20.h }, z4.h
// CHECK-ENCODING: [0x60,0x5e,0x24,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1245e60 <unknown>

fmla    za.h[w10, 0], {z19.h - z20.h}, z4.h  // 11000001-00100100-01011110-01100000
// CHECK-INST: fmla    za.h[w10, 0, vgx2], { z19.h, z20.h }, z4.h
// CHECK-ENCODING: [0x60,0x5e,0x24,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1245e60 <unknown>

fmla    za.h[w8, 0, vgx2], {z12.h, z13.h}, z2.h  // 11000001-00100010-00011101-10000000
// CHECK-INST: fmla    za.h[w8, 0, vgx2], { z12.h, z13.h }, z2.h
// CHECK-ENCODING: [0x80,0x1d,0x22,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1221d80 <unknown>

fmla    za.h[w8, 0], {z12.h - z13.h}, z2.h  // 11000001-00100010-00011101-10000000
// CHECK-INST: fmla    za.h[w8, 0, vgx2], { z12.h, z13.h }, z2.h
// CHECK-ENCODING: [0x80,0x1d,0x22,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1221d80 <unknown>

fmla    za.h[w10, 1, vgx2], {z1.h, z2.h}, z10.h  // 11000001-00101010-01011100-00100001
// CHECK-INST: fmla    za.h[w10, 1, vgx2], { z1.h, z2.h }, z10.h
// CHECK-ENCODING: [0x21,0x5c,0x2a,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c12a5c21 <unknown>

fmla    za.h[w10, 1], {z1.h - z2.h}, z10.h  // 11000001-00101010-01011100-00100001
// CHECK-INST: fmla    za.h[w10, 1, vgx2], { z1.h, z2.h }, z10.h
// CHECK-ENCODING: [0x21,0x5c,0x2a,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c12a5c21 <unknown>

fmla    za.h[w8, 5, vgx2], {z22.h, z23.h}, z14.h  // 11000001-00101110-00011110-11000101
// CHECK-INST: fmla    za.h[w8, 5, vgx2], { z22.h, z23.h }, z14.h
// CHECK-ENCODING: [0xc5,0x1e,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c12e1ec5 <unknown>

fmla    za.h[w8, 5], {z22.h - z23.h}, z14.h  // 11000001-00101110-00011110-11000101
// CHECK-INST: fmla    za.h[w8, 5, vgx2], { z22.h, z23.h }, z14.h
// CHECK-ENCODING: [0xc5,0x1e,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c12e1ec5 <unknown>

fmla    za.h[w11, 2, vgx2], {z9.h, z10.h}, z1.h  // 11000001-00100001-01111101-00100010
// CHECK-INST: fmla    za.h[w11, 2, vgx2], { z9.h, z10.h }, z1.h
// CHECK-ENCODING: [0x22,0x7d,0x21,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1217d22 <unknown>

fmla    za.h[w11, 2], {z9.h - z10.h}, z1.h  // 11000001-00100001-01111101-00100010
// CHECK-INST: fmla    za.h[w11, 2, vgx2], { z9.h, z10.h }, z1.h
// CHECK-ENCODING: [0x22,0x7d,0x21,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1217d22 <unknown>

fmla    za.h[w9, 7, vgx2], {z12.h, z13.h}, z11.h  // 11000001-00101011-00111101-10000111
// CHECK-INST: fmla    za.h[w9, 7, vgx2], { z12.h, z13.h }, z11.h
// CHECK-ENCODING: [0x87,0x3d,0x2b,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c12b3d87 <unknown>

fmla    za.h[w9, 7], {z12.h - z13.h}, z11.h  // 11000001-00101011-00111101-10000111
// CHECK-INST: fmla    za.h[w9, 7, vgx2], { z12.h, z13.h }, z11.h
// CHECK-ENCODING: [0x87,0x3d,0x2b,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c12b3d87 <unknown>

fmla    za.h[w8, 0, vgx2], {z0.h, z1.h}, z0.h[0]  // 11000001-00010000-00010000-00000000
// CHECK-INST: fmla    za.h[w8, 0, vgx2], { z0.h, z1.h }, z0.h[0]
// CHECK-ENCODING: [0x00,0x10,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1101000 <unknown>

fmla    za.h[w8, 0], {z0.h - z1.h}, z0.h[0]  // 11000001-00010000-00010000-00000000
// CHECK-INST: fmla    za.h[w8, 0, vgx2], { z0.h, z1.h }, z0.h[0]
// CHECK-ENCODING: [0x00,0x10,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1101000 <unknown>

fmla    za.h[w10, 5, vgx2], {z10.h, z11.h}, z5.h[2]  // 11000001-00010101-01010101-01000101
// CHECK-INST: fmla    za.h[w10, 5, vgx2], { z10.h, z11.h }, z5.h[2]
// CHECK-ENCODING: [0x45,0x55,0x15,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1155545 <unknown>

fmla    za.h[w10, 5], {z10.h - z11.h}, z5.h[2]  // 11000001-00010101-01010101-01000101
// CHECK-INST: fmla    za.h[w10, 5, vgx2], { z10.h, z11.h }, z5.h[2]
// CHECK-ENCODING: [0x45,0x55,0x15,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1155545 <unknown>

fmla    za.h[w11, 7, vgx2], {z12.h, z13.h}, z8.h[6]  // 11000001-00011000-01111101-10000111
// CHECK-INST: fmla    za.h[w11, 7, vgx2], { z12.h, z13.h }, z8.h[6]
// CHECK-ENCODING: [0x87,0x7d,0x18,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1187d87 <unknown>

fmla    za.h[w11, 7], {z12.h - z13.h}, z8.h[6]  // 11000001-00011000-01111101-10000111
// CHECK-INST: fmla    za.h[w11, 7, vgx2], { z12.h, z13.h }, z8.h[6]
// CHECK-ENCODING: [0x87,0x7d,0x18,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1187d87 <unknown>

fmla    za.h[w11, 7, vgx2], {z30.h, z31.h}, z15.h[7]  // 11000001-00011111-01111111-11001111
// CHECK-INST: fmla    za.h[w11, 7, vgx2], { z30.h, z31.h }, z15.h[7]
// CHECK-ENCODING: [0xcf,0x7f,0x1f,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c11f7fcf <unknown>

fmla    za.h[w11, 7], {z30.h - z31.h}, z15.h[7]  // 11000001-00011111-01111111-11001111
// CHECK-INST: fmla    za.h[w11, 7, vgx2], { z30.h, z31.h }, z15.h[7]
// CHECK-ENCODING: [0xcf,0x7f,0x1f,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c11f7fcf <unknown>

fmla    za.h[w8, 5, vgx2], {z16.h, z17.h}, z0.h[6]  // 11000001-00010000-00011110-00000101
// CHECK-INST: fmla    za.h[w8, 5, vgx2], { z16.h, z17.h }, z0.h[6]
// CHECK-ENCODING: [0x05,0x1e,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1101e05 <unknown>

fmla    za.h[w8, 5], {z16.h - z17.h}, z0.h[6]  // 11000001-00010000-00011110-00000101
// CHECK-INST: fmla    za.h[w8, 5, vgx2], { z16.h, z17.h }, z0.h[6]
// CHECK-ENCODING: [0x05,0x1e,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1101e05 <unknown>

fmla    za.h[w8, 1, vgx2], {z0.h, z1.h}, z14.h[2]  // 11000001-00011110-00010100-00000001
// CHECK-INST: fmla    za.h[w8, 1, vgx2], { z0.h, z1.h }, z14.h[2]
// CHECK-ENCODING: [0x01,0x14,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c11e1401 <unknown>

fmla    za.h[w8, 1], {z0.h - z1.h}, z14.h[2]  // 11000001-00011110-00010100-00000001
// CHECK-INST: fmla    za.h[w8, 1, vgx2], { z0.h, z1.h }, z14.h[2]
// CHECK-ENCODING: [0x01,0x14,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c11e1401 <unknown>

fmla    za.h[w10, 0, vgx2], {z18.h, z19.h}, z4.h[3]  // 11000001-00010100-01010110-01001000
// CHECK-INST: fmla    za.h[w10, 0, vgx2], { z18.h, z19.h }, z4.h[3]
// CHECK-ENCODING: [0x48,0x56,0x14,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1145648 <unknown>

fmla    za.h[w10, 0], {z18.h - z19.h}, z4.h[3]  // 11000001-00010100-01010110-01001000
// CHECK-INST: fmla    za.h[w10, 0, vgx2], { z18.h, z19.h }, z4.h[3]
// CHECK-ENCODING: [0x48,0x56,0x14,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1145648 <unknown>

fmla    za.h[w8, 0, vgx2], {z12.h, z13.h}, z2.h[4]  // 11000001-00010010-00011001-10000000
// CHECK-INST: fmla    za.h[w8, 0, vgx2], { z12.h, z13.h }, z2.h[4]
// CHECK-ENCODING: [0x80,0x19,0x12,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1121980 <unknown>

fmla    za.h[w8, 0], {z12.h - z13.h}, z2.h[4]  // 11000001-00010010-00011001-10000000
// CHECK-INST: fmla    za.h[w8, 0, vgx2], { z12.h, z13.h }, z2.h[4]
// CHECK-ENCODING: [0x80,0x19,0x12,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1121980 <unknown>

fmla    za.h[w10, 1, vgx2], {z0.h, z1.h}, z10.h[4]  // 11000001-00011010-01011000-00000001
// CHECK-INST: fmla    za.h[w10, 1, vgx2], { z0.h, z1.h }, z10.h[4]
// CHECK-ENCODING: [0x01,0x58,0x1a,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c11a5801 <unknown>

fmla    za.h[w10, 1], {z0.h - z1.h}, z10.h[4]  // 11000001-00011010-01011000-00000001
// CHECK-INST: fmla    za.h[w10, 1, vgx2], { z0.h, z1.h }, z10.h[4]
// CHECK-ENCODING: [0x01,0x58,0x1a,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c11a5801 <unknown>

fmla    za.h[w8, 5, vgx2], {z22.h, z23.h}, z14.h[5]  // 11000001-00011110-00011010-11001101
// CHECK-INST: fmla    za.h[w8, 5, vgx2], { z22.h, z23.h }, z14.h[5]
// CHECK-ENCODING: [0xcd,0x1a,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c11e1acd <unknown>

fmla    za.h[w8, 5], {z22.h - z23.h}, z14.h[5]  // 11000001-00011110-00011010-11001101
// CHECK-INST: fmla    za.h[w8, 5, vgx2], { z22.h, z23.h }, z14.h[5]
// CHECK-ENCODING: [0xcd,0x1a,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c11e1acd <unknown>

fmla    za.h[w11, 2, vgx2], {z8.h, z9.h}, z1.h[2]  // 11000001-00010001-01110101-00000010
// CHECK-INST: fmla    za.h[w11, 2, vgx2], { z8.h, z9.h }, z1.h[2]
// CHECK-ENCODING: [0x02,0x75,0x11,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1117502 <unknown>

fmla    za.h[w11, 2], {z8.h - z9.h}, z1.h[2]  // 11000001-00010001-01110101-00000010
// CHECK-INST: fmla    za.h[w11, 2, vgx2], { z8.h, z9.h }, z1.h[2]
// CHECK-ENCODING: [0x02,0x75,0x11,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1117502 <unknown>

fmla    za.h[w9, 7, vgx2], {z12.h, z13.h}, z11.h[4]  // 11000001-00011011-00111001-10000111
// CHECK-INST: fmla    za.h[w9, 7, vgx2], { z12.h, z13.h }, z11.h[4]
// CHECK-ENCODING: [0x87,0x39,0x1b,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c11b3987 <unknown>

fmla    za.h[w9, 7], {z12.h - z13.h}, z11.h[4]  // 11000001-00011011-00111001-10000111
// CHECK-INST: fmla    za.h[w9, 7, vgx2], { z12.h, z13.h }, z11.h[4]
// CHECK-ENCODING: [0x87,0x39,0x1b,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c11b3987 <unknown>

fmla    za.h[w8, 0, vgx2], {z0.h, z1.h}, {z0.h, z1.h}  // 11000001-10100000-00010000-00001000
// CHECK-INST: fmla    za.h[w8, 0, vgx2], { z0.h, z1.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x08,0x10,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1a01008 <unknown>

fmla    za.h[w8, 0], {z0.h - z1.h}, {z0.h - z1.h}  // 11000001-10100000-00010000-00001000
// CHECK-INST: fmla    za.h[w8, 0, vgx2], { z0.h, z1.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x08,0x10,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1a01008 <unknown>

fmla    za.h[w10, 5, vgx2], {z10.h, z11.h}, {z20.h, z21.h}  // 11000001-10110100-01010001-01001101
// CHECK-INST: fmla    za.h[w10, 5, vgx2], { z10.h, z11.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x4d,0x51,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1b4514d <unknown>

fmla    za.h[w10, 5], {z10.h - z11.h}, {z20.h - z21.h}  // 11000001-10110100-01010001-01001101
// CHECK-INST: fmla    za.h[w10, 5, vgx2], { z10.h, z11.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x4d,0x51,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1b4514d <unknown>

fmla    za.h[w11, 7, vgx2], {z12.h, z13.h}, {z8.h, z9.h}  // 11000001-10101000-01110001-10001111
// CHECK-INST: fmla    za.h[w11, 7, vgx2], { z12.h, z13.h }, { z8.h, z9.h }
// CHECK-ENCODING: [0x8f,0x71,0xa8,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1a8718f <unknown>

fmla    za.h[w11, 7], {z12.h - z13.h}, {z8.h - z9.h}  // 11000001-10101000-01110001-10001111
// CHECK-INST: fmla    za.h[w11, 7, vgx2], { z12.h, z13.h }, { z8.h, z9.h }
// CHECK-ENCODING: [0x8f,0x71,0xa8,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1a8718f <unknown>

fmla    za.h[w11, 7, vgx2], {z30.h, z31.h}, {z30.h, z31.h}  // 11000001-10111110-01110011-11001111
// CHECK-INST: fmla    za.h[w11, 7, vgx2], { z30.h, z31.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xcf,0x73,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1be73cf <unknown>

fmla    za.h[w11, 7], {z30.h - z31.h}, {z30.h - z31.h}  // 11000001-10111110-01110011-11001111
// CHECK-INST: fmla    za.h[w11, 7, vgx2], { z30.h, z31.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xcf,0x73,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1be73cf <unknown>

fmla    za.h[w8, 5, vgx2], {z16.h, z17.h}, {z16.h, z17.h}  // 11000001-10110000-00010010-00001101
// CHECK-INST: fmla    za.h[w8, 5, vgx2], { z16.h, z17.h }, { z16.h, z17.h }
// CHECK-ENCODING: [0x0d,0x12,0xb0,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1b0120d <unknown>

fmla    za.h[w8, 5], {z16.h - z17.h}, {z16.h - z17.h}  // 11000001-10110000-00010010-00001101
// CHECK-INST: fmla    za.h[w8, 5, vgx2], { z16.h, z17.h }, { z16.h, z17.h }
// CHECK-ENCODING: [0x0d,0x12,0xb0,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1b0120d <unknown>

fmla    za.h[w8, 1, vgx2], {z0.h, z1.h}, {z30.h, z31.h}  // 11000001-10111110-00010000-00001001
// CHECK-INST: fmla    za.h[w8, 1, vgx2], { z0.h, z1.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0x09,0x10,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1be1009 <unknown>

fmla    za.h[w8, 1], {z0.h - z1.h}, {z30.h - z31.h}  // 11000001-10111110-00010000-00001001
// CHECK-INST: fmla    za.h[w8, 1, vgx2], { z0.h, z1.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0x09,0x10,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1be1009 <unknown>

fmla    za.h[w10, 0, vgx2], {z18.h, z19.h}, {z20.h, z21.h}  // 11000001-10110100-01010010-01001000
// CHECK-INST: fmla    za.h[w10, 0, vgx2], { z18.h, z19.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x48,0x52,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1b45248 <unknown>

fmla    za.h[w10, 0], {z18.h - z19.h}, {z20.h - z21.h}  // 11000001-10110100-01010010-01001000
// CHECK-INST: fmla    za.h[w10, 0, vgx2], { z18.h, z19.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x48,0x52,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1b45248 <unknown>

fmla    za.h[w8, 0, vgx2], {z12.h, z13.h}, {z2.h, z3.h}  // 11000001-10100010-00010001-10001000
// CHECK-INST: fmla    za.h[w8, 0, vgx2], { z12.h, z13.h }, { z2.h, z3.h }
// CHECK-ENCODING: [0x88,0x11,0xa2,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1a21188 <unknown>

fmla    za.h[w8, 0], {z12.h - z13.h}, {z2.h - z3.h}  // 11000001-10100010-00010001-10001000
// CHECK-INST: fmla    za.h[w8, 0, vgx2], { z12.h, z13.h }, { z2.h, z3.h }
// CHECK-ENCODING: [0x88,0x11,0xa2,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1a21188 <unknown>

fmla    za.h[w10, 1, vgx2], {z0.h, z1.h}, {z26.h, z27.h}  // 11000001-10111010-01010000-00001001
// CHECK-INST: fmla    za.h[w10, 1, vgx2], { z0.h, z1.h }, { z26.h, z27.h }
// CHECK-ENCODING: [0x09,0x50,0xba,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1ba5009 <unknown>

fmla    za.h[w10, 1], {z0.h - z1.h}, {z26.h - z27.h}  // 11000001-10111010-01010000-00001001
// CHECK-INST: fmla    za.h[w10, 1, vgx2], { z0.h, z1.h }, { z26.h, z27.h }
// CHECK-ENCODING: [0x09,0x50,0xba,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1ba5009 <unknown>

fmla    za.h[w8, 5, vgx2], {z22.h, z23.h}, {z30.h, z31.h}  // 11000001-10111110-00010010-11001101
// CHECK-INST: fmla    za.h[w8, 5, vgx2], { z22.h, z23.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xcd,0x12,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1be12cd <unknown>

fmla    za.h[w8, 5], {z22.h - z23.h}, {z30.h - z31.h}  // 11000001-10111110-00010010-11001101
// CHECK-INST: fmla    za.h[w8, 5, vgx2], { z22.h, z23.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xcd,0x12,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1be12cd <unknown>

fmla    za.h[w11, 2, vgx2], {z8.h, z9.h}, {z0.h, z1.h}  // 11000001-10100000-01110001-00001010
// CHECK-INST: fmla    za.h[w11, 2, vgx2], { z8.h, z9.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x0a,0x71,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1a0710a <unknown>

fmla    za.h[w11, 2], {z8.h - z9.h}, {z0.h - z1.h}  // 11000001-10100000-01110001-00001010
// CHECK-INST: fmla    za.h[w11, 2, vgx2], { z8.h, z9.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x0a,0x71,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1a0710a <unknown>

fmla    za.h[w9, 7, vgx2], {z12.h, z13.h}, {z10.h, z11.h}  // 11000001-10101010-00110001-10001111
// CHECK-INST: fmla    za.h[w9, 7, vgx2], { z12.h, z13.h }, { z10.h, z11.h }
// CHECK-ENCODING: [0x8f,0x31,0xaa,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1aa318f <unknown>

fmla    za.h[w9, 7], {z12.h - z13.h}, {z10.h - z11.h}  // 11000001-10101010-00110001-10001111
// CHECK-INST: fmla    za.h[w9, 7, vgx2], { z12.h, z13.h }, { z10.h, z11.h }
// CHECK-ENCODING: [0x8f,0x31,0xaa,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1aa318f <unknown>


fmla    za.h[w8, 0, vgx4], {z0.h - z3.h}, z0.h  // 11000001-00110000-00011100-00000000
// CHECK-INST: fmla    za.h[w8, 0, vgx4], { z0.h - z3.h }, z0.h
// CHECK-ENCODING: [0x00,0x1c,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1301c00 <unknown>

fmla    za.h[w8, 0], {z0.h - z3.h}, z0.h  // 11000001-00110000-00011100-00000000
// CHECK-INST: fmla    za.h[w8, 0, vgx4], { z0.h - z3.h }, z0.h
// CHECK-ENCODING: [0x00,0x1c,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1301c00 <unknown>

fmla    za.h[w10, 5, vgx4], {z10.h - z13.h}, z5.h  // 11000001-00110101-01011101-01000101
// CHECK-INST: fmla    za.h[w10, 5, vgx4], { z10.h - z13.h }, z5.h
// CHECK-ENCODING: [0x45,0x5d,0x35,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1355d45 <unknown>

fmla    za.h[w10, 5], {z10.h - z13.h}, z5.h  // 11000001-00110101-01011101-01000101
// CHECK-INST: fmla    za.h[w10, 5, vgx4], { z10.h - z13.h }, z5.h
// CHECK-ENCODING: [0x45,0x5d,0x35,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1355d45 <unknown>

fmla    za.h[w11, 7, vgx4], {z13.h - z16.h}, z8.h  // 11000001-00111000-01111101-10100111
// CHECK-INST: fmla    za.h[w11, 7, vgx4], { z13.h - z16.h }, z8.h
// CHECK-ENCODING: [0xa7,0x7d,0x38,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1387da7 <unknown>

fmla    za.h[w11, 7], {z13.h - z16.h}, z8.h  // 11000001-00111000-01111101-10100111
// CHECK-INST: fmla    za.h[w11, 7, vgx4], { z13.h - z16.h }, z8.h
// CHECK-ENCODING: [0xa7,0x7d,0x38,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1387da7 <unknown>

fmla    za.h[w11, 7, vgx4], {z31.h, z0.h, z1.h, z2.h}, z15.h  // 11000001-00111111-01111111-11100111
// CHECK-INST: fmla    za.h[w11, 7, vgx4], { z31.h, z0.h, z1.h, z2.h }, z15.h
// CHECK-ENCODING: [0xe7,0x7f,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c13f7fe7 <unknown>

fmla    za.h[w11, 7], {z31.h, z0.h, z1.h, z2.h}, z15.h  // 11000001-00111111-01111111-11100111
// CHECK-INST: fmla    za.h[w11, 7, vgx4], { z31.h, z0.h, z1.h, z2.h }, z15.h
// CHECK-ENCODING: [0xe7,0x7f,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c13f7fe7 <unknown>

fmla    za.h[w8, 5, vgx4], {z17.h - z20.h}, z0.h  // 11000001-00110000-00011110-00100101
// CHECK-INST: fmla    za.h[w8, 5, vgx4], { z17.h - z20.h }, z0.h
// CHECK-ENCODING: [0x25,0x1e,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1301e25 <unknown>

fmla    za.h[w8, 5], {z17.h - z20.h}, z0.h  // 11000001-00110000-00011110-00100101
// CHECK-INST: fmla    za.h[w8, 5, vgx4], { z17.h - z20.h }, z0.h
// CHECK-ENCODING: [0x25,0x1e,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1301e25 <unknown>

fmla    za.h[w8, 1, vgx4], {z1.h - z4.h}, z14.h  // 11000001-00111110-00011100-00100001
// CHECK-INST: fmla    za.h[w8, 1, vgx4], { z1.h - z4.h }, z14.h
// CHECK-ENCODING: [0x21,0x1c,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c13e1c21 <unknown>

fmla    za.h[w8, 1], {z1.h - z4.h}, z14.h  // 11000001-00111110-00011100-00100001
// CHECK-INST: fmla    za.h[w8, 1, vgx4], { z1.h - z4.h }, z14.h
// CHECK-ENCODING: [0x21,0x1c,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c13e1c21 <unknown>

fmla    za.h[w10, 0, vgx4], {z19.h - z22.h}, z4.h  // 11000001-00110100-01011110-01100000
// CHECK-INST: fmla    za.h[w10, 0, vgx4], { z19.h - z22.h }, z4.h
// CHECK-ENCODING: [0x60,0x5e,0x34,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1345e60 <unknown>

fmla    za.h[w10, 0], {z19.h - z22.h}, z4.h  // 11000001-00110100-01011110-01100000
// CHECK-INST: fmla    za.h[w10, 0, vgx4], { z19.h - z22.h }, z4.h
// CHECK-ENCODING: [0x60,0x5e,0x34,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1345e60 <unknown>

fmla    za.h[w8, 0, vgx4], {z12.h - z15.h}, z2.h  // 11000001-00110010-00011101-10000000
// CHECK-INST: fmla    za.h[w8, 0, vgx4], { z12.h - z15.h }, z2.h
// CHECK-ENCODING: [0x80,0x1d,0x32,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1321d80 <unknown>

fmla    za.h[w8, 0], {z12.h - z15.h}, z2.h  // 11000001-00110010-00011101-10000000
// CHECK-INST: fmla    za.h[w8, 0, vgx4], { z12.h - z15.h }, z2.h
// CHECK-ENCODING: [0x80,0x1d,0x32,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1321d80 <unknown>

fmla    za.h[w10, 1, vgx4], {z1.h - z4.h}, z10.h  // 11000001-00111010-01011100-00100001
// CHECK-INST: fmla    za.h[w10, 1, vgx4], { z1.h - z4.h }, z10.h
// CHECK-ENCODING: [0x21,0x5c,0x3a,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c13a5c21 <unknown>

fmla    za.h[w10, 1], {z1.h - z4.h}, z10.h  // 11000001-00111010-01011100-00100001
// CHECK-INST: fmla    za.h[w10, 1, vgx4], { z1.h - z4.h }, z10.h
// CHECK-ENCODING: [0x21,0x5c,0x3a,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c13a5c21 <unknown>

fmla    za.h[w8, 5, vgx4], {z22.h - z25.h}, z14.h  // 11000001-00111110-00011110-11000101
// CHECK-INST: fmla    za.h[w8, 5, vgx4], { z22.h - z25.h }, z14.h
// CHECK-ENCODING: [0xc5,0x1e,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c13e1ec5 <unknown>

fmla    za.h[w8, 5], {z22.h - z25.h}, z14.h  // 11000001-00111110-00011110-11000101
// CHECK-INST: fmla    za.h[w8, 5, vgx4], { z22.h - z25.h }, z14.h
// CHECK-ENCODING: [0xc5,0x1e,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c13e1ec5 <unknown>

fmla    za.h[w11, 2, vgx4], {z9.h - z12.h}, z1.h  // 11000001-00110001-01111101-00100010
// CHECK-INST: fmla    za.h[w11, 2, vgx4], { z9.h - z12.h }, z1.h
// CHECK-ENCODING: [0x22,0x7d,0x31,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1317d22 <unknown>

fmla    za.h[w11, 2], {z9.h - z12.h}, z1.h  // 11000001-00110001-01111101-00100010
// CHECK-INST: fmla    za.h[w11, 2, vgx4], { z9.h - z12.h }, z1.h
// CHECK-ENCODING: [0x22,0x7d,0x31,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1317d22 <unknown>

fmla    za.h[w9, 7, vgx4], {z12.h - z15.h}, z11.h  // 11000001-00111011-00111101-10000111
// CHECK-INST: fmla    za.h[w9, 7, vgx4], { z12.h - z15.h }, z11.h
// CHECK-ENCODING: [0x87,0x3d,0x3b,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c13b3d87 <unknown>

fmla    za.h[w9, 7], {z12.h - z15.h}, z11.h  // 11000001-00111011-00111101-10000111
// CHECK-INST: fmla    za.h[w9, 7, vgx4], { z12.h - z15.h }, z11.h
// CHECK-ENCODING: [0x87,0x3d,0x3b,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c13b3d87 <unknown>

fmla    za.h[w8, 0, vgx4], {z0.h - z3.h}, z0.h[0]  // 11000001-00010000-10010000-00000000
// CHECK-INST: fmla    za.h[w8, 0, vgx4], { z0.h - z3.h }, z0.h[0]
// CHECK-ENCODING: [0x00,0x90,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1109000 <unknown>

fmla    za.h[w8, 0], {z0.h - z3.h}, z0.h[0]  // 11000001-00010000-10010000-00000000
// CHECK-INST: fmla    za.h[w8, 0, vgx4], { z0.h - z3.h }, z0.h[0]
// CHECK-ENCODING: [0x00,0x90,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1109000 <unknown>

fmla    za.h[w10, 5, vgx4], {z8.h - z11.h}, z5.h[2]  // 11000001-00010101-11010101-00000101
// CHECK-INST: fmla    za.h[w10, 5, vgx4], { z8.h - z11.h }, z5.h[2]
// CHECK-ENCODING: [0x05,0xd5,0x15,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c115d505 <unknown>

fmla    za.h[w10, 5], {z8.h - z11.h}, z5.h[2]  // 11000001-00010101-11010101-00000101
// CHECK-INST: fmla    za.h[w10, 5, vgx4], { z8.h - z11.h }, z5.h[2]
// CHECK-ENCODING: [0x05,0xd5,0x15,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c115d505 <unknown>

fmla    za.h[w11, 7, vgx4], {z12.h - z15.h}, z8.h[6]  // 11000001-00011000-11111101-10000111
// CHECK-INST: fmla    za.h[w11, 7, vgx4], { z12.h - z15.h }, z8.h[6]
// CHECK-ENCODING: [0x87,0xfd,0x18,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c118fd87 <unknown>

fmla    za.h[w11, 7], {z12.h - z15.h}, z8.h[6]  // 11000001-00011000-11111101-10000111
// CHECK-INST: fmla    za.h[w11, 7, vgx4], { z12.h - z15.h }, z8.h[6]
// CHECK-ENCODING: [0x87,0xfd,0x18,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c118fd87 <unknown>

fmla    za.h[w11, 7, vgx4], {z28.h - z31.h}, z15.h[7]  // 11000001-00011111-11111111-10001111
// CHECK-INST: fmla    za.h[w11, 7, vgx4], { z28.h - z31.h }, z15.h[7]
// CHECK-ENCODING: [0x8f,0xff,0x1f,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c11fff8f <unknown>

fmla    za.h[w11, 7], {z28.h - z31.h}, z15.h[7]  // 11000001-00011111-11111111-10001111
// CHECK-INST: fmla    za.h[w11, 7, vgx4], { z28.h - z31.h }, z15.h[7]
// CHECK-ENCODING: [0x8f,0xff,0x1f,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c11fff8f <unknown>

fmla    za.h[w8, 5, vgx4], {z16.h - z19.h}, z0.h[6]  // 11000001-00010000-10011110-00000101
// CHECK-INST: fmla    za.h[w8, 5, vgx4], { z16.h - z19.h }, z0.h[6]
// CHECK-ENCODING: [0x05,0x9e,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1109e05 <unknown>

fmla    za.h[w8, 5], {z16.h - z19.h}, z0.h[6]  // 11000001-00010000-10011110-00000101
// CHECK-INST: fmla    za.h[w8, 5, vgx4], { z16.h - z19.h }, z0.h[6]
// CHECK-ENCODING: [0x05,0x9e,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1109e05 <unknown>

fmla    za.h[w8, 1, vgx4], {z0.h - z3.h}, z14.h[2]  // 11000001-00011110-10010100-00000001
// CHECK-INST: fmla    za.h[w8, 1, vgx4], { z0.h - z3.h }, z14.h[2]
// CHECK-ENCODING: [0x01,0x94,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c11e9401 <unknown>

fmla    za.h[w8, 1], {z0.h - z3.h}, z14.h[2]  // 11000001-00011110-10010100-00000001
// CHECK-INST: fmla    za.h[w8, 1, vgx4], { z0.h - z3.h }, z14.h[2]
// CHECK-ENCODING: [0x01,0x94,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c11e9401 <unknown>

fmla    za.h[w10, 0, vgx4], {z16.h - z19.h}, z4.h[3]  // 11000001-00010100-11010110-00001000
// CHECK-INST: fmla    za.h[w10, 0, vgx4], { z16.h - z19.h }, z4.h[3]
// CHECK-ENCODING: [0x08,0xd6,0x14,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c114d608 <unknown>

fmla    za.h[w10, 0], {z16.h - z19.h}, z4.h[3]  // 11000001-00010100-11010110-00001000
// CHECK-INST: fmla    za.h[w10, 0, vgx4], { z16.h - z19.h }, z4.h[3]
// CHECK-ENCODING: [0x08,0xd6,0x14,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c114d608 <unknown>

fmla    za.h[w8, 0, vgx4], {z12.h - z15.h}, z2.h[4]  // 11000001-00010010-10011001-10000000
// CHECK-INST: fmla    za.h[w8, 0, vgx4], { z12.h - z15.h }, z2.h[4]
// CHECK-ENCODING: [0x80,0x99,0x12,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1129980 <unknown>

fmla    za.h[w8, 0], {z12.h - z15.h}, z2.h[4]  // 11000001-00010010-10011001-10000000
// CHECK-INST: fmla    za.h[w8, 0, vgx4], { z12.h - z15.h }, z2.h[4]
// CHECK-ENCODING: [0x80,0x99,0x12,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1129980 <unknown>

fmla    za.h[w10, 1, vgx4], {z0.h - z3.h}, z10.h[4]  // 11000001-00011010-11011000-00000001
// CHECK-INST: fmla    za.h[w10, 1, vgx4], { z0.h - z3.h }, z10.h[4]
// CHECK-ENCODING: [0x01,0xd8,0x1a,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c11ad801 <unknown>

fmla    za.h[w10, 1], {z0.h - z3.h}, z10.h[4]  // 11000001-00011010-11011000-00000001
// CHECK-INST: fmla    za.h[w10, 1, vgx4], { z0.h - z3.h }, z10.h[4]
// CHECK-ENCODING: [0x01,0xd8,0x1a,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c11ad801 <unknown>

fmla    za.h[w8, 5, vgx4], {z20.h - z23.h}, z14.h[5]  // 11000001-00011110-10011010-10001101
// CHECK-INST: fmla    za.h[w8, 5, vgx4], { z20.h - z23.h }, z14.h[5]
// CHECK-ENCODING: [0x8d,0x9a,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c11e9a8d <unknown>

fmla    za.h[w8, 5], {z20.h - z23.h}, z14.h[5]  // 11000001-00011110-10011010-10001101
// CHECK-INST: fmla    za.h[w8, 5, vgx4], { z20.h - z23.h }, z14.h[5]
// CHECK-ENCODING: [0x8d,0x9a,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c11e9a8d <unknown>

fmla    za.h[w11, 2, vgx4], {z8.h - z11.h}, z1.h[2]  // 11000001-00010001-11110101-00000010
// CHECK-INST: fmla    za.h[w11, 2, vgx4], { z8.h - z11.h }, z1.h[2]
// CHECK-ENCODING: [0x02,0xf5,0x11,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c111f502 <unknown>

fmla    za.h[w11, 2], {z8.h - z11.h}, z1.h[2]  // 11000001-00010001-11110101-00000010
// CHECK-INST: fmla    za.h[w11, 2, vgx4], { z8.h - z11.h }, z1.h[2]
// CHECK-ENCODING: [0x02,0xf5,0x11,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c111f502 <unknown>

fmla    za.h[w9, 7, vgx4], {z12.h - z15.h}, z11.h[4]  // 11000001-00011011-10111001-10000111
// CHECK-INST: fmla    za.h[w9, 7, vgx4], { z12.h - z15.h }, z11.h[4]
// CHECK-ENCODING: [0x87,0xb9,0x1b,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c11bb987 <unknown>

fmla    za.h[w9, 7], {z12.h - z15.h}, z11.h[4]  // 11000001-00011011-10111001-10000111
// CHECK-INST: fmla    za.h[w9, 7, vgx4], { z12.h - z15.h }, z11.h[4]
// CHECK-ENCODING: [0x87,0xb9,0x1b,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c11bb987 <unknown>

fmla    za.h[w8, 0, vgx4], {z0.h - z3.h}, {z0.h - z3.h}  // 11000001-10100001-00010000-00001000
// CHECK-INST: fmla    za.h[w8, 0, vgx4], { z0.h - z3.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x08,0x10,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1a11008 <unknown>

fmla    za.h[w8, 0], {z0.h - z3.h}, {z0.h - z3.h}  // 11000001-10100001-00010000-00001000
// CHECK-INST: fmla    za.h[w8, 0, vgx4], { z0.h - z3.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x08,0x10,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1a11008 <unknown>

fmla    za.h[w10, 5, vgx4], {z8.h - z11.h}, {z20.h - z23.h}  // 11000001-10110101-01010001-00001101
// CHECK-INST: fmla    za.h[w10, 5, vgx4], { z8.h - z11.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x0d,0x51,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1b5510d <unknown>

fmla    za.h[w10, 5], {z8.h - z11.h}, {z20.h - z23.h}  // 11000001-10110101-01010001-00001101
// CHECK-INST: fmla    za.h[w10, 5, vgx4], { z8.h - z11.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x0d,0x51,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1b5510d <unknown>

fmla    za.h[w11, 7, vgx4], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-10101001-01110001-10001111
// CHECK-INST: fmla    za.h[w11, 7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x8f,0x71,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1a9718f <unknown>

fmla    za.h[w11, 7], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-10101001-01110001-10001111
// CHECK-INST: fmla    za.h[w11, 7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x8f,0x71,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1a9718f <unknown>

fmla    za.h[w11, 7, vgx4], {z28.h - z31.h}, {z28.h - z31.h}  // 11000001-10111101-01110011-10001111
// CHECK-INST: fmla    za.h[w11, 7, vgx4], { z28.h - z31.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x8f,0x73,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1bd738f <unknown>

fmla    za.h[w11, 7], {z28.h - z31.h}, {z28.h - z31.h}  // 11000001-10111101-01110011-10001111
// CHECK-INST: fmla    za.h[w11, 7, vgx4], { z28.h - z31.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x8f,0x73,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1bd738f <unknown>

fmla    za.h[w8, 5, vgx4], {z16.h - z19.h}, {z16.h - z19.h}  // 11000001-10110001-00010010-00001101
// CHECK-INST: fmla    za.h[w8, 5, vgx4], { z16.h - z19.h }, { z16.h - z19.h }
// CHECK-ENCODING: [0x0d,0x12,0xb1,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1b1120d <unknown>

fmla    za.h[w8, 5], {z16.h - z19.h}, {z16.h - z19.h}  // 11000001-10110001-00010010-00001101
// CHECK-INST: fmla    za.h[w8, 5, vgx4], { z16.h - z19.h }, { z16.h - z19.h }
// CHECK-ENCODING: [0x0d,0x12,0xb1,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1b1120d <unknown>

fmla    za.h[w8, 1, vgx4], {z0.h - z3.h}, {z28.h - z31.h}  // 11000001-10111101-00010000-00001001
// CHECK-INST: fmla    za.h[w8, 1, vgx4], { z0.h - z3.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x09,0x10,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1bd1009 <unknown>

fmla    za.h[w8, 1], {z0.h - z3.h}, {z28.h - z31.h}  // 11000001-10111101-00010000-00001001
// CHECK-INST: fmla    za.h[w8, 1, vgx4], { z0.h - z3.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x09,0x10,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1bd1009 <unknown>

fmla    za.h[w10, 0, vgx4], {z16.h - z19.h}, {z20.h - z23.h}  // 11000001-10110101-01010010-00001000
// CHECK-INST: fmla    za.h[w10, 0, vgx4], { z16.h - z19.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x08,0x52,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1b55208 <unknown>

fmla    za.h[w10, 0], {z16.h - z19.h}, {z20.h - z23.h}  // 11000001-10110101-01010010-00001000
// CHECK-INST: fmla    za.h[w10, 0, vgx4], { z16.h - z19.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x08,0x52,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1b55208 <unknown>

fmla    za.h[w8, 0, vgx4], {z12.h - z15.h}, {z0.h - z3.h}  // 11000001-10100001-00010001-10001000
// CHECK-INST: fmla    za.h[w8, 0, vgx4], { z12.h - z15.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x88,0x11,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1a11188 <unknown>

fmla    za.h[w8, 0], {z12.h - z15.h}, {z0.h - z3.h}  // 11000001-10100001-00010001-10001000
// CHECK-INST: fmla    za.h[w8, 0, vgx4], { z12.h - z15.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x88,0x11,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1a11188 <unknown>

fmla    za.h[w10, 1, vgx4], {z0.h - z3.h}, {z24.h - z27.h}  // 11000001-10111001-01010000-00001001
// CHECK-INST: fmla    za.h[w10, 1, vgx4], { z0.h - z3.h }, { z24.h - z27.h }
// CHECK-ENCODING: [0x09,0x50,0xb9,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1b95009 <unknown>

fmla    za.h[w10, 1], {z0.h - z3.h}, {z24.h - z27.h}  // 11000001-10111001-01010000-00001001
// CHECK-INST: fmla    za.h[w10, 1, vgx4], { z0.h - z3.h }, { z24.h - z27.h }
// CHECK-ENCODING: [0x09,0x50,0xb9,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1b95009 <unknown>

fmla    za.h[w8, 5, vgx4], {z20.h - z23.h}, {z28.h - z31.h}  // 11000001-10111101-00010010-10001101
// CHECK-INST: fmla    za.h[w8, 5, vgx4], { z20.h - z23.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x8d,0x12,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1bd128d <unknown>

fmla    za.h[w8, 5], {z20.h - z23.h}, {z28.h - z31.h}  // 11000001-10111101-00010010-10001101
// CHECK-INST: fmla    za.h[w8, 5, vgx4], { z20.h - z23.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x8d,0x12,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1bd128d <unknown>

fmla    za.h[w11, 2, vgx4], {z8.h - z11.h}, {z0.h - z3.h}  // 11000001-10100001-01110001-00001010
// CHECK-INST: fmla    za.h[w11, 2, vgx4], { z8.h - z11.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x0a,0x71,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1a1710a <unknown>

fmla    za.h[w11, 2], {z8.h - z11.h}, {z0.h - z3.h}  // 11000001-10100001-01110001-00001010
// CHECK-INST: fmla    za.h[w11, 2, vgx4], { z8.h - z11.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x0a,0x71,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1a1710a <unknown>

fmla    za.h[w9, 7, vgx4], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-10101001-00110001-10001111
// CHECK-INST: fmla    za.h[w9, 7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x8f,0x31,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1a9318f <unknown>

fmla    za.h[w9, 7], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-10101001-00110001-10001111
// CHECK-INST: fmla    za.h[w9, 7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x8f,0x31,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: c1a9318f <unknown>
