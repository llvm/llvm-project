// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+sme-f64f64 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2,+sme-f64f64 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2,+sme-f64f64 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2,+sme-f64f64 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+sme-f64f64 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2,+sme-f64f64 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


fmls    za.d[w8, 0, vgx2], {z0.d, z1.d}, z0.d  // 11000001-01100000-00011000-00001000
// CHECK-INST: fmls    za.d[w8, 0, vgx2], { z0.d, z1.d }, z0.d
// CHECK-ENCODING: [0x08,0x18,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1601808 <unknown>

fmls    za.d[w8, 0], {z0.d - z1.d}, z0.d  // 11000001-01100000-00011000-00001000
// CHECK-INST: fmls    za.d[w8, 0, vgx2], { z0.d, z1.d }, z0.d
// CHECK-ENCODING: [0x08,0x18,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1601808 <unknown>

fmls    za.d[w10, 5, vgx2], {z10.d, z11.d}, z5.d  // 11000001-01100101-01011001-01001101
// CHECK-INST: fmls    za.d[w10, 5, vgx2], { z10.d, z11.d }, z5.d
// CHECK-ENCODING: [0x4d,0x59,0x65,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c165594d <unknown>

fmls    za.d[w10, 5], {z10.d - z11.d}, z5.d  // 11000001-01100101-01011001-01001101
// CHECK-INST: fmls    za.d[w10, 5, vgx2], { z10.d, z11.d }, z5.d
// CHECK-ENCODING: [0x4d,0x59,0x65,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c165594d <unknown>

fmls    za.d[w11, 7, vgx2], {z13.d, z14.d}, z8.d  // 11000001-01101000-01111001-10101111
// CHECK-INST: fmls    za.d[w11, 7, vgx2], { z13.d, z14.d }, z8.d
// CHECK-ENCODING: [0xaf,0x79,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16879af <unknown>

fmls    za.d[w11, 7], {z13.d - z14.d}, z8.d  // 11000001-01101000-01111001-10101111
// CHECK-INST: fmls    za.d[w11, 7, vgx2], { z13.d, z14.d }, z8.d
// CHECK-ENCODING: [0xaf,0x79,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16879af <unknown>

fmls    za.d[w11, 7, vgx2], {z31.d, z0.d}, z15.d  // 11000001-01101111-01111011-11101111
// CHECK-INST: fmls    za.d[w11, 7, vgx2], { z31.d, z0.d }, z15.d
// CHECK-ENCODING: [0xef,0x7b,0x6f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16f7bef <unknown>

fmls    za.d[w11, 7], {z31.d - z0.d}, z15.d  // 11000001-01101111-01111011-11101111
// CHECK-INST: fmls    za.d[w11, 7, vgx2], { z31.d, z0.d }, z15.d
// CHECK-ENCODING: [0xef,0x7b,0x6f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16f7bef <unknown>

fmls    za.d[w8, 5, vgx2], {z17.d, z18.d}, z0.d  // 11000001-01100000-00011010-00101101
// CHECK-INST: fmls    za.d[w8, 5, vgx2], { z17.d, z18.d }, z0.d
// CHECK-ENCODING: [0x2d,0x1a,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1601a2d <unknown>

fmls    za.d[w8, 5], {z17.d - z18.d}, z0.d  // 11000001-01100000-00011010-00101101
// CHECK-INST: fmls    za.d[w8, 5, vgx2], { z17.d, z18.d }, z0.d
// CHECK-ENCODING: [0x2d,0x1a,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1601a2d <unknown>

fmls    za.d[w8, 1, vgx2], {z1.d, z2.d}, z14.d  // 11000001-01101110-00011000-00101001
// CHECK-INST: fmls    za.d[w8, 1, vgx2], { z1.d, z2.d }, z14.d
// CHECK-ENCODING: [0x29,0x18,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e1829 <unknown>

fmls    za.d[w8, 1], {z1.d - z2.d}, z14.d  // 11000001-01101110-00011000-00101001
// CHECK-INST: fmls    za.d[w8, 1, vgx2], { z1.d, z2.d }, z14.d
// CHECK-ENCODING: [0x29,0x18,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e1829 <unknown>

fmls    za.d[w10, 0, vgx2], {z19.d, z20.d}, z4.d  // 11000001-01100100-01011010-01101000
// CHECK-INST: fmls    za.d[w10, 0, vgx2], { z19.d, z20.d }, z4.d
// CHECK-ENCODING: [0x68,0x5a,0x64,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1645a68 <unknown>

fmls    za.d[w10, 0], {z19.d - z20.d}, z4.d  // 11000001-01100100-01011010-01101000
// CHECK-INST: fmls    za.d[w10, 0, vgx2], { z19.d, z20.d }, z4.d
// CHECK-ENCODING: [0x68,0x5a,0x64,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1645a68 <unknown>

fmls    za.d[w8, 0, vgx2], {z12.d, z13.d}, z2.d  // 11000001-01100010-00011001-10001000
// CHECK-INST: fmls    za.d[w8, 0, vgx2], { z12.d, z13.d }, z2.d
// CHECK-ENCODING: [0x88,0x19,0x62,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1621988 <unknown>

fmls    za.d[w8, 0], {z12.d - z13.d}, z2.d  // 11000001-01100010-00011001-10001000
// CHECK-INST: fmls    za.d[w8, 0, vgx2], { z12.d, z13.d }, z2.d
// CHECK-ENCODING: [0x88,0x19,0x62,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1621988 <unknown>

fmls    za.d[w10, 1, vgx2], {z1.d, z2.d}, z10.d  // 11000001-01101010-01011000-00101001
// CHECK-INST: fmls    za.d[w10, 1, vgx2], { z1.d, z2.d }, z10.d
// CHECK-ENCODING: [0x29,0x58,0x6a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16a5829 <unknown>

fmls    za.d[w10, 1], {z1.d - z2.d}, z10.d  // 11000001-01101010-01011000-00101001
// CHECK-INST: fmls    za.d[w10, 1, vgx2], { z1.d, z2.d }, z10.d
// CHECK-ENCODING: [0x29,0x58,0x6a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16a5829 <unknown>

fmls    za.d[w8, 5, vgx2], {z22.d, z23.d}, z14.d  // 11000001-01101110-00011010-11001101
// CHECK-INST: fmls    za.d[w8, 5, vgx2], { z22.d, z23.d }, z14.d
// CHECK-ENCODING: [0xcd,0x1a,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e1acd <unknown>

fmls    za.d[w8, 5], {z22.d - z23.d}, z14.d  // 11000001-01101110-00011010-11001101
// CHECK-INST: fmls    za.d[w8, 5, vgx2], { z22.d, z23.d }, z14.d
// CHECK-ENCODING: [0xcd,0x1a,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e1acd <unknown>

fmls    za.d[w11, 2, vgx2], {z9.d, z10.d}, z1.d  // 11000001-01100001-01111001-00101010
// CHECK-INST: fmls    za.d[w11, 2, vgx2], { z9.d, z10.d }, z1.d
// CHECK-ENCODING: [0x2a,0x79,0x61,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c161792a <unknown>

fmls    za.d[w11, 2], {z9.d - z10.d}, z1.d  // 11000001-01100001-01111001-00101010
// CHECK-INST: fmls    za.d[w11, 2, vgx2], { z9.d, z10.d }, z1.d
// CHECK-ENCODING: [0x2a,0x79,0x61,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c161792a <unknown>

fmls    za.d[w9, 7, vgx2], {z12.d, z13.d}, z11.d  // 11000001-01101011-00111001-10001111
// CHECK-INST: fmls    za.d[w9, 7, vgx2], { z12.d, z13.d }, z11.d
// CHECK-ENCODING: [0x8f,0x39,0x6b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16b398f <unknown>

fmls    za.d[w9, 7], {z12.d - z13.d}, z11.d  // 11000001-01101011-00111001-10001111
// CHECK-INST: fmls    za.d[w9, 7, vgx2], { z12.d, z13.d }, z11.d
// CHECK-ENCODING: [0x8f,0x39,0x6b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16b398f <unknown>


fmls    za.d[w8, 0, vgx2], {z0.d, z1.d}, z0.d[0]  // 11000001-11010000-00000000-00010000
// CHECK-INST: fmls    za.d[w8, 0, vgx2], { z0.d, z1.d }, z0.d[0]
// CHECK-ENCODING: [0x10,0x00,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d00010 <unknown>

fmls    za.d[w8, 0], {z0.d, z1.d}, z0.d[0]  // 11000001-11010000-00000000-00010000
// CHECK-INST: fmls    za.d[w8, 0, vgx2], { z0.d, z1.d }, z0.d[0]
// CHECK-ENCODING: [0x10,0x00,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d00010 <unknown>

fmls    za.d[w10, 5, vgx2], {z10.d, z11.d}, z5.d[1]  // 11000001-11010101-01000101-01010101
// CHECK-INST: fmls    za.d[w10, 5, vgx2], { z10.d, z11.d }, z5.d[1]
// CHECK-ENCODING: [0x55,0x45,0xd5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d54555 <unknown>

fmls    za.d[w10, 5], {z10.d, z11.d}, z5.d[1]  // 11000001-11010101-01000101-01010101
// CHECK-INST: fmls    za.d[w10, 5, vgx2], { z10.d, z11.d }, z5.d[1]
// CHECK-ENCODING: [0x55,0x45,0xd5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d54555 <unknown>

fmls    za.d[w11, 7, vgx2], {z12.d, z13.d}, z8.d[1]  // 11000001-11011000-01100101-10010111
// CHECK-INST: fmls    za.d[w11, 7, vgx2], { z12.d, z13.d }, z8.d[1]
// CHECK-ENCODING: [0x97,0x65,0xd8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d86597 <unknown>

fmls    za.d[w11, 7], {z12.d, z13.d}, z8.d[1]  // 11000001-11011000-01100101-10010111
// CHECK-INST: fmls    za.d[w11, 7, vgx2], { z12.d, z13.d }, z8.d[1]
// CHECK-ENCODING: [0x97,0x65,0xd8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d86597 <unknown>

fmls    za.d[w11, 7, vgx2], {z30.d, z31.d}, z15.d[1]  // 11000001-11011111-01100111-11010111
// CHECK-INST: fmls    za.d[w11, 7, vgx2], { z30.d, z31.d }, z15.d[1]
// CHECK-ENCODING: [0xd7,0x67,0xdf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1df67d7 <unknown>

fmls    za.d[w11, 7], {z30.d, z31.d}, z15.d[1]  // 11000001-11011111-01100111-11010111
// CHECK-INST: fmls    za.d[w11, 7, vgx2], { z30.d, z31.d }, z15.d[1]
// CHECK-ENCODING: [0xd7,0x67,0xdf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1df67d7 <unknown>

fmls    za.d[w8, 5, vgx2], {z16.d, z17.d}, z0.d[1]  // 11000001-11010000-00000110-00010101
// CHECK-INST: fmls    za.d[w8, 5, vgx2], { z16.d, z17.d }, z0.d[1]
// CHECK-ENCODING: [0x15,0x06,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d00615 <unknown>

fmls    za.d[w8, 5], {z16.d, z17.d}, z0.d[1]  // 11000001-11010000-00000110-00010101
// CHECK-INST: fmls    za.d[w8, 5, vgx2], { z16.d, z17.d }, z0.d[1]
// CHECK-ENCODING: [0x15,0x06,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d00615 <unknown>

fmls    za.d[w8, 1, vgx2], {z0.d, z1.d}, z14.d[1]  // 11000001-11011110-00000100-00010001
// CHECK-INST: fmls    za.d[w8, 1, vgx2], { z0.d, z1.d }, z14.d[1]
// CHECK-ENCODING: [0x11,0x04,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de0411 <unknown>

fmls    za.d[w8, 1], {z0.d, z1.d}, z14.d[1]  // 11000001-11011110-00000100-00010001
// CHECK-INST: fmls    za.d[w8, 1, vgx2], { z0.d, z1.d }, z14.d[1]
// CHECK-ENCODING: [0x11,0x04,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de0411 <unknown>

fmls    za.d[w10, 0, vgx2], {z18.d, z19.d}, z4.d[1]  // 11000001-11010100-01000110-01010000
// CHECK-INST: fmls    za.d[w10, 0, vgx2], { z18.d, z19.d }, z4.d[1]
// CHECK-ENCODING: [0x50,0x46,0xd4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d44650 <unknown>

fmls    za.d[w10, 0], {z18.d, z19.d}, z4.d[1]  // 11000001-11010100-01000110-01010000
// CHECK-INST: fmls    za.d[w10, 0, vgx2], { z18.d, z19.d }, z4.d[1]
// CHECK-ENCODING: [0x50,0x46,0xd4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d44650 <unknown>

fmls    za.d[w8, 0, vgx2], {z12.d, z13.d}, z2.d[0]  // 11000001-11010010-00000001-10010000
// CHECK-INST: fmls    za.d[w8, 0, vgx2], { z12.d, z13.d }, z2.d[0]
// CHECK-ENCODING: [0x90,0x01,0xd2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d20190 <unknown>

fmls    za.d[w8, 0], {z12.d, z13.d}, z2.d[0]  // 11000001-11010010-00000001-10010000
// CHECK-INST: fmls    za.d[w8, 0, vgx2], { z12.d, z13.d }, z2.d[0]
// CHECK-ENCODING: [0x90,0x01,0xd2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d20190 <unknown>

fmls    za.d[w10, 1, vgx2], {z0.d, z1.d}, z10.d[0]  // 11000001-11011010-01000000-00010001
// CHECK-INST: fmls    za.d[w10, 1, vgx2], { z0.d, z1.d }, z10.d[0]
// CHECK-ENCODING: [0x11,0x40,0xda,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1da4011 <unknown>

fmls    za.d[w10, 1], {z0.d, z1.d}, z10.d[0]  // 11000001-11011010-01000000-00010001
// CHECK-INST: fmls    za.d[w10, 1, vgx2], { z0.d, z1.d }, z10.d[0]
// CHECK-ENCODING: [0x11,0x40,0xda,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1da4011 <unknown>

fmls    za.d[w8, 5, vgx2], {z22.d, z23.d}, z14.d[0]  // 11000001-11011110-00000010-11010101
// CHECK-INST: fmls    za.d[w8, 5, vgx2], { z22.d, z23.d }, z14.d[0]
// CHECK-ENCODING: [0xd5,0x02,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de02d5 <unknown>

fmls    za.d[w8, 5], {z22.d, z23.d}, z14.d[0]  // 11000001-11011110-00000010-11010101
// CHECK-INST: fmls    za.d[w8, 5, vgx2], { z22.d, z23.d }, z14.d[0]
// CHECK-ENCODING: [0xd5,0x02,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de02d5 <unknown>

fmls    za.d[w11, 2, vgx2], {z8.d, z9.d}, z1.d[1]  // 11000001-11010001-01100101-00010010
// CHECK-INST: fmls    za.d[w11, 2, vgx2], { z8.d, z9.d }, z1.d[1]
// CHECK-ENCODING: [0x12,0x65,0xd1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d16512 <unknown>

fmls    za.d[w11, 2], {z8.d, z9.d}, z1.d[1]  // 11000001-11010001-01100101-00010010
// CHECK-INST: fmls    za.d[w11, 2, vgx2], { z8.d, z9.d }, z1.d[1]
// CHECK-ENCODING: [0x12,0x65,0xd1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d16512 <unknown>

fmls    za.d[w9, 7, vgx2], {z12.d, z13.d}, z11.d[0]  // 11000001-11011011-00100001-10010111
// CHECK-INST: fmls    za.d[w9, 7, vgx2], { z12.d, z13.d }, z11.d[0]
// CHECK-ENCODING: [0x97,0x21,0xdb,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1db2197 <unknown>

fmls    za.d[w9, 7], {z12.d, z13.d}, z11.d[0]  // 11000001-11011011-00100001-10010111
// CHECK-INST: fmls    za.d[w9, 7, vgx2], { z12.d, z13.d }, z11.d[0]
// CHECK-ENCODING: [0x97,0x21,0xdb,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1db2197 <unknown>


fmls    za.d[w8, 0, vgx2], {z0.d, z1.d}, {z0.d, z1.d}  // 11000001-11100000-00011000-00001000
// CHECK-INST: fmls    za.d[w8, 0, vgx2], { z0.d, z1.d }, { z0.d, z1.d }
// CHECK-ENCODING: [0x08,0x18,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e01808 <unknown>

fmls    za.d[w8, 0], {z0.d - z1.d}, {z0.d - z1.d}  // 11000001-11100000-00011000-00001000
// CHECK-INST: fmls    za.d[w8, 0, vgx2], { z0.d, z1.d }, { z0.d, z1.d }
// CHECK-ENCODING: [0x08,0x18,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e01808 <unknown>

fmls    za.d[w10, 5, vgx2], {z10.d, z11.d}, {z20.d, z21.d}  // 11000001-11110100-01011001-01001101
// CHECK-INST: fmls    za.d[w10, 5, vgx2], { z10.d, z11.d }, { z20.d, z21.d }
// CHECK-ENCODING: [0x4d,0x59,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f4594d <unknown>

fmls    za.d[w10, 5], {z10.d - z11.d}, {z20.d - z21.d}  // 11000001-11110100-01011001-01001101
// CHECK-INST: fmls    za.d[w10, 5, vgx2], { z10.d, z11.d }, { z20.d, z21.d }
// CHECK-ENCODING: [0x4d,0x59,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f4594d <unknown>

fmls    za.d[w11, 7, vgx2], {z12.d, z13.d}, {z8.d, z9.d}  // 11000001-11101000-01111001-10001111
// CHECK-INST: fmls    za.d[w11, 7, vgx2], { z12.d, z13.d }, { z8.d, z9.d }
// CHECK-ENCODING: [0x8f,0x79,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e8798f <unknown>

fmls    za.d[w11, 7], {z12.d - z13.d}, {z8.d - z9.d}  // 11000001-11101000-01111001-10001111
// CHECK-INST: fmls    za.d[w11, 7, vgx2], { z12.d, z13.d }, { z8.d, z9.d }
// CHECK-ENCODING: [0x8f,0x79,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e8798f <unknown>

fmls    za.d[w11, 7, vgx2], {z30.d, z31.d}, {z30.d, z31.d}  // 11000001-11111110-01111011-11001111
// CHECK-INST: fmls    za.d[w11, 7, vgx2], { z30.d, z31.d }, { z30.d, z31.d }
// CHECK-ENCODING: [0xcf,0x7b,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe7bcf <unknown>

fmls    za.d[w11, 7], {z30.d - z31.d}, {z30.d - z31.d}  // 11000001-11111110-01111011-11001111
// CHECK-INST: fmls    za.d[w11, 7, vgx2], { z30.d, z31.d }, { z30.d, z31.d }
// CHECK-ENCODING: [0xcf,0x7b,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe7bcf <unknown>

fmls    za.d[w8, 5, vgx2], {z16.d, z17.d}, {z16.d, z17.d}  // 11000001-11110000-00011010-00001101
// CHECK-INST: fmls    za.d[w8, 5, vgx2], { z16.d, z17.d }, { z16.d, z17.d }
// CHECK-ENCODING: [0x0d,0x1a,0xf0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f01a0d <unknown>

fmls    za.d[w8, 5], {z16.d - z17.d}, {z16.d - z17.d}  // 11000001-11110000-00011010-00001101
// CHECK-INST: fmls    za.d[w8, 5, vgx2], { z16.d, z17.d }, { z16.d, z17.d }
// CHECK-ENCODING: [0x0d,0x1a,0xf0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f01a0d <unknown>

fmls    za.d[w8, 1, vgx2], {z0.d, z1.d}, {z30.d, z31.d}  // 11000001-11111110-00011000-00001001
// CHECK-INST: fmls    za.d[w8, 1, vgx2], { z0.d, z1.d }, { z30.d, z31.d }
// CHECK-ENCODING: [0x09,0x18,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe1809 <unknown>

fmls    za.d[w8, 1], {z0.d - z1.d}, {z30.d - z31.d}  // 11000001-11111110-00011000-00001001
// CHECK-INST: fmls    za.d[w8, 1, vgx2], { z0.d, z1.d }, { z30.d, z31.d }
// CHECK-ENCODING: [0x09,0x18,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe1809 <unknown>

fmls    za.d[w10, 0, vgx2], {z18.d, z19.d}, {z20.d, z21.d}  // 11000001-11110100-01011010-01001000
// CHECK-INST: fmls    za.d[w10, 0, vgx2], { z18.d, z19.d }, { z20.d, z21.d }
// CHECK-ENCODING: [0x48,0x5a,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f45a48 <unknown>

fmls    za.d[w10, 0], {z18.d - z19.d}, {z20.d - z21.d}  // 11000001-11110100-01011010-01001000
// CHECK-INST: fmls    za.d[w10, 0, vgx2], { z18.d, z19.d }, { z20.d, z21.d }
// CHECK-ENCODING: [0x48,0x5a,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f45a48 <unknown>

fmls    za.d[w8, 0, vgx2], {z12.d, z13.d}, {z2.d, z3.d}  // 11000001-11100010-00011001-10001000
// CHECK-INST: fmls    za.d[w8, 0, vgx2], { z12.d, z13.d }, { z2.d, z3.d }
// CHECK-ENCODING: [0x88,0x19,0xe2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e21988 <unknown>

fmls    za.d[w8, 0], {z12.d - z13.d}, {z2.d - z3.d}  // 11000001-11100010-00011001-10001000
// CHECK-INST: fmls    za.d[w8, 0, vgx2], { z12.d, z13.d }, { z2.d, z3.d }
// CHECK-ENCODING: [0x88,0x19,0xe2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e21988 <unknown>

fmls    za.d[w10, 1, vgx2], {z0.d, z1.d}, {z26.d, z27.d}  // 11000001-11111010-01011000-00001001
// CHECK-INST: fmls    za.d[w10, 1, vgx2], { z0.d, z1.d }, { z26.d, z27.d }
// CHECK-ENCODING: [0x09,0x58,0xfa,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fa5809 <unknown>

fmls    za.d[w10, 1], {z0.d - z1.d}, {z26.d - z27.d}  // 11000001-11111010-01011000-00001001
// CHECK-INST: fmls    za.d[w10, 1, vgx2], { z0.d, z1.d }, { z26.d, z27.d }
// CHECK-ENCODING: [0x09,0x58,0xfa,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fa5809 <unknown>

fmls    za.d[w8, 5, vgx2], {z22.d, z23.d}, {z30.d, z31.d}  // 11000001-11111110-00011010-11001101
// CHECK-INST: fmls    za.d[w8, 5, vgx2], { z22.d, z23.d }, { z30.d, z31.d }
// CHECK-ENCODING: [0xcd,0x1a,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe1acd <unknown>

fmls    za.d[w8, 5], {z22.d - z23.d}, {z30.d - z31.d}  // 11000001-11111110-00011010-11001101
// CHECK-INST: fmls    za.d[w8, 5, vgx2], { z22.d, z23.d }, { z30.d, z31.d }
// CHECK-ENCODING: [0xcd,0x1a,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe1acd <unknown>

fmls    za.d[w11, 2, vgx2], {z8.d, z9.d}, {z0.d, z1.d}  // 11000001-11100000-01111001-00001010
// CHECK-INST: fmls    za.d[w11, 2, vgx2], { z8.d, z9.d }, { z0.d, z1.d }
// CHECK-ENCODING: [0x0a,0x79,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e0790a <unknown>

fmls    za.d[w11, 2], {z8.d - z9.d}, {z0.d - z1.d}  // 11000001-11100000-01111001-00001010
// CHECK-INST: fmls    za.d[w11, 2, vgx2], { z8.d, z9.d }, { z0.d, z1.d }
// CHECK-ENCODING: [0x0a,0x79,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e0790a <unknown>

fmls    za.d[w9, 7, vgx2], {z12.d, z13.d}, {z10.d, z11.d}  // 11000001-11101010-00111001-10001111
// CHECK-INST: fmls    za.d[w9, 7, vgx2], { z12.d, z13.d }, { z10.d, z11.d }
// CHECK-ENCODING: [0x8f,0x39,0xea,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ea398f <unknown>

fmls    za.d[w9, 7], {z12.d - z13.d}, {z10.d - z11.d}  // 11000001-11101010-00111001-10001111
// CHECK-INST: fmls    za.d[w9, 7, vgx2], { z12.d, z13.d }, { z10.d, z11.d }
// CHECK-ENCODING: [0x8f,0x39,0xea,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ea398f <unknown>


fmls    za.s[w8, 0, vgx2], {z0.s, z1.s}, z0.s  // 11000001-00100000-00011000-00001000
// CHECK-INST: fmls    za.s[w8, 0, vgx2], { z0.s, z1.s }, z0.s
// CHECK-ENCODING: [0x08,0x18,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1201808 <unknown>

fmls    za.s[w8, 0], {z0.s - z1.s}, z0.s  // 11000001-00100000-00011000-00001000
// CHECK-INST: fmls    za.s[w8, 0, vgx2], { z0.s, z1.s }, z0.s
// CHECK-ENCODING: [0x08,0x18,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1201808 <unknown>

fmls    za.s[w10, 5, vgx2], {z10.s, z11.s}, z5.s  // 11000001-00100101-01011001-01001101
// CHECK-INST: fmls    za.s[w10, 5, vgx2], { z10.s, z11.s }, z5.s
// CHECK-ENCODING: [0x4d,0x59,0x25,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c125594d <unknown>

fmls    za.s[w10, 5], {z10.s - z11.s}, z5.s  // 11000001-00100101-01011001-01001101
// CHECK-INST: fmls    za.s[w10, 5, vgx2], { z10.s, z11.s }, z5.s
// CHECK-ENCODING: [0x4d,0x59,0x25,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c125594d <unknown>

fmls    za.s[w11, 7, vgx2], {z13.s, z14.s}, z8.s  // 11000001-00101000-01111001-10101111
// CHECK-INST: fmls    za.s[w11, 7, vgx2], { z13.s, z14.s }, z8.s
// CHECK-ENCODING: [0xaf,0x79,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12879af <unknown>

fmls    za.s[w11, 7], {z13.s - z14.s}, z8.s  // 11000001-00101000-01111001-10101111
// CHECK-INST: fmls    za.s[w11, 7, vgx2], { z13.s, z14.s }, z8.s
// CHECK-ENCODING: [0xaf,0x79,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12879af <unknown>

fmls    za.s[w11, 7, vgx2], {z31.s, z0.s}, z15.s  // 11000001-00101111-01111011-11101111
// CHECK-INST: fmls    za.s[w11, 7, vgx2], { z31.s, z0.s }, z15.s
// CHECK-ENCODING: [0xef,0x7b,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12f7bef <unknown>

fmls    za.s[w11, 7], {z31.s - z0.s}, z15.s  // 11000001-00101111-01111011-11101111
// CHECK-INST: fmls    za.s[w11, 7, vgx2], { z31.s, z0.s }, z15.s
// CHECK-ENCODING: [0xef,0x7b,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12f7bef <unknown>

fmls    za.s[w8, 5, vgx2], {z17.s, z18.s}, z0.s  // 11000001-00100000-00011010-00101101
// CHECK-INST: fmls    za.s[w8, 5, vgx2], { z17.s, z18.s }, z0.s
// CHECK-ENCODING: [0x2d,0x1a,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1201a2d <unknown>

fmls    za.s[w8, 5], {z17.s - z18.s}, z0.s  // 11000001-00100000-00011010-00101101
// CHECK-INST: fmls    za.s[w8, 5, vgx2], { z17.s, z18.s }, z0.s
// CHECK-ENCODING: [0x2d,0x1a,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1201a2d <unknown>

fmls    za.s[w8, 1, vgx2], {z1.s, z2.s}, z14.s  // 11000001-00101110-00011000-00101001
// CHECK-INST: fmls    za.s[w8, 1, vgx2], { z1.s, z2.s }, z14.s
// CHECK-ENCODING: [0x29,0x18,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e1829 <unknown>

fmls    za.s[w8, 1], {z1.s - z2.s}, z14.s  // 11000001-00101110-00011000-00101001
// CHECK-INST: fmls    za.s[w8, 1, vgx2], { z1.s, z2.s }, z14.s
// CHECK-ENCODING: [0x29,0x18,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e1829 <unknown>

fmls    za.s[w10, 0, vgx2], {z19.s, z20.s}, z4.s  // 11000001-00100100-01011010-01101000
// CHECK-INST: fmls    za.s[w10, 0, vgx2], { z19.s, z20.s }, z4.s
// CHECK-ENCODING: [0x68,0x5a,0x24,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1245a68 <unknown>

fmls    za.s[w10, 0], {z19.s - z20.s}, z4.s  // 11000001-00100100-01011010-01101000
// CHECK-INST: fmls    za.s[w10, 0, vgx2], { z19.s, z20.s }, z4.s
// CHECK-ENCODING: [0x68,0x5a,0x24,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1245a68 <unknown>

fmls    za.s[w8, 0, vgx2], {z12.s, z13.s}, z2.s  // 11000001-00100010-00011001-10001000
// CHECK-INST: fmls    za.s[w8, 0, vgx2], { z12.s, z13.s }, z2.s
// CHECK-ENCODING: [0x88,0x19,0x22,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1221988 <unknown>

fmls    za.s[w8, 0], {z12.s - z13.s}, z2.s  // 11000001-00100010-00011001-10001000
// CHECK-INST: fmls    za.s[w8, 0, vgx2], { z12.s, z13.s }, z2.s
// CHECK-ENCODING: [0x88,0x19,0x22,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1221988 <unknown>

fmls    za.s[w10, 1, vgx2], {z1.s, z2.s}, z10.s  // 11000001-00101010-01011000-00101001
// CHECK-INST: fmls    za.s[w10, 1, vgx2], { z1.s, z2.s }, z10.s
// CHECK-ENCODING: [0x29,0x58,0x2a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12a5829 <unknown>

fmls    za.s[w10, 1], {z1.s - z2.s}, z10.s  // 11000001-00101010-01011000-00101001
// CHECK-INST: fmls    za.s[w10, 1, vgx2], { z1.s, z2.s }, z10.s
// CHECK-ENCODING: [0x29,0x58,0x2a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12a5829 <unknown>

fmls    za.s[w8, 5, vgx2], {z22.s, z23.s}, z14.s  // 11000001-00101110-00011010-11001101
// CHECK-INST: fmls    za.s[w8, 5, vgx2], { z22.s, z23.s }, z14.s
// CHECK-ENCODING: [0xcd,0x1a,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e1acd <unknown>

fmls    za.s[w8, 5], {z22.s - z23.s}, z14.s  // 11000001-00101110-00011010-11001101
// CHECK-INST: fmls    za.s[w8, 5, vgx2], { z22.s, z23.s }, z14.s
// CHECK-ENCODING: [0xcd,0x1a,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e1acd <unknown>

fmls    za.s[w11, 2, vgx2], {z9.s, z10.s}, z1.s  // 11000001-00100001-01111001-00101010
// CHECK-INST: fmls    za.s[w11, 2, vgx2], { z9.s, z10.s }, z1.s
// CHECK-ENCODING: [0x2a,0x79,0x21,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c121792a <unknown>

fmls    za.s[w11, 2], {z9.s - z10.s}, z1.s  // 11000001-00100001-01111001-00101010
// CHECK-INST: fmls    za.s[w11, 2, vgx2], { z9.s, z10.s }, z1.s
// CHECK-ENCODING: [0x2a,0x79,0x21,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c121792a <unknown>

fmls    za.s[w9, 7, vgx2], {z12.s, z13.s}, z11.s  // 11000001-00101011-00111001-10001111
// CHECK-INST: fmls    za.s[w9, 7, vgx2], { z12.s, z13.s }, z11.s
// CHECK-ENCODING: [0x8f,0x39,0x2b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12b398f <unknown>

fmls    za.s[w9, 7], {z12.s - z13.s}, z11.s  // 11000001-00101011-00111001-10001111
// CHECK-INST: fmls    za.s[w9, 7, vgx2], { z12.s, z13.s }, z11.s
// CHECK-ENCODING: [0x8f,0x39,0x2b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12b398f <unknown>


fmls    za.s[w8, 0, vgx2], {z0.s, z1.s}, z0.s[0]  // 11000001-01010000-00000000-00010000
// CHECK-INST: fmls    za.s[w8, 0, vgx2], { z0.s, z1.s }, z0.s[0]
// CHECK-ENCODING: [0x10,0x00,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1500010 <unknown>

fmls    za.s[w8, 0], {z0.s, z1.s}, z0.s[0]  // 11000001-01010000-00000000-00010000
// CHECK-INST: fmls    za.s[w8, 0, vgx2], { z0.s, z1.s }, z0.s[0]
// CHECK-ENCODING: [0x10,0x00,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1500010 <unknown>

fmls    za.s[w10, 5, vgx2], {z10.s, z11.s}, z5.s[1]  // 11000001-01010101-01000101-01010101
// CHECK-INST: fmls    za.s[w10, 5, vgx2], { z10.s, z11.s }, z5.s[1]
// CHECK-ENCODING: [0x55,0x45,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1554555 <unknown>

fmls    za.s[w10, 5], {z10.s, z11.s}, z5.s[1]  // 11000001-01010101-01000101-01010101
// CHECK-INST: fmls    za.s[w10, 5, vgx2], { z10.s, z11.s }, z5.s[1]
// CHECK-ENCODING: [0x55,0x45,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1554555 <unknown>

fmls    za.s[w11, 7, vgx2], {z12.s, z13.s}, z8.s[3]  // 11000001-01011000-01101101-10010111
// CHECK-INST: fmls    za.s[w11, 7, vgx2], { z12.s, z13.s }, z8.s[3]
// CHECK-ENCODING: [0x97,0x6d,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1586d97 <unknown>

fmls    za.s[w11, 7], {z12.s, z13.s}, z8.s[3]  // 11000001-01011000-01101101-10010111
// CHECK-INST: fmls    za.s[w11, 7, vgx2], { z12.s, z13.s }, z8.s[3]
// CHECK-ENCODING: [0x97,0x6d,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1586d97 <unknown>

fmls    za.s[w11, 7, vgx2], {z30.s, z31.s}, z15.s[3]  // 11000001-01011111-01101111-11010111
// CHECK-INST: fmls    za.s[w11, 7, vgx2], { z30.s, z31.s }, z15.s[3]
// CHECK-ENCODING: [0xd7,0x6f,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15f6fd7 <unknown>

fmls    za.s[w11, 7], {z30.s, z31.s}, z15.s[3]  // 11000001-01011111-01101111-11010111
// CHECK-INST: fmls    za.s[w11, 7, vgx2], { z30.s, z31.s }, z15.s[3]
// CHECK-ENCODING: [0xd7,0x6f,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15f6fd7 <unknown>

fmls    za.s[w8, 5, vgx2], {z16.s, z17.s}, z0.s[3]  // 11000001-01010000-00001110-00010101
// CHECK-INST: fmls    za.s[w8, 5, vgx2], { z16.s, z17.s }, z0.s[3]
// CHECK-ENCODING: [0x15,0x0e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1500e15 <unknown>

fmls    za.s[w8, 5], {z16.s, z17.s}, z0.s[3]  // 11000001-01010000-00001110-00010101
// CHECK-INST: fmls    za.s[w8, 5, vgx2], { z16.s, z17.s }, z0.s[3]
// CHECK-ENCODING: [0x15,0x0e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1500e15 <unknown>

fmls    za.s[w8, 1, vgx2], {z0.s, z1.s}, z14.s[1]  // 11000001-01011110-00000100-00010001
// CHECK-INST: fmls    za.s[w8, 1, vgx2], { z0.s, z1.s }, z14.s[1]
// CHECK-ENCODING: [0x11,0x04,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e0411 <unknown>

fmls    za.s[w8, 1], {z0.s, z1.s}, z14.s[1]  // 11000001-01011110-00000100-00010001
// CHECK-INST: fmls    za.s[w8, 1, vgx2], { z0.s, z1.s }, z14.s[1]
// CHECK-ENCODING: [0x11,0x04,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e0411 <unknown>

fmls    za.s[w10, 0, vgx2], {z18.s, z19.s}, z4.s[1]  // 11000001-01010100-01000110-01010000
// CHECK-INST: fmls    za.s[w10, 0, vgx2], { z18.s, z19.s }, z4.s[1]
// CHECK-ENCODING: [0x50,0x46,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1544650 <unknown>

fmls    za.s[w10, 0], {z18.s, z19.s}, z4.s[1]  // 11000001-01010100-01000110-01010000
// CHECK-INST: fmls    za.s[w10, 0, vgx2], { z18.s, z19.s }, z4.s[1]
// CHECK-ENCODING: [0x50,0x46,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1544650 <unknown>

fmls    za.s[w8, 0, vgx2], {z12.s, z13.s}, z2.s[2]  // 11000001-01010010-00001001-10010000
// CHECK-INST: fmls    za.s[w8, 0, vgx2], { z12.s, z13.s }, z2.s[2]
// CHECK-ENCODING: [0x90,0x09,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1520990 <unknown>

fmls    za.s[w8, 0], {z12.s, z13.s}, z2.s[2]  // 11000001-01010010-00001001-10010000
// CHECK-INST: fmls    za.s[w8, 0, vgx2], { z12.s, z13.s }, z2.s[2]
// CHECK-ENCODING: [0x90,0x09,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1520990 <unknown>

fmls    za.s[w10, 1, vgx2], {z0.s, z1.s}, z10.s[2]  // 11000001-01011010-01001000-00010001
// CHECK-INST: fmls    za.s[w10, 1, vgx2], { z0.s, z1.s }, z10.s[2]
// CHECK-ENCODING: [0x11,0x48,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15a4811 <unknown>

fmls    za.s[w10, 1], {z0.s, z1.s}, z10.s[2]  // 11000001-01011010-01001000-00010001
// CHECK-INST: fmls    za.s[w10, 1, vgx2], { z0.s, z1.s }, z10.s[2]
// CHECK-ENCODING: [0x11,0x48,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15a4811 <unknown>

fmls    za.s[w8, 5, vgx2], {z22.s, z23.s}, z14.s[2]  // 11000001-01011110-00001010-11010101
// CHECK-INST: fmls    za.s[w8, 5, vgx2], { z22.s, z23.s }, z14.s[2]
// CHECK-ENCODING: [0xd5,0x0a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e0ad5 <unknown>

fmls    za.s[w8, 5], {z22.s, z23.s}, z14.s[2]  // 11000001-01011110-00001010-11010101
// CHECK-INST: fmls    za.s[w8, 5, vgx2], { z22.s, z23.s }, z14.s[2]
// CHECK-ENCODING: [0xd5,0x0a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e0ad5 <unknown>

fmls    za.s[w11, 2, vgx2], {z8.s, z9.s}, z1.s[1]  // 11000001-01010001-01100101-00010010
// CHECK-INST: fmls    za.s[w11, 2, vgx2], { z8.s, z9.s }, z1.s[1]
// CHECK-ENCODING: [0x12,0x65,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1516512 <unknown>

fmls    za.s[w11, 2], {z8.s, z9.s}, z1.s[1]  // 11000001-01010001-01100101-00010010
// CHECK-INST: fmls    za.s[w11, 2, vgx2], { z8.s, z9.s }, z1.s[1]
// CHECK-ENCODING: [0x12,0x65,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1516512 <unknown>

fmls    za.s[w9, 7, vgx2], {z12.s, z13.s}, z11.s[2]  // 11000001-01011011-00101001-10010111
// CHECK-INST: fmls    za.s[w9, 7, vgx2], { z12.s, z13.s }, z11.s[2]
// CHECK-ENCODING: [0x97,0x29,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15b2997 <unknown>

fmls    za.s[w9, 7], {z12.s, z13.s}, z11.s[2]  // 11000001-01011011-00101001-10010111
// CHECK-INST: fmls    za.s[w9, 7, vgx2], { z12.s, z13.s }, z11.s[2]
// CHECK-ENCODING: [0x97,0x29,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15b2997 <unknown>


fmls    za.s[w8, 0, vgx2], {z0.s, z1.s}, {z0.s, z1.s}  // 11000001-10100000-00011000-00001000
// CHECK-INST: fmls    za.s[w8, 0, vgx2], { z0.s, z1.s }, { z0.s, z1.s }
// CHECK-ENCODING: [0x08,0x18,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a01808 <unknown>

fmls    za.s[w8, 0], {z0.s - z1.s}, {z0.s - z1.s}  // 11000001-10100000-00011000-00001000
// CHECK-INST: fmls    za.s[w8, 0, vgx2], { z0.s, z1.s }, { z0.s, z1.s }
// CHECK-ENCODING: [0x08,0x18,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a01808 <unknown>

fmls    za.s[w10, 5, vgx2], {z10.s, z11.s}, {z20.s, z21.s}  // 11000001-10110100-01011001-01001101
// CHECK-INST: fmls    za.s[w10, 5, vgx2], { z10.s, z11.s }, { z20.s, z21.s }
// CHECK-ENCODING: [0x4d,0x59,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b4594d <unknown>

fmls    za.s[w10, 5], {z10.s - z11.s}, {z20.s - z21.s}  // 11000001-10110100-01011001-01001101
// CHECK-INST: fmls    za.s[w10, 5, vgx2], { z10.s, z11.s }, { z20.s, z21.s }
// CHECK-ENCODING: [0x4d,0x59,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b4594d <unknown>

fmls    za.s[w11, 7, vgx2], {z12.s, z13.s}, {z8.s, z9.s}  // 11000001-10101000-01111001-10001111
// CHECK-INST: fmls    za.s[w11, 7, vgx2], { z12.s, z13.s }, { z8.s, z9.s }
// CHECK-ENCODING: [0x8f,0x79,0xa8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a8798f <unknown>

fmls    za.s[w11, 7], {z12.s - z13.s}, {z8.s - z9.s}  // 11000001-10101000-01111001-10001111
// CHECK-INST: fmls    za.s[w11, 7, vgx2], { z12.s, z13.s }, { z8.s, z9.s }
// CHECK-ENCODING: [0x8f,0x79,0xa8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a8798f <unknown>

fmls    za.s[w11, 7, vgx2], {z30.s, z31.s}, {z30.s, z31.s}  // 11000001-10111110-01111011-11001111
// CHECK-INST: fmls    za.s[w11, 7, vgx2], { z30.s, z31.s }, { z30.s, z31.s }
// CHECK-ENCODING: [0xcf,0x7b,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be7bcf <unknown>

fmls    za.s[w11, 7], {z30.s - z31.s}, {z30.s - z31.s}  // 11000001-10111110-01111011-11001111
// CHECK-INST: fmls    za.s[w11, 7, vgx2], { z30.s, z31.s }, { z30.s, z31.s }
// CHECK-ENCODING: [0xcf,0x7b,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be7bcf <unknown>

fmls    za.s[w8, 5, vgx2], {z16.s, z17.s}, {z16.s, z17.s}  // 11000001-10110000-00011010-00001101
// CHECK-INST: fmls    za.s[w8, 5, vgx2], { z16.s, z17.s }, { z16.s, z17.s }
// CHECK-ENCODING: [0x0d,0x1a,0xb0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b01a0d <unknown>

fmls    za.s[w8, 5], {z16.s - z17.s}, {z16.s - z17.s}  // 11000001-10110000-00011010-00001101
// CHECK-INST: fmls    za.s[w8, 5, vgx2], { z16.s, z17.s }, { z16.s, z17.s }
// CHECK-ENCODING: [0x0d,0x1a,0xb0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b01a0d <unknown>

fmls    za.s[w8, 1, vgx2], {z0.s, z1.s}, {z30.s, z31.s}  // 11000001-10111110-00011000-00001001
// CHECK-INST: fmls    za.s[w8, 1, vgx2], { z0.s, z1.s }, { z30.s, z31.s }
// CHECK-ENCODING: [0x09,0x18,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be1809 <unknown>

fmls    za.s[w8, 1], {z0.s - z1.s}, {z30.s - z31.s}  // 11000001-10111110-00011000-00001001
// CHECK-INST: fmls    za.s[w8, 1, vgx2], { z0.s, z1.s }, { z30.s, z31.s }
// CHECK-ENCODING: [0x09,0x18,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be1809 <unknown>

fmls    za.s[w10, 0, vgx2], {z18.s, z19.s}, {z20.s, z21.s}  // 11000001-10110100-01011010-01001000
// CHECK-INST: fmls    za.s[w10, 0, vgx2], { z18.s, z19.s }, { z20.s, z21.s }
// CHECK-ENCODING: [0x48,0x5a,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b45a48 <unknown>

fmls    za.s[w10, 0], {z18.s - z19.s}, {z20.s - z21.s}  // 11000001-10110100-01011010-01001000
// CHECK-INST: fmls    za.s[w10, 0, vgx2], { z18.s, z19.s }, { z20.s, z21.s }
// CHECK-ENCODING: [0x48,0x5a,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b45a48 <unknown>

fmls    za.s[w8, 0, vgx2], {z12.s, z13.s}, {z2.s, z3.s}  // 11000001-10100010-00011001-10001000
// CHECK-INST: fmls    za.s[w8, 0, vgx2], { z12.s, z13.s }, { z2.s, z3.s }
// CHECK-ENCODING: [0x88,0x19,0xa2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a21988 <unknown>

fmls    za.s[w8, 0], {z12.s - z13.s}, {z2.s - z3.s}  // 11000001-10100010-00011001-10001000
// CHECK-INST: fmls    za.s[w8, 0, vgx2], { z12.s, z13.s }, { z2.s, z3.s }
// CHECK-ENCODING: [0x88,0x19,0xa2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a21988 <unknown>

fmls    za.s[w10, 1, vgx2], {z0.s, z1.s}, {z26.s, z27.s}  // 11000001-10111010-01011000-00001001
// CHECK-INST: fmls    za.s[w10, 1, vgx2], { z0.s, z1.s }, { z26.s, z27.s }
// CHECK-ENCODING: [0x09,0x58,0xba,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ba5809 <unknown>

fmls    za.s[w10, 1], {z0.s - z1.s}, {z26.s - z27.s}  // 11000001-10111010-01011000-00001001
// CHECK-INST: fmls    za.s[w10, 1, vgx2], { z0.s, z1.s }, { z26.s, z27.s }
// CHECK-ENCODING: [0x09,0x58,0xba,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ba5809 <unknown>

fmls    za.s[w8, 5, vgx2], {z22.s, z23.s}, {z30.s, z31.s}  // 11000001-10111110-00011010-11001101
// CHECK-INST: fmls    za.s[w8, 5, vgx2], { z22.s, z23.s }, { z30.s, z31.s }
// CHECK-ENCODING: [0xcd,0x1a,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be1acd <unknown>

fmls    za.s[w8, 5], {z22.s - z23.s}, {z30.s - z31.s}  // 11000001-10111110-00011010-11001101
// CHECK-INST: fmls    za.s[w8, 5, vgx2], { z22.s, z23.s }, { z30.s, z31.s }
// CHECK-ENCODING: [0xcd,0x1a,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be1acd <unknown>

fmls    za.s[w11, 2, vgx2], {z8.s, z9.s}, {z0.s, z1.s}  // 11000001-10100000-01111001-00001010
// CHECK-INST: fmls    za.s[w11, 2, vgx2], { z8.s, z9.s }, { z0.s, z1.s }
// CHECK-ENCODING: [0x0a,0x79,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a0790a <unknown>

fmls    za.s[w11, 2], {z8.s - z9.s}, {z0.s - z1.s}  // 11000001-10100000-01111001-00001010
// CHECK-INST: fmls    za.s[w11, 2, vgx2], { z8.s, z9.s }, { z0.s, z1.s }
// CHECK-ENCODING: [0x0a,0x79,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a0790a <unknown>

fmls    za.s[w9, 7, vgx2], {z12.s, z13.s}, {z10.s, z11.s}  // 11000001-10101010-00111001-10001111
// CHECK-INST: fmls    za.s[w9, 7, vgx2], { z12.s, z13.s }, { z10.s, z11.s }
// CHECK-ENCODING: [0x8f,0x39,0xaa,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1aa398f <unknown>

fmls    za.s[w9, 7], {z12.s - z13.s}, {z10.s - z11.s}  // 11000001-10101010-00111001-10001111
// CHECK-INST: fmls    za.s[w9, 7, vgx2], { z12.s, z13.s }, { z10.s, z11.s }
// CHECK-ENCODING: [0x8f,0x39,0xaa,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1aa398f <unknown>


fmls    za.d[w8, 0, vgx4], {z0.d - z3.d}, z0.d  // 11000001-01110000-00011000-00001000
// CHECK-INST: fmls    za.d[w8, 0, vgx4], { z0.d - z3.d }, z0.d
// CHECK-ENCODING: [0x08,0x18,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1701808 <unknown>

fmls    za.d[w8, 0], {z0.d - z3.d}, z0.d  // 11000001-01110000-00011000-00001000
// CHECK-INST: fmls    za.d[w8, 0, vgx4], { z0.d - z3.d }, z0.d
// CHECK-ENCODING: [0x08,0x18,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1701808 <unknown>

fmls    za.d[w10, 5, vgx4], {z10.d - z13.d}, z5.d  // 11000001-01110101-01011001-01001101
// CHECK-INST: fmls    za.d[w10, 5, vgx4], { z10.d - z13.d }, z5.d
// CHECK-ENCODING: [0x4d,0x59,0x75,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c175594d <unknown>

fmls    za.d[w10, 5], {z10.d - z13.d}, z5.d  // 11000001-01110101-01011001-01001101
// CHECK-INST: fmls    za.d[w10, 5, vgx4], { z10.d - z13.d }, z5.d
// CHECK-ENCODING: [0x4d,0x59,0x75,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c175594d <unknown>

fmls    za.d[w11, 7, vgx4], {z13.d - z16.d}, z8.d  // 11000001-01111000-01111001-10101111
// CHECK-INST: fmls    za.d[w11, 7, vgx4], { z13.d - z16.d }, z8.d
// CHECK-ENCODING: [0xaf,0x79,0x78,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17879af <unknown>

fmls    za.d[w11, 7], {z13.d - z16.d}, z8.d  // 11000001-01111000-01111001-10101111
// CHECK-INST: fmls    za.d[w11, 7, vgx4], { z13.d - z16.d }, z8.d
// CHECK-ENCODING: [0xaf,0x79,0x78,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17879af <unknown>

fmls    za.d[w11, 7, vgx4], {z31.d - z2.d}, z15.d  // 11000001-01111111-01111011-11101111
// CHECK-INST: fmls    za.d[w11, 7, vgx4], { z31.d, z0.d, z1.d, z2.d }, z15.d
// CHECK-ENCODING: [0xef,0x7b,0x7f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17f7bef <unknown>

fmls    za.d[w11, 7], {z31.d - z2.d}, z15.d  // 11000001-01111111-01111011-11101111
// CHECK-INST: fmls    za.d[w11, 7, vgx4], { z31.d, z0.d, z1.d, z2.d }, z15.d
// CHECK-ENCODING: [0xef,0x7b,0x7f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17f7bef <unknown>

fmls    za.d[w8, 5, vgx4], {z17.d - z20.d}, z0.d  // 11000001-01110000-00011010-00101101
// CHECK-INST: fmls    za.d[w8, 5, vgx4], { z17.d - z20.d }, z0.d
// CHECK-ENCODING: [0x2d,0x1a,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1701a2d <unknown>

fmls    za.d[w8, 5], {z17.d - z20.d}, z0.d  // 11000001-01110000-00011010-00101101
// CHECK-INST: fmls    za.d[w8, 5, vgx4], { z17.d - z20.d }, z0.d
// CHECK-ENCODING: [0x2d,0x1a,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1701a2d <unknown>

fmls    za.d[w8, 1, vgx4], {z1.d - z4.d}, z14.d  // 11000001-01111110-00011000-00101001
// CHECK-INST: fmls    za.d[w8, 1, vgx4], { z1.d - z4.d }, z14.d
// CHECK-ENCODING: [0x29,0x18,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17e1829 <unknown>

fmls    za.d[w8, 1], {z1.d - z4.d}, z14.d  // 11000001-01111110-00011000-00101001
// CHECK-INST: fmls    za.d[w8, 1, vgx4], { z1.d - z4.d }, z14.d
// CHECK-ENCODING: [0x29,0x18,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17e1829 <unknown>

fmls    za.d[w10, 0, vgx4], {z19.d - z22.d}, z4.d  // 11000001-01110100-01011010-01101000
// CHECK-INST: fmls    za.d[w10, 0, vgx4], { z19.d - z22.d }, z4.d
// CHECK-ENCODING: [0x68,0x5a,0x74,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1745a68 <unknown>

fmls    za.d[w10, 0], {z19.d - z22.d}, z4.d  // 11000001-01110100-01011010-01101000
// CHECK-INST: fmls    za.d[w10, 0, vgx4], { z19.d - z22.d }, z4.d
// CHECK-ENCODING: [0x68,0x5a,0x74,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1745a68 <unknown>

fmls    za.d[w8, 0, vgx4], {z12.d - z15.d}, z2.d  // 11000001-01110010-00011001-10001000
// CHECK-INST: fmls    za.d[w8, 0, vgx4], { z12.d - z15.d }, z2.d
// CHECK-ENCODING: [0x88,0x19,0x72,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1721988 <unknown>

fmls    za.d[w8, 0], {z12.d - z15.d}, z2.d  // 11000001-01110010-00011001-10001000
// CHECK-INST: fmls    za.d[w8, 0, vgx4], { z12.d - z15.d }, z2.d
// CHECK-ENCODING: [0x88,0x19,0x72,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1721988 <unknown>

fmls    za.d[w10, 1, vgx4], {z1.d - z4.d}, z10.d  // 11000001-01111010-01011000-00101001
// CHECK-INST: fmls    za.d[w10, 1, vgx4], { z1.d - z4.d }, z10.d
// CHECK-ENCODING: [0x29,0x58,0x7a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17a5829 <unknown>

fmls    za.d[w10, 1], {z1.d - z4.d}, z10.d  // 11000001-01111010-01011000-00101001
// CHECK-INST: fmls    za.d[w10, 1, vgx4], { z1.d - z4.d }, z10.d
// CHECK-ENCODING: [0x29,0x58,0x7a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17a5829 <unknown>

fmls    za.d[w8, 5, vgx4], {z22.d - z25.d}, z14.d  // 11000001-01111110-00011010-11001101
// CHECK-INST: fmls    za.d[w8, 5, vgx4], { z22.d - z25.d }, z14.d
// CHECK-ENCODING: [0xcd,0x1a,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17e1acd <unknown>

fmls    za.d[w8, 5], {z22.d - z25.d}, z14.d  // 11000001-01111110-00011010-11001101
// CHECK-INST: fmls    za.d[w8, 5, vgx4], { z22.d - z25.d }, z14.d
// CHECK-ENCODING: [0xcd,0x1a,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17e1acd <unknown>

fmls    za.d[w11, 2, vgx4], {z9.d - z12.d}, z1.d  // 11000001-01110001-01111001-00101010
// CHECK-INST: fmls    za.d[w11, 2, vgx4], { z9.d - z12.d }, z1.d
// CHECK-ENCODING: [0x2a,0x79,0x71,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c171792a <unknown>

fmls    za.d[w11, 2], {z9.d - z12.d}, z1.d  // 11000001-01110001-01111001-00101010
// CHECK-INST: fmls    za.d[w11, 2, vgx4], { z9.d - z12.d }, z1.d
// CHECK-ENCODING: [0x2a,0x79,0x71,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c171792a <unknown>

fmls    za.d[w9, 7, vgx4], {z12.d - z15.d}, z11.d  // 11000001-01111011-00111001-10001111
// CHECK-INST: fmls    za.d[w9, 7, vgx4], { z12.d - z15.d }, z11.d
// CHECK-ENCODING: [0x8f,0x39,0x7b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17b398f <unknown>

fmls    za.d[w9, 7], {z12.d - z15.d}, z11.d  // 11000001-01111011-00111001-10001111
// CHECK-INST: fmls    za.d[w9, 7, vgx4], { z12.d - z15.d }, z11.d
// CHECK-ENCODING: [0x8f,0x39,0x7b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17b398f <unknown>


fmls    za.d[w8, 0, vgx4], {z0.d - z3.d}, z0.d[0]  // 11000001-11010000-10000000-00010000
// CHECK-INST: fmls    za.d[w8, 0, vgx4], { z0.d - z3.d }, z0.d[0]
// CHECK-ENCODING: [0x10,0x80,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d08010 <unknown>

fmls    za.d[w8, 0], {z0.d - z3.d}, z0.d[0]  // 11000001-11010000-10000000-00010000
// CHECK-INST: fmls    za.d[w8, 0, vgx4], { z0.d - z3.d }, z0.d[0]
// CHECK-ENCODING: [0x10,0x80,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d08010 <unknown>

fmls    za.d[w10, 5, vgx4], {z8.d - z11.d}, z5.d[1]  // 11000001-11010101-11000101-00010101
// CHECK-INST: fmls    za.d[w10, 5, vgx4], { z8.d - z11.d }, z5.d[1]
// CHECK-ENCODING: [0x15,0xc5,0xd5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d5c515 <unknown>

fmls    za.d[w10, 5], {z8.d - z11.d}, z5.d[1]  // 11000001-11010101-11000101-00010101
// CHECK-INST: fmls    za.d[w10, 5, vgx4], { z8.d - z11.d }, z5.d[1]
// CHECK-ENCODING: [0x15,0xc5,0xd5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d5c515 <unknown>

fmls    za.d[w11, 7, vgx4], {z12.d - z15.d}, z8.d[1]  // 11000001-11011000-11100101-10010111
// CHECK-INST: fmls    za.d[w11, 7, vgx4], { z12.d - z15.d }, z8.d[1]
// CHECK-ENCODING: [0x97,0xe5,0xd8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d8e597 <unknown>

fmls    za.d[w11, 7], {z12.d - z15.d}, z8.d[1]  // 11000001-11011000-11100101-10010111
// CHECK-INST: fmls    za.d[w11, 7, vgx4], { z12.d - z15.d }, z8.d[1]
// CHECK-ENCODING: [0x97,0xe5,0xd8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d8e597 <unknown>

fmls    za.d[w11, 7, vgx4], {z28.d - z31.d}, z15.d[1]  // 11000001-11011111-11100111-10010111
// CHECK-INST: fmls    za.d[w11, 7, vgx4], { z28.d - z31.d }, z15.d[1]
// CHECK-ENCODING: [0x97,0xe7,0xdf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dfe797 <unknown>

fmls    za.d[w11, 7], {z28.d - z31.d}, z15.d[1]  // 11000001-11011111-11100111-10010111
// CHECK-INST: fmls    za.d[w11, 7, vgx4], { z28.d - z31.d }, z15.d[1]
// CHECK-ENCODING: [0x97,0xe7,0xdf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dfe797 <unknown>

fmls    za.d[w8, 5, vgx4], {z16.d - z19.d}, z0.d[1]  // 11000001-11010000-10000110-00010101
// CHECK-INST: fmls    za.d[w8, 5, vgx4], { z16.d - z19.d }, z0.d[1]
// CHECK-ENCODING: [0x15,0x86,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d08615 <unknown>

fmls    za.d[w8, 5], {z16.d - z19.d}, z0.d[1]  // 11000001-11010000-10000110-00010101
// CHECK-INST: fmls    za.d[w8, 5, vgx4], { z16.d - z19.d }, z0.d[1]
// CHECK-ENCODING: [0x15,0x86,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d08615 <unknown>

fmls    za.d[w8, 1, vgx4], {z0.d - z3.d}, z14.d[1]  // 11000001-11011110-10000100-00010001
// CHECK-INST: fmls    za.d[w8, 1, vgx4], { z0.d - z3.d }, z14.d[1]
// CHECK-ENCODING: [0x11,0x84,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de8411 <unknown>

fmls    za.d[w8, 1], {z0.d - z3.d}, z14.d[1]  // 11000001-11011110-10000100-00010001
// CHECK-INST: fmls    za.d[w8, 1, vgx4], { z0.d - z3.d }, z14.d[1]
// CHECK-ENCODING: [0x11,0x84,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de8411 <unknown>

fmls    za.d[w10, 0, vgx4], {z16.d - z19.d}, z4.d[1]  // 11000001-11010100-11000110-00010000
// CHECK-INST: fmls    za.d[w10, 0, vgx4], { z16.d - z19.d }, z4.d[1]
// CHECK-ENCODING: [0x10,0xc6,0xd4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d4c610 <unknown>

fmls    za.d[w10, 0], {z16.d - z19.d}, z4.d[1]  // 11000001-11010100-11000110-00010000
// CHECK-INST: fmls    za.d[w10, 0, vgx4], { z16.d - z19.d }, z4.d[1]
// CHECK-ENCODING: [0x10,0xc6,0xd4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d4c610 <unknown>

fmls    za.d[w8, 0, vgx4], {z12.d - z15.d}, z2.d[0]  // 11000001-11010010-10000001-10010000
// CHECK-INST: fmls    za.d[w8, 0, vgx4], { z12.d - z15.d }, z2.d[0]
// CHECK-ENCODING: [0x90,0x81,0xd2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d28190 <unknown>

fmls    za.d[w8, 0], {z12.d - z15.d}, z2.d[0]  // 11000001-11010010-10000001-10010000
// CHECK-INST: fmls    za.d[w8, 0, vgx4], { z12.d - z15.d }, z2.d[0]
// CHECK-ENCODING: [0x90,0x81,0xd2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d28190 <unknown>

fmls    za.d[w10, 1, vgx4], {z0.d - z3.d}, z10.d[0]  // 11000001-11011010-11000000-00010001
// CHECK-INST: fmls    za.d[w10, 1, vgx4], { z0.d - z3.d }, z10.d[0]
// CHECK-ENCODING: [0x11,0xc0,0xda,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dac011 <unknown>

fmls    za.d[w10, 1], {z0.d - z3.d}, z10.d[0]  // 11000001-11011010-11000000-00010001
// CHECK-INST: fmls    za.d[w10, 1, vgx4], { z0.d - z3.d }, z10.d[0]
// CHECK-ENCODING: [0x11,0xc0,0xda,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dac011 <unknown>

fmls    za.d[w8, 5, vgx4], {z20.d - z23.d}, z14.d[0]  // 11000001-11011110-10000010-10010101
// CHECK-INST: fmls    za.d[w8, 5, vgx4], { z20.d - z23.d }, z14.d[0]
// CHECK-ENCODING: [0x95,0x82,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de8295 <unknown>

fmls    za.d[w8, 5], {z20.d - z23.d}, z14.d[0]  // 11000001-11011110-10000010-10010101
// CHECK-INST: fmls    za.d[w8, 5, vgx4], { z20.d - z23.d }, z14.d[0]
// CHECK-ENCODING: [0x95,0x82,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de8295 <unknown>

fmls    za.d[w11, 2, vgx4], {z8.d - z11.d}, z1.d[1]  // 11000001-11010001-11100101-00010010
// CHECK-INST: fmls    za.d[w11, 2, vgx4], { z8.d - z11.d }, z1.d[1]
// CHECK-ENCODING: [0x12,0xe5,0xd1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d1e512 <unknown>

fmls    za.d[w11, 2], {z8.d - z11.d}, z1.d[1]  // 11000001-11010001-11100101-00010010
// CHECK-INST: fmls    za.d[w11, 2, vgx4], { z8.d - z11.d }, z1.d[1]
// CHECK-ENCODING: [0x12,0xe5,0xd1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d1e512 <unknown>

fmls    za.d[w9, 7, vgx4], {z12.d - z15.d}, z11.d[0]  // 11000001-11011011-10100001-10010111
// CHECK-INST: fmls    za.d[w9, 7, vgx4], { z12.d - z15.d }, z11.d[0]
// CHECK-ENCODING: [0x97,0xa1,0xdb,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dba197 <unknown>

fmls    za.d[w9, 7], {z12.d - z15.d}, z11.d[0]  // 11000001-11011011-10100001-10010111
// CHECK-INST: fmls    za.d[w9, 7, vgx4], { z12.d - z15.d }, z11.d[0]
// CHECK-ENCODING: [0x97,0xa1,0xdb,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dba197 <unknown>


fmls    za.d[w8, 0, vgx4], {z0.d - z3.d}, {z0.d - z3.d}  // 11000001-11100001-00011000-00001000
// CHECK-INST: fmls    za.d[w8, 0, vgx4], { z0.d - z3.d }, { z0.d - z3.d }
// CHECK-ENCODING: [0x08,0x18,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11808 <unknown>

fmls    za.d[w8, 0], {z0.d - z3.d}, {z0.d - z3.d}  // 11000001-11100001-00011000-00001000
// CHECK-INST: fmls    za.d[w8, 0, vgx4], { z0.d - z3.d }, { z0.d - z3.d }
// CHECK-ENCODING: [0x08,0x18,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11808 <unknown>

fmls    za.d[w10, 5, vgx4], {z8.d - z11.d}, {z20.d - z23.d}  // 11000001-11110101-01011001-00001101
// CHECK-INST: fmls    za.d[w10, 5, vgx4], { z8.d - z11.d }, { z20.d - z23.d }
// CHECK-ENCODING: [0x0d,0x59,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f5590d <unknown>

fmls    za.d[w10, 5], {z8.d - z11.d}, {z20.d - z23.d}  // 11000001-11110101-01011001-00001101
// CHECK-INST: fmls    za.d[w10, 5, vgx4], { z8.d - z11.d }, { z20.d - z23.d }
// CHECK-ENCODING: [0x0d,0x59,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f5590d <unknown>

fmls    za.d[w11, 7, vgx4], {z12.d - z15.d}, {z8.d - z11.d}  // 11000001-11101001-01111001-10001111
// CHECK-INST: fmls    za.d[w11, 7, vgx4], { z12.d - z15.d }, { z8.d - z11.d }
// CHECK-ENCODING: [0x8f,0x79,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e9798f <unknown>

fmls    za.d[w11, 7], {z12.d - z15.d}, {z8.d - z11.d}  // 11000001-11101001-01111001-10001111
// CHECK-INST: fmls    za.d[w11, 7, vgx4], { z12.d - z15.d }, { z8.d - z11.d }
// CHECK-ENCODING: [0x8f,0x79,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e9798f <unknown>

fmls    za.d[w11, 7, vgx4], {z28.d - z31.d}, {z28.d - z31.d}  // 11000001-11111101-01111011-10001111
// CHECK-INST: fmls    za.d[w11, 7, vgx4], { z28.d - z31.d }, { z28.d - z31.d }
// CHECK-ENCODING: [0x8f,0x7b,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd7b8f <unknown>

fmls    za.d[w11, 7], {z28.d - z31.d}, {z28.d - z31.d}  // 11000001-11111101-01111011-10001111
// CHECK-INST: fmls    za.d[w11, 7, vgx4], { z28.d - z31.d }, { z28.d - z31.d }
// CHECK-ENCODING: [0x8f,0x7b,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd7b8f <unknown>

fmls    za.d[w8, 5, vgx4], {z16.d - z19.d}, {z16.d - z19.d}  // 11000001-11110001-00011010-00001101
// CHECK-INST: fmls    za.d[w8, 5, vgx4], { z16.d - z19.d }, { z16.d - z19.d }
// CHECK-ENCODING: [0x0d,0x1a,0xf1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f11a0d <unknown>

fmls    za.d[w8, 5], {z16.d - z19.d}, {z16.d - z19.d}  // 11000001-11110001-00011010-00001101
// CHECK-INST: fmls    za.d[w8, 5, vgx4], { z16.d - z19.d }, { z16.d - z19.d }
// CHECK-ENCODING: [0x0d,0x1a,0xf1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f11a0d <unknown>

fmls    za.d[w8, 1, vgx4], {z0.d - z3.d}, {z28.d - z31.d}  // 11000001-11111101-00011000-00001001
// CHECK-INST: fmls    za.d[w8, 1, vgx4], { z0.d - z3.d }, { z28.d - z31.d }
// CHECK-ENCODING: [0x09,0x18,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd1809 <unknown>

fmls    za.d[w8, 1], {z0.d - z3.d}, {z28.d - z31.d}  // 11000001-11111101-00011000-00001001
// CHECK-INST: fmls    za.d[w8, 1, vgx4], { z0.d - z3.d }, { z28.d - z31.d }
// CHECK-ENCODING: [0x09,0x18,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd1809 <unknown>

fmls    za.d[w10, 0, vgx4], {z16.d - z19.d}, {z20.d - z23.d}  // 11000001-11110101-01011010-00001000
// CHECK-INST: fmls    za.d[w10, 0, vgx4], { z16.d - z19.d }, { z20.d - z23.d }
// CHECK-ENCODING: [0x08,0x5a,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f55a08 <unknown>

fmls    za.d[w10, 0], {z16.d - z19.d}, {z20.d - z23.d}  // 11000001-11110101-01011010-00001000
// CHECK-INST: fmls    za.d[w10, 0, vgx4], { z16.d - z19.d }, { z20.d - z23.d }
// CHECK-ENCODING: [0x08,0x5a,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f55a08 <unknown>

fmls    za.d[w8, 0, vgx4], {z12.d - z15.d}, {z0.d - z3.d}  // 11000001-11100001-00011001-10001000
// CHECK-INST: fmls    za.d[w8, 0, vgx4], { z12.d - z15.d }, { z0.d - z3.d }
// CHECK-ENCODING: [0x88,0x19,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11988 <unknown>

fmls    za.d[w8, 0], {z12.d - z15.d}, {z0.d - z3.d}  // 11000001-11100001-00011001-10001000
// CHECK-INST: fmls    za.d[w8, 0, vgx4], { z12.d - z15.d }, { z0.d - z3.d }
// CHECK-ENCODING: [0x88,0x19,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11988 <unknown>

fmls    za.d[w10, 1, vgx4], {z0.d - z3.d}, {z24.d - z27.d}  // 11000001-11111001-01011000-00001001
// CHECK-INST: fmls    za.d[w10, 1, vgx4], { z0.d - z3.d }, { z24.d - z27.d }
// CHECK-ENCODING: [0x09,0x58,0xf9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f95809 <unknown>

fmls    za.d[w10, 1], {z0.d - z3.d}, {z24.d - z27.d}  // 11000001-11111001-01011000-00001001
// CHECK-INST: fmls    za.d[w10, 1, vgx4], { z0.d - z3.d }, { z24.d - z27.d }
// CHECK-ENCODING: [0x09,0x58,0xf9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f95809 <unknown>

fmls    za.d[w8, 5, vgx4], {z20.d - z23.d}, {z28.d - z31.d}  // 11000001-11111101-00011010-10001101
// CHECK-INST: fmls    za.d[w8, 5, vgx4], { z20.d - z23.d }, { z28.d - z31.d }
// CHECK-ENCODING: [0x8d,0x1a,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd1a8d <unknown>

fmls    za.d[w8, 5], {z20.d - z23.d}, {z28.d - z31.d}  // 11000001-11111101-00011010-10001101
// CHECK-INST: fmls    za.d[w8, 5, vgx4], { z20.d - z23.d }, { z28.d - z31.d }
// CHECK-ENCODING: [0x8d,0x1a,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd1a8d <unknown>

fmls    za.d[w11, 2, vgx4], {z8.d - z11.d}, {z0.d - z3.d}  // 11000001-11100001-01111001-00001010
// CHECK-INST: fmls    za.d[w11, 2, vgx4], { z8.d - z11.d }, { z0.d - z3.d }
// CHECK-ENCODING: [0x0a,0x79,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e1790a <unknown>

fmls    za.d[w11, 2], {z8.d - z11.d}, {z0.d - z3.d}  // 11000001-11100001-01111001-00001010
// CHECK-INST: fmls    za.d[w11, 2, vgx4], { z8.d - z11.d }, { z0.d - z3.d }
// CHECK-ENCODING: [0x0a,0x79,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e1790a <unknown>

fmls    za.d[w9, 7, vgx4], {z12.d - z15.d}, {z8.d - z11.d}  // 11000001-11101001-00111001-10001111
// CHECK-INST: fmls    za.d[w9, 7, vgx4], { z12.d - z15.d }, { z8.d - z11.d }
// CHECK-ENCODING: [0x8f,0x39,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e9398f <unknown>

fmls    za.d[w9, 7], {z12.d - z15.d}, {z8.d - z11.d}  // 11000001-11101001-00111001-10001111
// CHECK-INST: fmls    za.d[w9, 7, vgx4], { z12.d - z15.d }, { z8.d - z11.d }
// CHECK-ENCODING: [0x8f,0x39,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e9398f <unknown>


fmls    za.s[w8, 0, vgx4], {z0.s - z3.s}, z0.s  // 11000001-00110000-00011000-00001000
// CHECK-INST: fmls    za.s[w8, 0, vgx4], { z0.s - z3.s }, z0.s
// CHECK-ENCODING: [0x08,0x18,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1301808 <unknown>

fmls    za.s[w8, 0], {z0.s - z3.s}, z0.s  // 11000001-00110000-00011000-00001000
// CHECK-INST: fmls    za.s[w8, 0, vgx4], { z0.s - z3.s }, z0.s
// CHECK-ENCODING: [0x08,0x18,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1301808 <unknown>

fmls    za.s[w10, 5, vgx4], {z10.s - z13.s}, z5.s  // 11000001-00110101-01011001-01001101
// CHECK-INST: fmls    za.s[w10, 5, vgx4], { z10.s - z13.s }, z5.s
// CHECK-ENCODING: [0x4d,0x59,0x35,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c135594d <unknown>

fmls    za.s[w10, 5], {z10.s - z13.s}, z5.s  // 11000001-00110101-01011001-01001101
// CHECK-INST: fmls    za.s[w10, 5, vgx4], { z10.s - z13.s }, z5.s
// CHECK-ENCODING: [0x4d,0x59,0x35,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c135594d <unknown>

fmls    za.s[w11, 7, vgx4], {z13.s - z16.s}, z8.s  // 11000001-00111000-01111001-10101111
// CHECK-INST: fmls    za.s[w11, 7, vgx4], { z13.s - z16.s }, z8.s
// CHECK-ENCODING: [0xaf,0x79,0x38,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13879af <unknown>

fmls    za.s[w11, 7], {z13.s - z16.s}, z8.s  // 11000001-00111000-01111001-10101111
// CHECK-INST: fmls    za.s[w11, 7, vgx4], { z13.s - z16.s }, z8.s
// CHECK-ENCODING: [0xaf,0x79,0x38,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13879af <unknown>

fmls    za.s[w11, 7, vgx4], {z31.s - z2.s}, z15.s  // 11000001-00111111-01111011-11101111
// CHECK-INST: fmls    za.s[w11, 7, vgx4], { z31.s, z0.s, z1.s, z2.s }, z15.s
// CHECK-ENCODING: [0xef,0x7b,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13f7bef <unknown>

fmls    za.s[w11, 7], {z31.s - z2.s}, z15.s  // 11000001-00111111-01111011-11101111
// CHECK-INST: fmls    za.s[w11, 7, vgx4], { z31.s, z0.s, z1.s, z2.s }, z15.s
// CHECK-ENCODING: [0xef,0x7b,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13f7bef <unknown>

fmls    za.s[w8, 5, vgx4], {z17.s - z20.s}, z0.s  // 11000001-00110000-00011010-00101101
// CHECK-INST: fmls    za.s[w8, 5, vgx4], { z17.s - z20.s }, z0.s
// CHECK-ENCODING: [0x2d,0x1a,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1301a2d <unknown>

fmls    za.s[w8, 5], {z17.s - z20.s}, z0.s  // 11000001-00110000-00011010-00101101
// CHECK-INST: fmls    za.s[w8, 5, vgx4], { z17.s - z20.s }, z0.s
// CHECK-ENCODING: [0x2d,0x1a,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1301a2d <unknown>

fmls    za.s[w8, 1, vgx4], {z1.s - z4.s}, z14.s  // 11000001-00111110-00011000-00101001
// CHECK-INST: fmls    za.s[w8, 1, vgx4], { z1.s - z4.s }, z14.s
// CHECK-ENCODING: [0x29,0x18,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e1829 <unknown>

fmls    za.s[w8, 1], {z1.s - z4.s}, z14.s  // 11000001-00111110-00011000-00101001
// CHECK-INST: fmls    za.s[w8, 1, vgx4], { z1.s - z4.s }, z14.s
// CHECK-ENCODING: [0x29,0x18,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e1829 <unknown>

fmls    za.s[w10, 0, vgx4], {z19.s - z22.s}, z4.s  // 11000001-00110100-01011010-01101000
// CHECK-INST: fmls    za.s[w10, 0, vgx4], { z19.s - z22.s }, z4.s
// CHECK-ENCODING: [0x68,0x5a,0x34,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1345a68 <unknown>

fmls    za.s[w10, 0], {z19.s - z22.s}, z4.s  // 11000001-00110100-01011010-01101000
// CHECK-INST: fmls    za.s[w10, 0, vgx4], { z19.s - z22.s }, z4.s
// CHECK-ENCODING: [0x68,0x5a,0x34,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1345a68 <unknown>

fmls    za.s[w8, 0, vgx4], {z12.s - z15.s}, z2.s  // 11000001-00110010-00011001-10001000
// CHECK-INST: fmls    za.s[w8, 0, vgx4], { z12.s - z15.s }, z2.s
// CHECK-ENCODING: [0x88,0x19,0x32,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1321988 <unknown>

fmls    za.s[w8, 0], {z12.s - z15.s}, z2.s  // 11000001-00110010-00011001-10001000
// CHECK-INST: fmls    za.s[w8, 0, vgx4], { z12.s - z15.s }, z2.s
// CHECK-ENCODING: [0x88,0x19,0x32,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1321988 <unknown>

fmls    za.s[w10, 1, vgx4], {z1.s - z4.s}, z10.s  // 11000001-00111010-01011000-00101001
// CHECK-INST: fmls    za.s[w10, 1, vgx4], { z1.s - z4.s }, z10.s
// CHECK-ENCODING: [0x29,0x58,0x3a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13a5829 <unknown>

fmls    za.s[w10, 1], {z1.s - z4.s}, z10.s  // 11000001-00111010-01011000-00101001
// CHECK-INST: fmls    za.s[w10, 1, vgx4], { z1.s - z4.s }, z10.s
// CHECK-ENCODING: [0x29,0x58,0x3a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13a5829 <unknown>

fmls    za.s[w8, 5, vgx4], {z22.s - z25.s}, z14.s  // 11000001-00111110-00011010-11001101
// CHECK-INST: fmls    za.s[w8, 5, vgx4], { z22.s - z25.s }, z14.s
// CHECK-ENCODING: [0xcd,0x1a,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e1acd <unknown>

fmls    za.s[w8, 5], {z22.s - z25.s}, z14.s  // 11000001-00111110-00011010-11001101
// CHECK-INST: fmls    za.s[w8, 5, vgx4], { z22.s - z25.s }, z14.s
// CHECK-ENCODING: [0xcd,0x1a,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e1acd <unknown>

fmls    za.s[w11, 2, vgx4], {z9.s - z12.s}, z1.s  // 11000001-00110001-01111001-00101010
// CHECK-INST: fmls    za.s[w11, 2, vgx4], { z9.s - z12.s }, z1.s
// CHECK-ENCODING: [0x2a,0x79,0x31,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c131792a <unknown>

fmls    za.s[w11, 2], {z9.s - z12.s}, z1.s  // 11000001-00110001-01111001-00101010
// CHECK-INST: fmls    za.s[w11, 2, vgx4], { z9.s - z12.s }, z1.s
// CHECK-ENCODING: [0x2a,0x79,0x31,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c131792a <unknown>

fmls    za.s[w9, 7, vgx4], {z12.s - z15.s}, z11.s  // 11000001-00111011-00111001-10001111
// CHECK-INST: fmls    za.s[w9, 7, vgx4], { z12.s - z15.s }, z11.s
// CHECK-ENCODING: [0x8f,0x39,0x3b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13b398f <unknown>

fmls    za.s[w9, 7], {z12.s - z15.s}, z11.s  // 11000001-00111011-00111001-10001111
// CHECK-INST: fmls    za.s[w9, 7, vgx4], { z12.s - z15.s }, z11.s
// CHECK-ENCODING: [0x8f,0x39,0x3b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13b398f <unknown>


fmls    za.s[w8, 0, vgx4], {z0.s - z3.s}, z0.s[0]  // 11000001-01010000-10000000-00010000
// CHECK-INST: fmls    za.s[w8, 0, vgx4], { z0.s - z3.s }, z0.s[0]
// CHECK-ENCODING: [0x10,0x80,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1508010 <unknown>

fmls    za.s[w8, 0], {z0.s - z3.s}, z0.s[0]  // 11000001-01010000-10000000-00010000
// CHECK-INST: fmls    za.s[w8, 0, vgx4], { z0.s - z3.s }, z0.s[0]
// CHECK-ENCODING: [0x10,0x80,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1508010 <unknown>

fmls    za.s[w10, 5, vgx4], {z8.s - z11.s}, z5.s[1]  // 11000001-01010101-11000101-00010101
// CHECK-INST: fmls    za.s[w10, 5, vgx4], { z8.s - z11.s }, z5.s[1]
// CHECK-ENCODING: [0x15,0xc5,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c155c515 <unknown>

fmls    za.s[w10, 5], {z8.s - z11.s}, z5.s[1]  // 11000001-01010101-11000101-00010101
// CHECK-INST: fmls    za.s[w10, 5, vgx4], { z8.s - z11.s }, z5.s[1]
// CHECK-ENCODING: [0x15,0xc5,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c155c515 <unknown>

fmls    za.s[w11, 7, vgx4], {z12.s - z15.s}, z8.s[3]  // 11000001-01011000-11101101-10010111
// CHECK-INST: fmls    za.s[w11, 7, vgx4], { z12.s - z15.s }, z8.s[3]
// CHECK-ENCODING: [0x97,0xed,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c158ed97 <unknown>

fmls    za.s[w11, 7], {z12.s - z15.s}, z8.s[3]  // 11000001-01011000-11101101-10010111
// CHECK-INST: fmls    za.s[w11, 7, vgx4], { z12.s - z15.s }, z8.s[3]
// CHECK-ENCODING: [0x97,0xed,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c158ed97 <unknown>

fmls    za.s[w11, 7, vgx4], {z28.s - z31.s}, z15.s[3]  // 11000001-01011111-11101111-10010111
// CHECK-INST: fmls    za.s[w11, 7, vgx4], { z28.s - z31.s }, z15.s[3]
// CHECK-ENCODING: [0x97,0xef,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15fef97 <unknown>

fmls    za.s[w11, 7], {z28.s - z31.s}, z15.s[3]  // 11000001-01011111-11101111-10010111
// CHECK-INST: fmls    za.s[w11, 7, vgx4], { z28.s - z31.s }, z15.s[3]
// CHECK-ENCODING: [0x97,0xef,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15fef97 <unknown>

fmls    za.s[w8, 5, vgx4], {z16.s - z19.s}, z0.s[3]  // 11000001-01010000-10001110-00010101
// CHECK-INST: fmls    za.s[w8, 5, vgx4], { z16.s - z19.s }, z0.s[3]
// CHECK-ENCODING: [0x15,0x8e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1508e15 <unknown>

fmls    za.s[w8, 5], {z16.s - z19.s}, z0.s[3]  // 11000001-01010000-10001110-00010101
// CHECK-INST: fmls    za.s[w8, 5, vgx4], { z16.s - z19.s }, z0.s[3]
// CHECK-ENCODING: [0x15,0x8e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1508e15 <unknown>

fmls    za.s[w8, 1, vgx4], {z0.s - z3.s}, z14.s[1]  // 11000001-01011110-10000100-00010001
// CHECK-INST: fmls    za.s[w8, 1, vgx4], { z0.s - z3.s }, z14.s[1]
// CHECK-ENCODING: [0x11,0x84,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e8411 <unknown>

fmls    za.s[w8, 1], {z0.s - z3.s}, z14.s[1]  // 11000001-01011110-10000100-00010001
// CHECK-INST: fmls    za.s[w8, 1, vgx4], { z0.s - z3.s }, z14.s[1]
// CHECK-ENCODING: [0x11,0x84,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e8411 <unknown>

fmls    za.s[w10, 0, vgx4], {z16.s - z19.s}, z4.s[1]  // 11000001-01010100-11000110-00010000
// CHECK-INST: fmls    za.s[w10, 0, vgx4], { z16.s - z19.s }, z4.s[1]
// CHECK-ENCODING: [0x10,0xc6,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c154c610 <unknown>

fmls    za.s[w10, 0], {z16.s - z19.s}, z4.s[1]  // 11000001-01010100-11000110-00010000
// CHECK-INST: fmls    za.s[w10, 0, vgx4], { z16.s - z19.s }, z4.s[1]
// CHECK-ENCODING: [0x10,0xc6,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c154c610 <unknown>

fmls    za.s[w8, 0, vgx4], {z12.s - z15.s}, z2.s[2]  // 11000001-01010010-10001001-10010000
// CHECK-INST: fmls    za.s[w8, 0, vgx4], { z12.s - z15.s }, z2.s[2]
// CHECK-ENCODING: [0x90,0x89,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1528990 <unknown>

fmls    za.s[w8, 0], {z12.s - z15.s}, z2.s[2]  // 11000001-01010010-10001001-10010000
// CHECK-INST: fmls    za.s[w8, 0, vgx4], { z12.s - z15.s }, z2.s[2]
// CHECK-ENCODING: [0x90,0x89,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1528990 <unknown>

fmls    za.s[w10, 1, vgx4], {z0.s - z3.s}, z10.s[2]  // 11000001-01011010-11001000-00010001
// CHECK-INST: fmls    za.s[w10, 1, vgx4], { z0.s - z3.s }, z10.s[2]
// CHECK-ENCODING: [0x11,0xc8,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15ac811 <unknown>

fmls    za.s[w10, 1], {z0.s - z3.s}, z10.s[2]  // 11000001-01011010-11001000-00010001
// CHECK-INST: fmls    za.s[w10, 1, vgx4], { z0.s - z3.s }, z10.s[2]
// CHECK-ENCODING: [0x11,0xc8,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15ac811 <unknown>

fmls    za.s[w8, 5, vgx4], {z20.s - z23.s}, z14.s[2]  // 11000001-01011110-10001010-10010101
// CHECK-INST: fmls    za.s[w8, 5, vgx4], { z20.s - z23.s }, z14.s[2]
// CHECK-ENCODING: [0x95,0x8a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e8a95 <unknown>

fmls    za.s[w8, 5], {z20.s - z23.s}, z14.s[2]  // 11000001-01011110-10001010-10010101
// CHECK-INST: fmls    za.s[w8, 5, vgx4], { z20.s - z23.s }, z14.s[2]
// CHECK-ENCODING: [0x95,0x8a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e8a95 <unknown>

fmls    za.s[w11, 2, vgx4], {z8.s - z11.s}, z1.s[1]  // 11000001-01010001-11100101-00010010
// CHECK-INST: fmls    za.s[w11, 2, vgx4], { z8.s - z11.s }, z1.s[1]
// CHECK-ENCODING: [0x12,0xe5,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c151e512 <unknown>

fmls    za.s[w11, 2], {z8.s - z11.s}, z1.s[1]  // 11000001-01010001-11100101-00010010
// CHECK-INST: fmls    za.s[w11, 2, vgx4], { z8.s - z11.s }, z1.s[1]
// CHECK-ENCODING: [0x12,0xe5,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c151e512 <unknown>

fmls    za.s[w9, 7, vgx4], {z12.s - z15.s}, z11.s[2]  // 11000001-01011011-10101001-10010111
// CHECK-INST: fmls    za.s[w9, 7, vgx4], { z12.s - z15.s }, z11.s[2]
// CHECK-ENCODING: [0x97,0xa9,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15ba997 <unknown>

fmls    za.s[w9, 7], {z12.s - z15.s}, z11.s[2]  // 11000001-01011011-10101001-10010111
// CHECK-INST: fmls    za.s[w9, 7, vgx4], { z12.s - z15.s }, z11.s[2]
// CHECK-ENCODING: [0x97,0xa9,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15ba997 <unknown>


fmls    za.s[w8, 0, vgx4], {z0.s - z3.s}, {z0.s - z3.s}  // 11000001-10100001-00011000-00001000
// CHECK-INST: fmls    za.s[w8, 0, vgx4], { z0.s - z3.s }, { z0.s - z3.s }
// CHECK-ENCODING: [0x08,0x18,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11808 <unknown>

fmls    za.s[w8, 0], {z0.s - z3.s}, {z0.s - z3.s}  // 11000001-10100001-00011000-00001000
// CHECK-INST: fmls    za.s[w8, 0, vgx4], { z0.s - z3.s }, { z0.s - z3.s }
// CHECK-ENCODING: [0x08,0x18,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11808 <unknown>

fmls    za.s[w10, 5, vgx4], {z8.s - z11.s}, {z20.s - z23.s}  // 11000001-10110101-01011001-00001101
// CHECK-INST: fmls    za.s[w10, 5, vgx4], { z8.s - z11.s }, { z20.s - z23.s }
// CHECK-ENCODING: [0x0d,0x59,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b5590d <unknown>

fmls    za.s[w10, 5], {z8.s - z11.s}, {z20.s - z23.s}  // 11000001-10110101-01011001-00001101
// CHECK-INST: fmls    za.s[w10, 5, vgx4], { z8.s - z11.s }, { z20.s - z23.s }
// CHECK-ENCODING: [0x0d,0x59,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b5590d <unknown>

fmls    za.s[w11, 7, vgx4], {z12.s - z15.s}, {z8.s - z11.s}  // 11000001-10101001-01111001-10001111
// CHECK-INST: fmls    za.s[w11, 7, vgx4], { z12.s - z15.s }, { z8.s - z11.s }
// CHECK-ENCODING: [0x8f,0x79,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a9798f <unknown>

fmls    za.s[w11, 7], {z12.s - z15.s}, {z8.s - z11.s}  // 11000001-10101001-01111001-10001111
// CHECK-INST: fmls    za.s[w11, 7, vgx4], { z12.s - z15.s }, { z8.s - z11.s }
// CHECK-ENCODING: [0x8f,0x79,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a9798f <unknown>

fmls    za.s[w11, 7, vgx4], {z28.s - z31.s}, {z28.s - z31.s}  // 11000001-10111101-01111011-10001111
// CHECK-INST: fmls    za.s[w11, 7, vgx4], { z28.s - z31.s }, { z28.s - z31.s }
// CHECK-ENCODING: [0x8f,0x7b,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd7b8f <unknown>

fmls    za.s[w11, 7], {z28.s - z31.s}, {z28.s - z31.s}  // 11000001-10111101-01111011-10001111
// CHECK-INST: fmls    za.s[w11, 7, vgx4], { z28.s - z31.s }, { z28.s - z31.s }
// CHECK-ENCODING: [0x8f,0x7b,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd7b8f <unknown>

fmls    za.s[w8, 5, vgx4], {z16.s - z19.s}, {z16.s - z19.s}  // 11000001-10110001-00011010-00001101
// CHECK-INST: fmls    za.s[w8, 5, vgx4], { z16.s - z19.s }, { z16.s - z19.s }
// CHECK-ENCODING: [0x0d,0x1a,0xb1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b11a0d <unknown>

fmls    za.s[w8, 5], {z16.s - z19.s}, {z16.s - z19.s}  // 11000001-10110001-00011010-00001101
// CHECK-INST: fmls    za.s[w8, 5, vgx4], { z16.s - z19.s }, { z16.s - z19.s }
// CHECK-ENCODING: [0x0d,0x1a,0xb1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b11a0d <unknown>

fmls    za.s[w8, 1, vgx4], {z0.s - z3.s}, {z28.s - z31.s}  // 11000001-10111101-00011000-00001001
// CHECK-INST: fmls    za.s[w8, 1, vgx4], { z0.s - z3.s }, { z28.s - z31.s }
// CHECK-ENCODING: [0x09,0x18,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd1809 <unknown>

fmls    za.s[w8, 1], {z0.s - z3.s}, {z28.s - z31.s}  // 11000001-10111101-00011000-00001001
// CHECK-INST: fmls    za.s[w8, 1, vgx4], { z0.s - z3.s }, { z28.s - z31.s }
// CHECK-ENCODING: [0x09,0x18,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd1809 <unknown>

fmls    za.s[w10, 0, vgx4], {z16.s - z19.s}, {z20.s - z23.s}  // 11000001-10110101-01011010-00001000
// CHECK-INST: fmls    za.s[w10, 0, vgx4], { z16.s - z19.s }, { z20.s - z23.s }
// CHECK-ENCODING: [0x08,0x5a,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b55a08 <unknown>

fmls    za.s[w10, 0], {z16.s - z19.s}, {z20.s - z23.s}  // 11000001-10110101-01011010-00001000
// CHECK-INST: fmls    za.s[w10, 0, vgx4], { z16.s - z19.s }, { z20.s - z23.s }
// CHECK-ENCODING: [0x08,0x5a,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b55a08 <unknown>

fmls    za.s[w8, 0, vgx4], {z12.s - z15.s}, {z0.s - z3.s}  // 11000001-10100001-00011001-10001000
// CHECK-INST: fmls    za.s[w8, 0, vgx4], { z12.s - z15.s }, { z0.s - z3.s }
// CHECK-ENCODING: [0x88,0x19,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11988 <unknown>

fmls    za.s[w8, 0], {z12.s - z15.s}, {z0.s - z3.s}  // 11000001-10100001-00011001-10001000
// CHECK-INST: fmls    za.s[w8, 0, vgx4], { z12.s - z15.s }, { z0.s - z3.s }
// CHECK-ENCODING: [0x88,0x19,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11988 <unknown>

fmls    za.s[w10, 1, vgx4], {z0.s - z3.s}, {z24.s - z27.s}  // 11000001-10111001-01011000-00001001
// CHECK-INST: fmls    za.s[w10, 1, vgx4], { z0.s - z3.s }, { z24.s - z27.s }
// CHECK-ENCODING: [0x09,0x58,0xb9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b95809 <unknown>

fmls    za.s[w10, 1], {z0.s - z3.s}, {z24.s - z27.s}  // 11000001-10111001-01011000-00001001
// CHECK-INST: fmls    za.s[w10, 1, vgx4], { z0.s - z3.s }, { z24.s - z27.s }
// CHECK-ENCODING: [0x09,0x58,0xb9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b95809 <unknown>

fmls    za.s[w8, 5, vgx4], {z20.s - z23.s}, {z28.s - z31.s}  // 11000001-10111101-00011010-10001101
// CHECK-INST: fmls    za.s[w8, 5, vgx4], { z20.s - z23.s }, { z28.s - z31.s }
// CHECK-ENCODING: [0x8d,0x1a,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd1a8d <unknown>

fmls    za.s[w8, 5], {z20.s - z23.s}, {z28.s - z31.s}  // 11000001-10111101-00011010-10001101
// CHECK-INST: fmls    za.s[w8, 5, vgx4], { z20.s - z23.s }, { z28.s - z31.s }
// CHECK-ENCODING: [0x8d,0x1a,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd1a8d <unknown>

fmls    za.s[w11, 2, vgx4], {z8.s - z11.s}, {z0.s - z3.s}  // 11000001-10100001-01111001-00001010
// CHECK-INST: fmls    za.s[w11, 2, vgx4], { z8.s - z11.s }, { z0.s - z3.s }
// CHECK-ENCODING: [0x0a,0x79,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a1790a <unknown>

fmls    za.s[w11, 2], {z8.s - z11.s}, {z0.s - z3.s}  // 11000001-10100001-01111001-00001010
// CHECK-INST: fmls    za.s[w11, 2, vgx4], { z8.s - z11.s }, { z0.s - z3.s }
// CHECK-ENCODING: [0x0a,0x79,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a1790a <unknown>

fmls    za.s[w9, 7, vgx4], {z12.s - z15.s}, {z8.s - z11.s}  // 11000001-10101001-00111001-10001111
// CHECK-INST: fmls    za.s[w9, 7, vgx4], { z12.s - z15.s }, { z8.s - z11.s }
// CHECK-ENCODING: [0x8f,0x39,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a9398f <unknown>

fmls    za.s[w9, 7], {z12.s - z15.s}, {z8.s - z11.s}  // 11000001-10101001-00111001-10001111
// CHECK-INST: fmls    za.s[w9, 7, vgx4], { z12.s - z15.s }, { z8.s - z11.s }
// CHECK-ENCODING: [0x8f,0x39,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a9398f <unknown>

