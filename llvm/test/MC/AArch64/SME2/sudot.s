// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


sudot   za.s[w8, 0, vgx2], {z0.b, z1.b}, z0.b  // 11000001-00100000-00010100-00011000
// CHECK-INST: sudot   za.s[w8, 0, vgx2], { z0.b, z1.b }, z0.b
// CHECK-ENCODING: [0x18,0x14,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1201418 <unknown>

sudot   za.s[w8, 0], {z0.b, z1.b}, z0.b  // 11000001-00100000-00010100-00011000
// CHECK-INST: sudot   za.s[w8, 0, vgx2], { z0.b, z1.b }, z0.b
// CHECK-ENCODING: [0x18,0x14,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1201418 <unknown>

sudot   za.s[w10, 5, vgx2], {z10.b, z11.b}, z5.b  // 11000001-00100101-01010101-01011101
// CHECK-INST: sudot   za.s[w10, 5, vgx2], { z10.b, z11.b }, z5.b
// CHECK-ENCODING: [0x5d,0x55,0x25,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c125555d <unknown>

sudot   za.s[w10, 5], {z10.b, z11.b}, z5.b  // 11000001-00100101-01010101-01011101
// CHECK-INST: sudot   za.s[w10, 5, vgx2], { z10.b, z11.b }, z5.b
// CHECK-ENCODING: [0x5d,0x55,0x25,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c125555d <unknown>

sudot   za.s[w11, 7, vgx2], {z13.b, z14.b}, z8.b  // 11000001-00101000-01110101-10111111
// CHECK-INST: sudot   za.s[w11, 7, vgx2], { z13.b, z14.b }, z8.b
// CHECK-ENCODING: [0xbf,0x75,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12875bf <unknown>

sudot   za.s[w11, 7], {z13.b, z14.b}, z8.b  // 11000001-00101000-01110101-10111111
// CHECK-INST: sudot   za.s[w11, 7, vgx2], { z13.b, z14.b }, z8.b
// CHECK-ENCODING: [0xbf,0x75,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12875bf <unknown>

sudot   za.s[w11, 7, vgx2], {z31.b, z0.b}, z15.b  // 11000001-00101111-01110111-11111111
// CHECK-INST: sudot   za.s[w11, 7, vgx2], { z31.b, z0.b }, z15.b
// CHECK-ENCODING: [0xff,0x77,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12f77ff <unknown>

sudot   za.s[w11, 7], {z31.b, z0.b}, z15.b  // 11000001-00101111-01110111-11111111
// CHECK-INST: sudot   za.s[w11, 7, vgx2], { z31.b, z0.b }, z15.b
// CHECK-ENCODING: [0xff,0x77,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12f77ff <unknown>

sudot   za.s[w8, 5, vgx2], {z17.b, z18.b}, z0.b  // 11000001-00100000-00010110-00111101
// CHECK-INST: sudot   za.s[w8, 5, vgx2], { z17.b, z18.b }, z0.b
// CHECK-ENCODING: [0x3d,0x16,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c120163d <unknown>

sudot   za.s[w8, 5], {z17.b, z18.b}, z0.b  // 11000001-00100000-00010110-00111101
// CHECK-INST: sudot   za.s[w8, 5, vgx2], { z17.b, z18.b }, z0.b
// CHECK-ENCODING: [0x3d,0x16,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c120163d <unknown>

sudot   za.s[w8, 1, vgx2], {z1.b, z2.b}, z14.b  // 11000001-00101110-00010100-00111001
// CHECK-INST: sudot   za.s[w8, 1, vgx2], { z1.b, z2.b }, z14.b
// CHECK-ENCODING: [0x39,0x14,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e1439 <unknown>

sudot   za.s[w8, 1], {z1.b, z2.b}, z14.b  // 11000001-00101110-00010100-00111001
// CHECK-INST: sudot   za.s[w8, 1, vgx2], { z1.b, z2.b }, z14.b
// CHECK-ENCODING: [0x39,0x14,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e1439 <unknown>


sudot   za.s[w10, 0, vgx2], {z19.b, z20.b}, z4.b  // 11000001-00100100-01010110-01111000
// CHECK-INST: sudot   za.s[w10, 0, vgx2], { z19.b, z20.b }, z4.b
// CHECK-ENCODING: [0x78,0x56,0x24,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1245678 <unknown>

sudot   za.s[w10, 0], {z19.b, z20.b}, z4.b  // 11000001-00100100-01010110-01111000
// CHECK-INST: sudot   za.s[w10, 0, vgx2], { z19.b, z20.b }, z4.b
// CHECK-ENCODING: [0x78,0x56,0x24,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1245678 <unknown>

sudot   za.s[w8, 0, vgx2], {z12.b, z13.b}, z2.b  // 11000001-00100010-00010101-10011000
// CHECK-INST: sudot   za.s[w8, 0, vgx2], { z12.b, z13.b }, z2.b
// CHECK-ENCODING: [0x98,0x15,0x22,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1221598 <unknown>

sudot   za.s[w8, 0], {z12.b, z13.b}, z2.b  // 11000001-00100010-00010101-10011000
// CHECK-INST: sudot   za.s[w8, 0, vgx2], { z12.b, z13.b }, z2.b
// CHECK-ENCODING: [0x98,0x15,0x22,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1221598 <unknown>

sudot   za.s[w10, 1, vgx2], {z1.b, z2.b}, z10.b  // 11000001-00101010-01010100-00111001
// CHECK-INST: sudot   za.s[w10, 1, vgx2], { z1.b, z2.b }, z10.b
// CHECK-ENCODING: [0x39,0x54,0x2a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12a5439 <unknown>

sudot   za.s[w10, 1], {z1.b, z2.b}, z10.b  // 11000001-00101010-01010100-00111001
// CHECK-INST: sudot   za.s[w10, 1, vgx2], { z1.b, z2.b }, z10.b
// CHECK-ENCODING: [0x39,0x54,0x2a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12a5439 <unknown>

sudot   za.s[w8, 5, vgx2], {z22.b, z23.b}, z14.b  // 11000001-00101110-00010110-11011101
// CHECK-INST: sudot   za.s[w8, 5, vgx2], { z22.b, z23.b }, z14.b
// CHECK-ENCODING: [0xdd,0x16,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e16dd <unknown>

sudot   za.s[w8, 5], {z22.b, z23.b}, z14.b  // 11000001-00101110-00010110-11011101
// CHECK-INST: sudot   za.s[w8, 5, vgx2], { z22.b, z23.b }, z14.b
// CHECK-ENCODING: [0xdd,0x16,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e16dd <unknown>

sudot   za.s[w11, 2, vgx2], {z9.b, z10.b}, z1.b  // 11000001-00100001-01110101-00111010
// CHECK-INST: sudot   za.s[w11, 2, vgx2], { z9.b, z10.b }, z1.b
// CHECK-ENCODING: [0x3a,0x75,0x21,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c121753a <unknown>

sudot   za.s[w11, 2], {z9.b, z10.b}, z1.b  // 11000001-00100001-01110101-00111010
// CHECK-INST: sudot   za.s[w11, 2, vgx2], { z9.b, z10.b }, z1.b
// CHECK-ENCODING: [0x3a,0x75,0x21,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c121753a <unknown>

sudot   za.s[w9, 7, vgx2], {z12.b, z13.b}, z11.b  // 11000001-00101011-00110101-10011111
// CHECK-INST: sudot   za.s[w9, 7, vgx2], { z12.b, z13.b }, z11.b
// CHECK-ENCODING: [0x9f,0x35,0x2b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12b359f <unknown>

sudot   za.s[w9, 7], {z12.b, z13.b}, z11.b  // 11000001-00101011-00110101-10011111
// CHECK-INST: sudot   za.s[w9, 7, vgx2], { z12.b, z13.b }, z11.b
// CHECK-ENCODING: [0x9f,0x35,0x2b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12b359f <unknown>


sudot   za.s[w8, 0, vgx2], {z0.b, z1.b}, z0.b[0]  // 11000001-01010000-00010000-00111000
// CHECK-INST: sudot   za.s[w8, 0, vgx2], { z0.b, z1.b }, z0.b[0]
// CHECK-ENCODING: [0x38,0x10,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1501038 <unknown>

sudot   za.s[w8, 0], {z0.b, z1.b}, z0.b[0]  // 11000001-01010000-00010000-00111000
// CHECK-INST: sudot   za.s[w8, 0, vgx2], { z0.b, z1.b }, z0.b[0]
// CHECK-ENCODING: [0x38,0x10,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1501038 <unknown>

sudot   za.s[w10, 5, vgx2], {z10.b, z11.b}, z5.b[1]  // 11000001-01010101-01010101-01111101
// CHECK-INST: sudot   za.s[w10, 5, vgx2], { z10.b, z11.b }, z5.b[1]
// CHECK-ENCODING: [0x7d,0x55,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c155557d <unknown>

sudot   za.s[w10, 5], {z10.b, z11.b}, z5.b[1]  // 11000001-01010101-01010101-01111101
// CHECK-INST: sudot   za.s[w10, 5, vgx2], { z10.b, z11.b }, z5.b[1]
// CHECK-ENCODING: [0x7d,0x55,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c155557d <unknown>

sudot   za.s[w11, 7, vgx2], {z12.b, z13.b}, z8.b[3]  // 11000001-01011000-01111101-10111111
// CHECK-INST: sudot   za.s[w11, 7, vgx2], { z12.b, z13.b }, z8.b[3]
// CHECK-ENCODING: [0xbf,0x7d,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1587dbf <unknown>

sudot   za.s[w11, 7], {z12.b, z13.b}, z8.b[3]  // 11000001-01011000-01111101-10111111
// CHECK-INST: sudot   za.s[w11, 7, vgx2], { z12.b, z13.b }, z8.b[3]
// CHECK-ENCODING: [0xbf,0x7d,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1587dbf <unknown>

sudot   za.s[w11, 7, vgx2], {z30.b, z31.b}, z15.b[3]  // 11000001-01011111-01111111-11111111
// CHECK-INST: sudot   za.s[w11, 7, vgx2], { z30.b, z31.b }, z15.b[3]
// CHECK-ENCODING: [0xff,0x7f,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15f7fff <unknown>

sudot   za.s[w11, 7], {z30.b, z31.b}, z15.b[3]  // 11000001-01011111-01111111-11111111
// CHECK-INST: sudot   za.s[w11, 7, vgx2], { z30.b, z31.b }, z15.b[3]
// CHECK-ENCODING: [0xff,0x7f,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15f7fff <unknown>

sudot   za.s[w8, 5, vgx2], {z16.b, z17.b}, z0.b[3]  // 11000001-01010000-00011110-00111101
// CHECK-INST: sudot   za.s[w8, 5, vgx2], { z16.b, z17.b }, z0.b[3]
// CHECK-ENCODING: [0x3d,0x1e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1501e3d <unknown>

sudot   za.s[w8, 5], {z16.b, z17.b}, z0.b[3]  // 11000001-01010000-00011110-00111101
// CHECK-INST: sudot   za.s[w8, 5, vgx2], { z16.b, z17.b }, z0.b[3]
// CHECK-ENCODING: [0x3d,0x1e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1501e3d <unknown>

sudot   za.s[w8, 1, vgx2], {z0.b, z1.b}, z14.b[1]  // 11000001-01011110-00010100-00111001
// CHECK-INST: sudot   za.s[w8, 1, vgx2], { z0.b, z1.b }, z14.b[1]
// CHECK-ENCODING: [0x39,0x14,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e1439 <unknown>

sudot   za.s[w8, 1], {z0.b, z1.b}, z14.b[1]  // 11000001-01011110-00010100-00111001
// CHECK-INST: sudot   za.s[w8, 1, vgx2], { z0.b, z1.b }, z14.b[1]
// CHECK-ENCODING: [0x39,0x14,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e1439 <unknown>

sudot   za.s[w10, 0, vgx2], {z18.b, z19.b}, z4.b[1]  // 11000001-01010100-01010110-01111000
// CHECK-INST: sudot   za.s[w10, 0, vgx2], { z18.b, z19.b }, z4.b[1]
// CHECK-ENCODING: [0x78,0x56,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1545678 <unknown>

sudot   za.s[w10, 0], {z18.b, z19.b}, z4.b[1]  // 11000001-01010100-01010110-01111000
// CHECK-INST: sudot   za.s[w10, 0, vgx2], { z18.b, z19.b }, z4.b[1]
// CHECK-ENCODING: [0x78,0x56,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1545678 <unknown>

sudot   za.s[w8, 0, vgx2], {z12.b, z13.b}, z2.b[2]  // 11000001-01010010-00011001-10111000
// CHECK-INST: sudot   za.s[w8, 0, vgx2], { z12.b, z13.b }, z2.b[2]
// CHECK-ENCODING: [0xb8,0x19,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15219b8 <unknown>

sudot   za.s[w8, 0], {z12.b, z13.b}, z2.b[2]  // 11000001-01010010-00011001-10111000
// CHECK-INST: sudot   za.s[w8, 0, vgx2], { z12.b, z13.b }, z2.b[2]
// CHECK-ENCODING: [0xb8,0x19,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15219b8 <unknown>

sudot   za.s[w10, 1, vgx2], {z0.b, z1.b}, z10.b[2]  // 11000001-01011010-01011000-00111001
// CHECK-INST: sudot   za.s[w10, 1, vgx2], { z0.b, z1.b }, z10.b[2]
// CHECK-ENCODING: [0x39,0x58,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15a5839 <unknown>

sudot   za.s[w10, 1], {z0.b, z1.b}, z10.b[2]  // 11000001-01011010-01011000-00111001
// CHECK-INST: sudot   za.s[w10, 1, vgx2], { z0.b, z1.b }, z10.b[2]
// CHECK-ENCODING: [0x39,0x58,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15a5839 <unknown>

sudot   za.s[w8, 5, vgx2], {z22.b, z23.b}, z14.b[2]  // 11000001-01011110-00011010-11111101
// CHECK-INST: sudot   za.s[w8, 5, vgx2], { z22.b, z23.b }, z14.b[2]
// CHECK-ENCODING: [0xfd,0x1a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e1afd <unknown>

sudot   za.s[w8, 5], {z22.b, z23.b}, z14.b[2]  // 11000001-01011110-00011010-11111101
// CHECK-INST: sudot   za.s[w8, 5, vgx2], { z22.b, z23.b }, z14.b[2]
// CHECK-ENCODING: [0xfd,0x1a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e1afd <unknown>

sudot   za.s[w11, 2, vgx2], {z8.b, z9.b}, z1.b[1]  // 11000001-01010001-01110101-00111010
// CHECK-INST: sudot   za.s[w11, 2, vgx2], { z8.b, z9.b }, z1.b[1]
// CHECK-ENCODING: [0x3a,0x75,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c151753a <unknown>

sudot   za.s[w11, 2], {z8.b, z9.b}, z1.b[1]  // 11000001-01010001-01110101-00111010
// CHECK-INST: sudot   za.s[w11, 2, vgx2], { z8.b, z9.b }, z1.b[1]
// CHECK-ENCODING: [0x3a,0x75,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c151753a <unknown>

sudot   za.s[w9, 7, vgx2], {z12.b, z13.b}, z11.b[2]  // 11000001-01011011-00111001-10111111
// CHECK-INST: sudot   za.s[w9, 7, vgx2], { z12.b, z13.b }, z11.b[2]
// CHECK-ENCODING: [0xbf,0x39,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15b39bf <unknown>

sudot   za.s[w9, 7], {z12.b, z13.b}, z11.b[2]  // 11000001-01011011-00111001-10111111
// CHECK-INST: sudot   za.s[w9, 7, vgx2], { z12.b, z13.b }, z11.b[2]
// CHECK-ENCODING: [0xbf,0x39,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15b39bf <unknown>

sudot   za.s[w8, 0, vgx4], {z0.b - z3.b}, z0.b  // 11000001-00110000-00010100-00011000
// CHECK-INST: sudot   za.s[w8, 0, vgx4], { z0.b - z3.b }, z0.b
// CHECK-ENCODING: [0x18,0x14,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1301418 <unknown>

sudot   za.s[w8, 0], {z0.b - z3.b}, z0.b  // 11000001-00110000-00010100-00011000
// CHECK-INST: sudot   za.s[w8, 0, vgx4], { z0.b - z3.b }, z0.b
// CHECK-ENCODING: [0x18,0x14,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1301418 <unknown>

sudot   za.s[w10, 5, vgx4], {z10.b - z13.b}, z5.b  // 11000001-00110101-01010101-01011101
// CHECK-INST: sudot   za.s[w10, 5, vgx4], { z10.b - z13.b }, z5.b
// CHECK-ENCODING: [0x5d,0x55,0x35,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c135555d <unknown>

sudot   za.s[w10, 5], {z10.b - z13.b}, z5.b  // 11000001-00110101-01010101-01011101
// CHECK-INST: sudot   za.s[w10, 5, vgx4], { z10.b - z13.b }, z5.b
// CHECK-ENCODING: [0x5d,0x55,0x35,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c135555d <unknown>

sudot   za.s[w11, 7, vgx4], {z13.b - z16.b}, z8.b  // 11000001-00111000-01110101-10111111
// CHECK-INST: sudot   za.s[w11, 7, vgx4], { z13.b - z16.b }, z8.b
// CHECK-ENCODING: [0xbf,0x75,0x38,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13875bf <unknown>

sudot   za.s[w11, 7], {z13.b - z16.b}, z8.b  // 11000001-00111000-01110101-10111111
// CHECK-INST: sudot   za.s[w11, 7, vgx4], { z13.b - z16.b }, z8.b
// CHECK-ENCODING: [0xbf,0x75,0x38,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13875bf <unknown>

sudot   za.s[w11, 7, vgx4], {z31.b - z2.b}, z15.b  // 11000001-00111111-01110111-11111111
// CHECK-INST: sudot   za.s[w11, 7, vgx4], { z31.b, z0.b, z1.b, z2.b }, z15.b
// CHECK-ENCODING: [0xff,0x77,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13f77ff <unknown>

sudot   za.s[w11, 7], {z31.b - z2.b}, z15.b  // 11000001-00111111-01110111-11111111
// CHECK-INST: sudot   za.s[w11, 7, vgx4], { z31.b, z0.b, z1.b, z2.b }, z15.b
// CHECK-ENCODING: [0xff,0x77,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13f77ff <unknown>

sudot   za.s[w8, 5, vgx4], {z17.b - z20.b}, z0.b  // 11000001-00110000-00010110-00111101
// CHECK-INST: sudot   za.s[w8, 5, vgx4], { z17.b - z20.b }, z0.b
// CHECK-ENCODING: [0x3d,0x16,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c130163d <unknown>

sudot   za.s[w8, 5], {z17.b - z20.b}, z0.b  // 11000001-00110000-00010110-00111101
// CHECK-INST: sudot   za.s[w8, 5, vgx4], { z17.b - z20.b }, z0.b
// CHECK-ENCODING: [0x3d,0x16,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c130163d <unknown>

sudot   za.s[w8, 1, vgx4], {z1.b - z4.b}, z14.b  // 11000001-00111110-00010100-00111001
// CHECK-INST: sudot   za.s[w8, 1, vgx4], { z1.b - z4.b }, z14.b
// CHECK-ENCODING: [0x39,0x14,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e1439 <unknown>

sudot   za.s[w8, 1], {z1.b - z4.b}, z14.b  // 11000001-00111110-00010100-00111001
// CHECK-INST: sudot   za.s[w8, 1, vgx4], { z1.b - z4.b }, z14.b
// CHECK-ENCODING: [0x39,0x14,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e1439 <unknown>

sudot   za.s[w10, 0, vgx4], {z19.b - z22.b}, z4.b  // 11000001-00110100-01010110-01111000
// CHECK-INST: sudot   za.s[w10, 0, vgx4], { z19.b - z22.b }, z4.b
// CHECK-ENCODING: [0x78,0x56,0x34,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1345678 <unknown>

sudot   za.s[w10, 0], {z19.b - z22.b}, z4.b  // 11000001-00110100-01010110-01111000
// CHECK-INST: sudot   za.s[w10, 0, vgx4], { z19.b - z22.b }, z4.b
// CHECK-ENCODING: [0x78,0x56,0x34,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1345678 <unknown>

sudot   za.s[w8, 0, vgx4], {z12.b - z15.b}, z2.b  // 11000001-00110010-00010101-10011000
// CHECK-INST: sudot   za.s[w8, 0, vgx4], { z12.b - z15.b }, z2.b
// CHECK-ENCODING: [0x98,0x15,0x32,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1321598 <unknown>

sudot   za.s[w8, 0], {z12.b - z15.b}, z2.b  // 11000001-00110010-00010101-10011000
// CHECK-INST: sudot   za.s[w8, 0, vgx4], { z12.b - z15.b }, z2.b
// CHECK-ENCODING: [0x98,0x15,0x32,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1321598 <unknown>

sudot   za.s[w10, 1, vgx4], {z1.b - z4.b}, z10.b  // 11000001-00111010-01010100-00111001
// CHECK-INST: sudot   za.s[w10, 1, vgx4], { z1.b - z4.b }, z10.b
// CHECK-ENCODING: [0x39,0x54,0x3a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13a5439 <unknown>

sudot   za.s[w10, 1], {z1.b - z4.b}, z10.b  // 11000001-00111010-01010100-00111001
// CHECK-INST: sudot   za.s[w10, 1, vgx4], { z1.b - z4.b }, z10.b
// CHECK-ENCODING: [0x39,0x54,0x3a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13a5439 <unknown>

sudot   za.s[w8, 5, vgx4], {z22.b - z25.b}, z14.b  // 11000001-00111110-00010110-11011101
// CHECK-INST: sudot   za.s[w8, 5, vgx4], { z22.b - z25.b }, z14.b
// CHECK-ENCODING: [0xdd,0x16,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e16dd <unknown>

sudot   za.s[w8, 5], {z22.b - z25.b}, z14.b  // 11000001-00111110-00010110-11011101
// CHECK-INST: sudot   za.s[w8, 5, vgx4], { z22.b - z25.b }, z14.b
// CHECK-ENCODING: [0xdd,0x16,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e16dd <unknown>

sudot   za.s[w11, 2, vgx4], {z9.b - z12.b}, z1.b  // 11000001-00110001-01110101-00111010
// CHECK-INST: sudot   za.s[w11, 2, vgx4], { z9.b - z12.b }, z1.b
// CHECK-ENCODING: [0x3a,0x75,0x31,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c131753a <unknown>

sudot   za.s[w11, 2], {z9.b - z12.b}, z1.b  // 11000001-00110001-01110101-00111010
// CHECK-INST: sudot   za.s[w11, 2, vgx4], { z9.b - z12.b }, z1.b
// CHECK-ENCODING: [0x3a,0x75,0x31,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c131753a <unknown>

sudot   za.s[w9, 7, vgx4], {z12.b - z15.b}, z11.b  // 11000001-00111011-00110101-10011111
// CHECK-INST: sudot   za.s[w9, 7, vgx4], { z12.b - z15.b }, z11.b
// CHECK-ENCODING: [0x9f,0x35,0x3b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13b359f <unknown>

sudot   za.s[w9, 7], {z12.b - z15.b}, z11.b  // 11000001-00111011-00110101-10011111
// CHECK-INST: sudot   za.s[w9, 7, vgx4], { z12.b - z15.b }, z11.b
// CHECK-ENCODING: [0x9f,0x35,0x3b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13b359f <unknown>


sudot   za.s[w8, 0, vgx4], {z0.b - z3.b}, z0.b[0]  // 11000001-01010000-10010000-00111000
// CHECK-INST: sudot   za.s[w8, 0, vgx4], { z0.b - z3.b }, z0.b[0]
// CHECK-ENCODING: [0x38,0x90,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1509038 <unknown>

sudot   za.s[w8, 0], {z0.b - z3.b}, z0.b[0]  // 11000001-01010000-10010000-00111000
// CHECK-INST: sudot   za.s[w8, 0, vgx4], { z0.b - z3.b }, z0.b[0]
// CHECK-ENCODING: [0x38,0x90,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1509038 <unknown>

sudot   za.s[w10, 5, vgx4], {z8.b - z11.b}, z5.b[1]  // 11000001-01010101-11010101-00111101
// CHECK-INST: sudot   za.s[w10, 5, vgx4], { z8.b - z11.b }, z5.b[1]
// CHECK-ENCODING: [0x3d,0xd5,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c155d53d <unknown>

sudot   za.s[w10, 5], {z8.b - z11.b}, z5.b[1]  // 11000001-01010101-11010101-00111101
// CHECK-INST: sudot   za.s[w10, 5, vgx4], { z8.b - z11.b }, z5.b[1]
// CHECK-ENCODING: [0x3d,0xd5,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c155d53d <unknown>

sudot   za.s[w11, 7, vgx4], {z12.b - z15.b}, z8.b[3]  // 11000001-01011000-11111101-10111111
// CHECK-INST: sudot   za.s[w11, 7, vgx4], { z12.b - z15.b }, z8.b[3]
// CHECK-ENCODING: [0xbf,0xfd,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c158fdbf <unknown>

sudot   za.s[w11, 7], {z12.b - z15.b}, z8.b[3]  // 11000001-01011000-11111101-10111111
// CHECK-INST: sudot   za.s[w11, 7, vgx4], { z12.b - z15.b }, z8.b[3]
// CHECK-ENCODING: [0xbf,0xfd,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c158fdbf <unknown>

sudot   za.s[w11, 7, vgx4], {z28.b - z31.b}, z15.b[3]  // 11000001-01011111-11111111-10111111
// CHECK-INST: sudot   za.s[w11, 7, vgx4], { z28.b - z31.b }, z15.b[3]
// CHECK-ENCODING: [0xbf,0xff,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15fffbf <unknown>

sudot   za.s[w11, 7], {z28.b - z31.b}, z15.b[3]  // 11000001-01011111-11111111-10111111
// CHECK-INST: sudot   za.s[w11, 7, vgx4], { z28.b - z31.b }, z15.b[3]
// CHECK-ENCODING: [0xbf,0xff,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15fffbf <unknown>

sudot   za.s[w8, 5, vgx4], {z16.b - z19.b}, z0.b[3]  // 11000001-01010000-10011110-00111101
// CHECK-INST: sudot   za.s[w8, 5, vgx4], { z16.b - z19.b }, z0.b[3]
// CHECK-ENCODING: [0x3d,0x9e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1509e3d <unknown>

sudot   za.s[w8, 5], {z16.b - z19.b}, z0.b[3]  // 11000001-01010000-10011110-00111101
// CHECK-INST: sudot   za.s[w8, 5, vgx4], { z16.b - z19.b }, z0.b[3]
// CHECK-ENCODING: [0x3d,0x9e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1509e3d <unknown>

sudot   za.s[w8, 1, vgx4], {z0.b - z3.b}, z14.b[1]  // 11000001-01011110-10010100-00111001
// CHECK-INST: sudot   za.s[w8, 1, vgx4], { z0.b - z3.b }, z14.b[1]
// CHECK-ENCODING: [0x39,0x94,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e9439 <unknown>

sudot   za.s[w8, 1], {z0.b - z3.b}, z14.b[1]  // 11000001-01011110-10010100-00111001
// CHECK-INST: sudot   za.s[w8, 1, vgx4], { z0.b - z3.b }, z14.b[1]
// CHECK-ENCODING: [0x39,0x94,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e9439 <unknown>

sudot   za.s[w10, 0, vgx4], {z16.b - z19.b}, z4.b[1]  // 11000001-01010100-11010110-00111000
// CHECK-INST: sudot   za.s[w10, 0, vgx4], { z16.b - z19.b }, z4.b[1]
// CHECK-ENCODING: [0x38,0xd6,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c154d638 <unknown>

sudot   za.s[w10, 0], {z16.b - z19.b}, z4.b[1]  // 11000001-01010100-11010110-00111000
// CHECK-INST: sudot   za.s[w10, 0, vgx4], { z16.b - z19.b }, z4.b[1]
// CHECK-ENCODING: [0x38,0xd6,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c154d638 <unknown>

sudot   za.s[w8, 0, vgx4], {z12.b - z15.b}, z2.b[2]  // 11000001-01010010-10011001-10111000
// CHECK-INST: sudot   za.s[w8, 0, vgx4], { z12.b - z15.b }, z2.b[2]
// CHECK-ENCODING: [0xb8,0x99,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15299b8 <unknown>

sudot   za.s[w8, 0], {z12.b - z15.b}, z2.b[2]  // 11000001-01010010-10011001-10111000
// CHECK-INST: sudot   za.s[w8, 0, vgx4], { z12.b - z15.b }, z2.b[2]
// CHECK-ENCODING: [0xb8,0x99,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15299b8 <unknown>

sudot   za.s[w10, 1, vgx4], {z0.b - z3.b}, z10.b[2]  // 11000001-01011010-11011000-00111001
// CHECK-INST: sudot   za.s[w10, 1, vgx4], { z0.b - z3.b }, z10.b[2]
// CHECK-ENCODING: [0x39,0xd8,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15ad839 <unknown>

sudot   za.s[w10, 1], {z0.b - z3.b}, z10.b[2]  // 11000001-01011010-11011000-00111001
// CHECK-INST: sudot   za.s[w10, 1, vgx4], { z0.b - z3.b }, z10.b[2]
// CHECK-ENCODING: [0x39,0xd8,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15ad839 <unknown>

sudot   za.s[w8, 5, vgx4], {z20.b - z23.b}, z14.b[2]  // 11000001-01011110-10011010-10111101
// CHECK-INST: sudot   za.s[w8, 5, vgx4], { z20.b - z23.b }, z14.b[2]
// CHECK-ENCODING: [0xbd,0x9a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e9abd <unknown>

sudot   za.s[w8, 5], {z20.b - z23.b}, z14.b[2]  // 11000001-01011110-10011010-10111101
// CHECK-INST: sudot   za.s[w8, 5, vgx4], { z20.b - z23.b }, z14.b[2]
// CHECK-ENCODING: [0xbd,0x9a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e9abd <unknown>

sudot   za.s[w11, 2, vgx4], {z8.b - z11.b}, z1.b[1]  // 11000001-01010001-11110101-00111010
// CHECK-INST: sudot   za.s[w11, 2, vgx4], { z8.b - z11.b }, z1.b[1]
// CHECK-ENCODING: [0x3a,0xf5,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c151f53a <unknown>

sudot   za.s[w11, 2], {z8.b - z11.b}, z1.b[1]  // 11000001-01010001-11110101-00111010
// CHECK-INST: sudot   za.s[w11, 2, vgx4], { z8.b - z11.b }, z1.b[1]
// CHECK-ENCODING: [0x3a,0xf5,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c151f53a <unknown>

sudot   za.s[w9, 7, vgx4], {z12.b - z15.b}, z11.b[2]  // 11000001-01011011-10111001-10111111
// CHECK-INST: sudot   za.s[w9, 7, vgx4], { z12.b - z15.b }, z11.b[2]
// CHECK-ENCODING: [0xbf,0xb9,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15bb9bf <unknown>

sudot   za.s[w9, 7], {z12.b - z15.b}, z11.b[2]  // 11000001-01011011-10111001-10111111
// CHECK-INST: sudot   za.s[w9, 7, vgx4], { z12.b - z15.b }, z11.b[2]
// CHECK-ENCODING: [0xbf,0xb9,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15bb9bf <unknown>

