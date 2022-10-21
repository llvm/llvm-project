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


usdot   za.s[w8, 0, vgx2], {z0.b, z1.b}, z0.b[0]  // 11000001-01010000-00010000-00101000
// CHECK-INST: usdot   za.s[w8, 0, vgx2], { z0.b, z1.b }, z0.b[0]
// CHECK-ENCODING: [0x28,0x10,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1501028 <unknown>

usdot   za.s[w8, 0], {z0.b, z1.b}, z0.b[0]  // 11000001-01010000-00010000-00101000
// CHECK-INST: usdot   za.s[w8, 0, vgx2], { z0.b, z1.b }, z0.b[0]
// CHECK-ENCODING: [0x28,0x10,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1501028 <unknown>

usdot   za.s[w10, 5, vgx2], {z10.b, z11.b}, z5.b[1]  // 11000001-01010101-01010101-01101101
// CHECK-INST: usdot   za.s[w10, 5, vgx2], { z10.b, z11.b }, z5.b[1]
// CHECK-ENCODING: [0x6d,0x55,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c155556d <unknown>

usdot   za.s[w10, 5], {z10.b, z11.b}, z5.b[1]  // 11000001-01010101-01010101-01101101
// CHECK-INST: usdot   za.s[w10, 5, vgx2], { z10.b, z11.b }, z5.b[1]
// CHECK-ENCODING: [0x6d,0x55,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c155556d <unknown>

usdot   za.s[w11, 7, vgx2], {z12.b, z13.b}, z8.b[3]  // 11000001-01011000-01111101-10101111
// CHECK-INST: usdot   za.s[w11, 7, vgx2], { z12.b, z13.b }, z8.b[3]
// CHECK-ENCODING: [0xaf,0x7d,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1587daf <unknown>

usdot   za.s[w11, 7], {z12.b, z13.b}, z8.b[3]  // 11000001-01011000-01111101-10101111
// CHECK-INST: usdot   za.s[w11, 7, vgx2], { z12.b, z13.b }, z8.b[3]
// CHECK-ENCODING: [0xaf,0x7d,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1587daf <unknown>

usdot   za.s[w11, 7, vgx2], {z30.b, z31.b}, z15.b[3]  // 11000001-01011111-01111111-11101111
// CHECK-INST: usdot   za.s[w11, 7, vgx2], { z30.b, z31.b }, z15.b[3]
// CHECK-ENCODING: [0xef,0x7f,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15f7fef <unknown>

usdot   za.s[w11, 7], {z30.b, z31.b}, z15.b[3]  // 11000001-01011111-01111111-11101111
// CHECK-INST: usdot   za.s[w11, 7, vgx2], { z30.b, z31.b }, z15.b[3]
// CHECK-ENCODING: [0xef,0x7f,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15f7fef <unknown>

usdot   za.s[w8, 5, vgx2], {z16.b, z17.b}, z0.b[3]  // 11000001-01010000-00011110-00101101
// CHECK-INST: usdot   za.s[w8, 5, vgx2], { z16.b, z17.b }, z0.b[3]
// CHECK-ENCODING: [0x2d,0x1e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1501e2d <unknown>

usdot   za.s[w8, 5], {z16.b, z17.b}, z0.b[3]  // 11000001-01010000-00011110-00101101
// CHECK-INST: usdot   za.s[w8, 5, vgx2], { z16.b, z17.b }, z0.b[3]
// CHECK-ENCODING: [0x2d,0x1e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1501e2d <unknown>

usdot   za.s[w8, 1, vgx2], {z0.b, z1.b}, z14.b[1]  // 11000001-01011110-00010100-00101001
// CHECK-INST: usdot   za.s[w8, 1, vgx2], { z0.b, z1.b }, z14.b[1]
// CHECK-ENCODING: [0x29,0x14,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e1429 <unknown>

usdot   za.s[w8, 1], {z0.b, z1.b}, z14.b[1]  // 11000001-01011110-00010100-00101001
// CHECK-INST: usdot   za.s[w8, 1, vgx2], { z0.b, z1.b }, z14.b[1]
// CHECK-ENCODING: [0x29,0x14,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e1429 <unknown>

usdot   za.s[w10, 0, vgx2], {z18.b, z19.b}, z4.b[1]  // 11000001-01010100-01010110-01101000
// CHECK-INST: usdot   za.s[w10, 0, vgx2], { z18.b, z19.b }, z4.b[1]
// CHECK-ENCODING: [0x68,0x56,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1545668 <unknown>

usdot   za.s[w10, 0], {z18.b, z19.b}, z4.b[1]  // 11000001-01010100-01010110-01101000
// CHECK-INST: usdot   za.s[w10, 0, vgx2], { z18.b, z19.b }, z4.b[1]
// CHECK-ENCODING: [0x68,0x56,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1545668 <unknown>

usdot   za.s[w8, 0, vgx2], {z12.b, z13.b}, z2.b[2]  // 11000001-01010010-00011001-10101000
// CHECK-INST: usdot   za.s[w8, 0, vgx2], { z12.b, z13.b }, z2.b[2]
// CHECK-ENCODING: [0xa8,0x19,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15219a8 <unknown>

usdot   za.s[w8, 0], {z12.b, z13.b}, z2.b[2]  // 11000001-01010010-00011001-10101000
// CHECK-INST: usdot   za.s[w8, 0, vgx2], { z12.b, z13.b }, z2.b[2]
// CHECK-ENCODING: [0xa8,0x19,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15219a8 <unknown>

usdot   za.s[w10, 1, vgx2], {z0.b, z1.b}, z10.b[2]  // 11000001-01011010-01011000-00101001
// CHECK-INST: usdot   za.s[w10, 1, vgx2], { z0.b, z1.b }, z10.b[2]
// CHECK-ENCODING: [0x29,0x58,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15a5829 <unknown>

usdot   za.s[w10, 1], {z0.b, z1.b}, z10.b[2]  // 11000001-01011010-01011000-00101001
// CHECK-INST: usdot   za.s[w10, 1, vgx2], { z0.b, z1.b }, z10.b[2]
// CHECK-ENCODING: [0x29,0x58,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15a5829 <unknown>

usdot   za.s[w8, 5, vgx2], {z22.b, z23.b}, z14.b[2]  // 11000001-01011110-00011010-11101101
// CHECK-INST: usdot   za.s[w8, 5, vgx2], { z22.b, z23.b }, z14.b[2]
// CHECK-ENCODING: [0xed,0x1a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e1aed <unknown>

usdot   za.s[w8, 5], {z22.b, z23.b}, z14.b[2]  // 11000001-01011110-00011010-11101101
// CHECK-INST: usdot   za.s[w8, 5, vgx2], { z22.b, z23.b }, z14.b[2]
// CHECK-ENCODING: [0xed,0x1a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e1aed <unknown>

usdot   za.s[w11, 2, vgx2], {z8.b, z9.b}, z1.b[1]  // 11000001-01010001-01110101-00101010
// CHECK-INST: usdot   za.s[w11, 2, vgx2], { z8.b, z9.b }, z1.b[1]
// CHECK-ENCODING: [0x2a,0x75,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c151752a <unknown>

usdot   za.s[w11, 2], {z8.b, z9.b}, z1.b[1]  // 11000001-01010001-01110101-00101010
// CHECK-INST: usdot   za.s[w11, 2, vgx2], { z8.b, z9.b }, z1.b[1]
// CHECK-ENCODING: [0x2a,0x75,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c151752a <unknown>

usdot   za.s[w9, 7, vgx2], {z12.b, z13.b}, z11.b[2]  // 11000001-01011011-00111001-10101111
// CHECK-INST: usdot   za.s[w9, 7, vgx2], { z12.b, z13.b }, z11.b[2]
// CHECK-ENCODING: [0xaf,0x39,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15b39af <unknown>

usdot   za.s[w9, 7], {z12.b, z13.b}, z11.b[2]  // 11000001-01011011-00111001-10101111
// CHECK-INST: usdot   za.s[w9, 7, vgx2], { z12.b, z13.b }, z11.b[2]
// CHECK-ENCODING: [0xaf,0x39,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15b39af <unknown>


usdot   za.s[w8, 0, vgx4], {z0.b - z3.b}, z0.b[0]  // 11000001-01010000-10010000-00101000
// CHECK-INST: usdot   za.s[w8, 0, vgx4], { z0.b - z3.b }, z0.b[0]
// CHECK-ENCODING: [0x28,0x90,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1509028 <unknown>

usdot   za.s[w8, 0], {z0.b - z3.b}, z0.b[0]  // 11000001-01010000-10010000-00101000
// CHECK-INST: usdot   za.s[w8, 0, vgx4], { z0.b - z3.b }, z0.b[0]
// CHECK-ENCODING: [0x28,0x90,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1509028 <unknown>

usdot   za.s[w10, 5, vgx4], {z8.b - z11.b}, z5.b[1]  // 11000001-01010101-11010101-00101101
// CHECK-INST: usdot   za.s[w10, 5, vgx4], { z8.b - z11.b }, z5.b[1]
// CHECK-ENCODING: [0x2d,0xd5,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c155d52d <unknown>

usdot   za.s[w10, 5], {z8.b - z11.b}, z5.b[1]  // 11000001-01010101-11010101-00101101
// CHECK-INST: usdot   za.s[w10, 5, vgx4], { z8.b - z11.b }, z5.b[1]
// CHECK-ENCODING: [0x2d,0xd5,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c155d52d <unknown>

usdot   za.s[w11, 7, vgx4], {z12.b - z15.b}, z8.b[3]  // 11000001-01011000-11111101-10101111
// CHECK-INST: usdot   za.s[w11, 7, vgx4], { z12.b - z15.b }, z8.b[3]
// CHECK-ENCODING: [0xaf,0xfd,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c158fdaf <unknown>

usdot   za.s[w11, 7], {z12.b - z15.b}, z8.b[3]  // 11000001-01011000-11111101-10101111
// CHECK-INST: usdot   za.s[w11, 7, vgx4], { z12.b - z15.b }, z8.b[3]
// CHECK-ENCODING: [0xaf,0xfd,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c158fdaf <unknown>

usdot   za.s[w11, 7, vgx4], {z28.b - z31.b}, z15.b[3]  // 11000001-01011111-11111111-10101111
// CHECK-INST: usdot   za.s[w11, 7, vgx4], { z28.b - z31.b }, z15.b[3]
// CHECK-ENCODING: [0xaf,0xff,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15fffaf <unknown>

usdot   za.s[w11, 7], {z28.b - z31.b}, z15.b[3]  // 11000001-01011111-11111111-10101111
// CHECK-INST: usdot   za.s[w11, 7, vgx4], { z28.b - z31.b }, z15.b[3]
// CHECK-ENCODING: [0xaf,0xff,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15fffaf <unknown>

usdot   za.s[w8, 5, vgx4], {z16.b - z19.b}, z0.b[3]  // 11000001-01010000-10011110-00101101
// CHECK-INST: usdot   za.s[w8, 5, vgx4], { z16.b - z19.b }, z0.b[3]
// CHECK-ENCODING: [0x2d,0x9e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1509e2d <unknown>

usdot   za.s[w8, 5], {z16.b - z19.b}, z0.b[3]  // 11000001-01010000-10011110-00101101
// CHECK-INST: usdot   za.s[w8, 5, vgx4], { z16.b - z19.b }, z0.b[3]
// CHECK-ENCODING: [0x2d,0x9e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1509e2d <unknown>

usdot   za.s[w8, 1, vgx4], {z0.b - z3.b}, z14.b[1]  // 11000001-01011110-10010100-00101001
// CHECK-INST: usdot   za.s[w8, 1, vgx4], { z0.b - z3.b }, z14.b[1]
// CHECK-ENCODING: [0x29,0x94,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e9429 <unknown>

usdot   za.s[w8, 1], {z0.b - z3.b}, z14.b[1]  // 11000001-01011110-10010100-00101001
// CHECK-INST: usdot   za.s[w8, 1, vgx4], { z0.b - z3.b }, z14.b[1]
// CHECK-ENCODING: [0x29,0x94,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e9429 <unknown>

usdot   za.s[w10, 0, vgx4], {z16.b - z19.b}, z4.b[1]  // 11000001-01010100-11010110-00101000
// CHECK-INST: usdot   za.s[w10, 0, vgx4], { z16.b - z19.b }, z4.b[1]
// CHECK-ENCODING: [0x28,0xd6,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c154d628 <unknown>

usdot   za.s[w10, 0], {z16.b - z19.b}, z4.b[1]  // 11000001-01010100-11010110-00101000
// CHECK-INST: usdot   za.s[w10, 0, vgx4], { z16.b - z19.b }, z4.b[1]
// CHECK-ENCODING: [0x28,0xd6,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c154d628 <unknown>

usdot   za.s[w8, 0, vgx4], {z12.b - z15.b}, z2.b[2]  // 11000001-01010010-10011001-10101000
// CHECK-INST: usdot   za.s[w8, 0, vgx4], { z12.b - z15.b }, z2.b[2]
// CHECK-ENCODING: [0xa8,0x99,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15299a8 <unknown>

usdot   za.s[w8, 0], {z12.b - z15.b}, z2.b[2]  // 11000001-01010010-10011001-10101000
// CHECK-INST: usdot   za.s[w8, 0, vgx4], { z12.b - z15.b }, z2.b[2]
// CHECK-ENCODING: [0xa8,0x99,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15299a8 <unknown>

usdot   za.s[w10, 1, vgx4], {z0.b - z3.b}, z10.b[2]  // 11000001-01011010-11011000-00101001
// CHECK-INST: usdot   za.s[w10, 1, vgx4], { z0.b - z3.b }, z10.b[2]
// CHECK-ENCODING: [0x29,0xd8,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15ad829 <unknown>

usdot   za.s[w10, 1], {z0.b - z3.b}, z10.b[2]  // 11000001-01011010-11011000-00101001
// CHECK-INST: usdot   za.s[w10, 1, vgx4], { z0.b - z3.b }, z10.b[2]
// CHECK-ENCODING: [0x29,0xd8,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15ad829 <unknown>

usdot   za.s[w8, 5, vgx4], {z20.b - z23.b}, z14.b[2]  // 11000001-01011110-10011010-10101101
// CHECK-INST: usdot   za.s[w8, 5, vgx4], { z20.b - z23.b }, z14.b[2]
// CHECK-ENCODING: [0xad,0x9a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e9aad <unknown>

usdot   za.s[w8, 5], {z20.b - z23.b}, z14.b[2]  // 11000001-01011110-10011010-10101101
// CHECK-INST: usdot   za.s[w8, 5, vgx4], { z20.b - z23.b }, z14.b[2]
// CHECK-ENCODING: [0xad,0x9a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e9aad <unknown>

usdot   za.s[w11, 2, vgx4], {z8.b - z11.b}, z1.b[1]  // 11000001-01010001-11110101-00101010
// CHECK-INST: usdot   za.s[w11, 2, vgx4], { z8.b - z11.b }, z1.b[1]
// CHECK-ENCODING: [0x2a,0xf5,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c151f52a <unknown>

usdot   za.s[w11, 2], {z8.b - z11.b}, z1.b[1]  // 11000001-01010001-11110101-00101010
// CHECK-INST: usdot   za.s[w11, 2, vgx4], { z8.b - z11.b }, z1.b[1]
// CHECK-ENCODING: [0x2a,0xf5,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c151f52a <unknown>

usdot   za.s[w9, 7, vgx4], {z12.b - z15.b}, z11.b[2]  // 11000001-01011011-10111001-10101111
// CHECK-INST: usdot   za.s[w9, 7, vgx4], { z12.b - z15.b }, z11.b[2]
// CHECK-ENCODING: [0xaf,0xb9,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15bb9af <unknown>

usdot   za.s[w9, 7], {z12.b - z15.b}, z11.b[2]  // 11000001-01011011-10111001-10101111
// CHECK-INST: usdot   za.s[w9, 7, vgx4], { z12.b - z15.b }, z11.b[2]
// CHECK-ENCODING: [0xaf,0xb9,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15bb9af <unknown>

