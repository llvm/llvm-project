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


usdot   za.s[w8, 0, vgx2], {z0.b, z1.b}, z0.b  // 11000001-00100000-00010100-00001000
// CHECK-INST: usdot   za.s[w8, 0, vgx2], { z0.b, z1.b }, z0.b
// CHECK-ENCODING: [0x08,0x14,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1201408 <unknown>

usdot   za.s[w8, 0], {z0.b, z1.b}, z0.b  // 11000001-00100000-00010100-00001000
// CHECK-INST: usdot   za.s[w8, 0, vgx2], { z0.b, z1.b }, z0.b
// CHECK-ENCODING: [0x08,0x14,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1201408 <unknown>

usdot   za.s[w10, 5, vgx2], {z10.b, z11.b}, z5.b  // 11000001-00100101-01010101-01001101
// CHECK-INST: usdot   za.s[w10, 5, vgx2], { z10.b, z11.b }, z5.b
// CHECK-ENCODING: [0x4d,0x55,0x25,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c125554d <unknown>

usdot   za.s[w10, 5], {z10.b, z11.b}, z5.b  // 11000001-00100101-01010101-01001101
// CHECK-INST: usdot   za.s[w10, 5, vgx2], { z10.b, z11.b }, z5.b
// CHECK-ENCODING: [0x4d,0x55,0x25,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c125554d <unknown>

usdot   za.s[w11, 7, vgx2], {z13.b, z14.b}, z8.b  // 11000001-00101000-01110101-10101111
// CHECK-INST: usdot   za.s[w11, 7, vgx2], { z13.b, z14.b }, z8.b
// CHECK-ENCODING: [0xaf,0x75,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12875af <unknown>

usdot   za.s[w11, 7], {z13.b, z14.b}, z8.b  // 11000001-00101000-01110101-10101111
// CHECK-INST: usdot   za.s[w11, 7, vgx2], { z13.b, z14.b }, z8.b
// CHECK-ENCODING: [0xaf,0x75,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12875af <unknown>

usdot   za.s[w11, 7, vgx2], {z31.b, z0.b}, z15.b  // 11000001-00101111-01110111-11101111
// CHECK-INST: usdot   za.s[w11, 7, vgx2], { z31.b, z0.b }, z15.b
// CHECK-ENCODING: [0xef,0x77,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12f77ef <unknown>

usdot   za.s[w11, 7], {z31.b, z0.b}, z15.b  // 11000001-00101111-01110111-11101111
// CHECK-INST: usdot   za.s[w11, 7, vgx2], { z31.b, z0.b }, z15.b
// CHECK-ENCODING: [0xef,0x77,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12f77ef <unknown>

usdot   za.s[w8, 5, vgx2], {z17.b, z18.b}, z0.b  // 11000001-00100000-00010110-00101101
// CHECK-INST: usdot   za.s[w8, 5, vgx2], { z17.b, z18.b }, z0.b
// CHECK-ENCODING: [0x2d,0x16,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c120162d <unknown>

usdot   za.s[w8, 5], {z17.b, z18.b}, z0.b  // 11000001-00100000-00010110-00101101
// CHECK-INST: usdot   za.s[w8, 5, vgx2], { z17.b, z18.b }, z0.b
// CHECK-ENCODING: [0x2d,0x16,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c120162d <unknown>

usdot   za.s[w8, 1, vgx2], {z1.b, z2.b}, z14.b  // 11000001-00101110-00010100-00101001
// CHECK-INST: usdot   za.s[w8, 1, vgx2], { z1.b, z2.b }, z14.b
// CHECK-ENCODING: [0x29,0x14,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e1429 <unknown>

usdot   za.s[w8, 1], {z1.b, z2.b}, z14.b  // 11000001-00101110-00010100-00101001
// CHECK-INST: usdot   za.s[w8, 1, vgx2], { z1.b, z2.b }, z14.b
// CHECK-ENCODING: [0x29,0x14,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e1429 <unknown>

usdot   za.s[w10, 0, vgx2], {z19.b, z20.b}, z4.b  // 11000001-00100100-01010110-01101000
// CHECK-INST: usdot   za.s[w10, 0, vgx2], { z19.b, z20.b }, z4.b
// CHECK-ENCODING: [0x68,0x56,0x24,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1245668 <unknown>

usdot   za.s[w10, 0], {z19.b, z20.b}, z4.b  // 11000001-00100100-01010110-01101000
// CHECK-INST: usdot   za.s[w10, 0, vgx2], { z19.b, z20.b }, z4.b
// CHECK-ENCODING: [0x68,0x56,0x24,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1245668 <unknown>

usdot   za.s[w8, 0, vgx2], {z12.b, z13.b}, z2.b  // 11000001-00100010-00010101-10001000
// CHECK-INST: usdot   za.s[w8, 0, vgx2], { z12.b, z13.b }, z2.b
// CHECK-ENCODING: [0x88,0x15,0x22,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1221588 <unknown>

usdot   za.s[w8, 0], {z12.b, z13.b}, z2.b  // 11000001-00100010-00010101-10001000
// CHECK-INST: usdot   za.s[w8, 0, vgx2], { z12.b, z13.b }, z2.b
// CHECK-ENCODING: [0x88,0x15,0x22,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1221588 <unknown>

usdot   za.s[w10, 1, vgx2], {z1.b, z2.b}, z10.b  // 11000001-00101010-01010100-00101001
// CHECK-INST: usdot   za.s[w10, 1, vgx2], { z1.b, z2.b }, z10.b
// CHECK-ENCODING: [0x29,0x54,0x2a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12a5429 <unknown>

usdot   za.s[w10, 1], {z1.b, z2.b}, z10.b  // 11000001-00101010-01010100-00101001
// CHECK-INST: usdot   za.s[w10, 1, vgx2], { z1.b, z2.b }, z10.b
// CHECK-ENCODING: [0x29,0x54,0x2a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12a5429 <unknown>

usdot   za.s[w8, 5, vgx2], {z22.b, z23.b}, z14.b  // 11000001-00101110-00010110-11001101
// CHECK-INST: usdot   za.s[w8, 5, vgx2], { z22.b, z23.b }, z14.b
// CHECK-ENCODING: [0xcd,0x16,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e16cd <unknown>

usdot   za.s[w8, 5], {z22.b, z23.b}, z14.b  // 11000001-00101110-00010110-11001101
// CHECK-INST: usdot   za.s[w8, 5, vgx2], { z22.b, z23.b }, z14.b
// CHECK-ENCODING: [0xcd,0x16,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e16cd <unknown>

usdot   za.s[w11, 2, vgx2], {z9.b, z10.b}, z1.b  // 11000001-00100001-01110101-00101010
// CHECK-INST: usdot   za.s[w11, 2, vgx2], { z9.b, z10.b }, z1.b
// CHECK-ENCODING: [0x2a,0x75,0x21,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c121752a <unknown>

usdot   za.s[w11, 2], {z9.b, z10.b}, z1.b  // 11000001-00100001-01110101-00101010
// CHECK-INST: usdot   za.s[w11, 2, vgx2], { z9.b, z10.b }, z1.b
// CHECK-ENCODING: [0x2a,0x75,0x21,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c121752a <unknown>

usdot   za.s[w9, 7, vgx2], {z12.b, z13.b}, z11.b  // 11000001-00101011-00110101-10001111
// CHECK-INST: usdot   za.s[w9, 7, vgx2], { z12.b, z13.b }, z11.b
// CHECK-ENCODING: [0x8f,0x35,0x2b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12b358f <unknown>

usdot   za.s[w9, 7], {z12.b, z13.b}, z11.b  // 11000001-00101011-00110101-10001111
// CHECK-INST: usdot   za.s[w9, 7, vgx2], { z12.b, z13.b }, z11.b
// CHECK-ENCODING: [0x8f,0x35,0x2b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12b358f <unknown>


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


usdot   za.s[w8, 0, vgx2], {z0.b, z1.b}, {z0.b, z1.b}  // 11000001-10100000-00010100-00001000
// CHECK-INST: usdot   za.s[w8, 0, vgx2], { z0.b, z1.b }, { z0.b, z1.b }
// CHECK-ENCODING: [0x08,0x14,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a01408 <unknown>

usdot   za.s[w8, 0], {z0.b, z1.b}, {z0.b, z1.b}  // 11000001-10100000-00010100-00001000
// CHECK-INST: usdot   za.s[w8, 0, vgx2], { z0.b, z1.b }, { z0.b, z1.b }
// CHECK-ENCODING: [0x08,0x14,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a01408 <unknown>

usdot   za.s[w10, 5, vgx2], {z10.b, z11.b}, {z20.b, z21.b}  // 11000001-10110100-01010101-01001101
// CHECK-INST: usdot   za.s[w10, 5, vgx2], { z10.b, z11.b }, { z20.b, z21.b }
// CHECK-ENCODING: [0x4d,0x55,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b4554d <unknown>

usdot   za.s[w10, 5], {z10.b, z11.b}, {z20.b, z21.b}  // 11000001-10110100-01010101-01001101
// CHECK-INST: usdot   za.s[w10, 5, vgx2], { z10.b, z11.b }, { z20.b, z21.b }
// CHECK-ENCODING: [0x4d,0x55,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b4554d <unknown>

usdot   za.s[w11, 7, vgx2], {z12.b, z13.b}, {z8.b, z9.b}  // 11000001-10101000-01110101-10001111
// CHECK-INST: usdot   za.s[w11, 7, vgx2], { z12.b, z13.b }, { z8.b, z9.b }
// CHECK-ENCODING: [0x8f,0x75,0xa8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a8758f <unknown>

usdot   za.s[w11, 7], {z12.b, z13.b}, {z8.b, z9.b}  // 11000001-10101000-01110101-10001111
// CHECK-INST: usdot   za.s[w11, 7, vgx2], { z12.b, z13.b }, { z8.b, z9.b }
// CHECK-ENCODING: [0x8f,0x75,0xa8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a8758f <unknown>

usdot   za.s[w11, 7, vgx2], {z30.b, z31.b}, {z30.b, z31.b}  // 11000001-10111110-01110111-11001111
// CHECK-INST: usdot   za.s[w11, 7, vgx2], { z30.b, z31.b }, { z30.b, z31.b }
// CHECK-ENCODING: [0xcf,0x77,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be77cf <unknown>

usdot   za.s[w11, 7], {z30.b, z31.b}, {z30.b, z31.b}  // 11000001-10111110-01110111-11001111
// CHECK-INST: usdot   za.s[w11, 7, vgx2], { z30.b, z31.b }, { z30.b, z31.b }
// CHECK-ENCODING: [0xcf,0x77,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be77cf <unknown>

usdot   za.s[w8, 5, vgx2], {z16.b, z17.b}, {z16.b, z17.b}  // 11000001-10110000-00010110-00001101
// CHECK-INST: usdot   za.s[w8, 5, vgx2], { z16.b, z17.b }, { z16.b, z17.b }
// CHECK-ENCODING: [0x0d,0x16,0xb0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b0160d <unknown>

usdot   za.s[w8, 5], {z16.b, z17.b}, {z16.b, z17.b}  // 11000001-10110000-00010110-00001101
// CHECK-INST: usdot   za.s[w8, 5, vgx2], { z16.b, z17.b }, { z16.b, z17.b }
// CHECK-ENCODING: [0x0d,0x16,0xb0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b0160d <unknown>

usdot   za.s[w8, 1, vgx2], {z0.b, z1.b}, {z30.b, z31.b}  // 11000001-10111110-00010100-00001001
// CHECK-INST: usdot   za.s[w8, 1, vgx2], { z0.b, z1.b }, { z30.b, z31.b }
// CHECK-ENCODING: [0x09,0x14,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be1409 <unknown>

usdot   za.s[w8, 1], {z0.b, z1.b}, {z30.b, z31.b}  // 11000001-10111110-00010100-00001001
// CHECK-INST: usdot   za.s[w8, 1, vgx2], { z0.b, z1.b }, { z30.b, z31.b }
// CHECK-ENCODING: [0x09,0x14,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be1409 <unknown>

usdot   za.s[w10, 0, vgx2], {z18.b, z19.b}, {z20.b, z21.b}  // 11000001-10110100-01010110-01001000
// CHECK-INST: usdot   za.s[w10, 0, vgx2], { z18.b, z19.b }, { z20.b, z21.b }
// CHECK-ENCODING: [0x48,0x56,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b45648 <unknown>

usdot   za.s[w10, 0], {z18.b, z19.b}, {z20.b, z21.b}  // 11000001-10110100-01010110-01001000
// CHECK-INST: usdot   za.s[w10, 0, vgx2], { z18.b, z19.b }, { z20.b, z21.b }
// CHECK-ENCODING: [0x48,0x56,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b45648 <unknown>

usdot   za.s[w8, 0, vgx2], {z12.b, z13.b}, {z2.b, z3.b}  // 11000001-10100010-00010101-10001000
// CHECK-INST: usdot   za.s[w8, 0, vgx2], { z12.b, z13.b }, { z2.b, z3.b }
// CHECK-ENCODING: [0x88,0x15,0xa2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a21588 <unknown>

usdot   za.s[w8, 0], {z12.b, z13.b}, {z2.b, z3.b}  // 11000001-10100010-00010101-10001000
// CHECK-INST: usdot   za.s[w8, 0, vgx2], { z12.b, z13.b }, { z2.b, z3.b }
// CHECK-ENCODING: [0x88,0x15,0xa2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a21588 <unknown>

usdot   za.s[w10, 1, vgx2], {z0.b, z1.b}, {z26.b, z27.b}  // 11000001-10111010-01010100-00001001
// CHECK-INST: usdot   za.s[w10, 1, vgx2], { z0.b, z1.b }, { z26.b, z27.b }
// CHECK-ENCODING: [0x09,0x54,0xba,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ba5409 <unknown>

usdot   za.s[w10, 1], {z0.b, z1.b}, {z26.b, z27.b}  // 11000001-10111010-01010100-00001001
// CHECK-INST: usdot   za.s[w10, 1, vgx2], { z0.b, z1.b }, { z26.b, z27.b }
// CHECK-ENCODING: [0x09,0x54,0xba,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ba5409 <unknown>

usdot   za.s[w8, 5, vgx2], {z22.b, z23.b}, {z30.b, z31.b}  // 11000001-10111110-00010110-11001101
// CHECK-INST: usdot   za.s[w8, 5, vgx2], { z22.b, z23.b }, { z30.b, z31.b }
// CHECK-ENCODING: [0xcd,0x16,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be16cd <unknown>

usdot   za.s[w8, 5], {z22.b, z23.b}, {z30.b, z31.b}  // 11000001-10111110-00010110-11001101
// CHECK-INST: usdot   za.s[w8, 5, vgx2], { z22.b, z23.b }, { z30.b, z31.b }
// CHECK-ENCODING: [0xcd,0x16,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be16cd <unknown>

usdot   za.s[w11, 2, vgx2], {z8.b, z9.b}, {z0.b, z1.b}  // 11000001-10100000-01110101-00001010
// CHECK-INST: usdot   za.s[w11, 2, vgx2], { z8.b, z9.b }, { z0.b, z1.b }
// CHECK-ENCODING: [0x0a,0x75,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a0750a <unknown>

usdot   za.s[w11, 2], {z8.b, z9.b}, {z0.b, z1.b}  // 11000001-10100000-01110101-00001010
// CHECK-INST: usdot   za.s[w11, 2, vgx2], { z8.b, z9.b }, { z0.b, z1.b }
// CHECK-ENCODING: [0x0a,0x75,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a0750a <unknown>

usdot   za.s[w9, 7, vgx2], {z12.b, z13.b}, {z10.b, z11.b}  // 11000001-10101010-00110101-10001111
// CHECK-INST: usdot   za.s[w9, 7, vgx2], { z12.b, z13.b }, { z10.b, z11.b }
// CHECK-ENCODING: [0x8f,0x35,0xaa,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1aa358f <unknown>

usdot   za.s[w9, 7], {z12.b, z13.b}, {z10.b, z11.b}  // 11000001-10101010-00110101-10001111
// CHECK-INST: usdot   za.s[w9, 7, vgx2], { z12.b, z13.b }, { z10.b, z11.b }
// CHECK-ENCODING: [0x8f,0x35,0xaa,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1aa358f <unknown>

usdot   za.s[w8, 0, vgx4], {z0.b - z3.b}, z0.b  // 11000001-00110000-00010100-00001000
// CHECK-INST: usdot   za.s[w8, 0, vgx4], { z0.b - z3.b }, z0.b
// CHECK-ENCODING: [0x08,0x14,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1301408 <unknown>

usdot   za.s[w8, 0], {z0.b - z3.b}, z0.b  // 11000001-00110000-00010100-00001000
// CHECK-INST: usdot   za.s[w8, 0, vgx4], { z0.b - z3.b }, z0.b
// CHECK-ENCODING: [0x08,0x14,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1301408 <unknown>

usdot   za.s[w10, 5, vgx4], {z10.b - z13.b}, z5.b  // 11000001-00110101-01010101-01001101
// CHECK-INST: usdot   za.s[w10, 5, vgx4], { z10.b - z13.b }, z5.b
// CHECK-ENCODING: [0x4d,0x55,0x35,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c135554d <unknown>

usdot   za.s[w10, 5], {z10.b - z13.b}, z5.b  // 11000001-00110101-01010101-01001101
// CHECK-INST: usdot   za.s[w10, 5, vgx4], { z10.b - z13.b }, z5.b
// CHECK-ENCODING: [0x4d,0x55,0x35,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c135554d <unknown>

usdot   za.s[w11, 7, vgx4], {z13.b - z16.b}, z8.b  // 11000001-00111000-01110101-10101111
// CHECK-INST: usdot   za.s[w11, 7, vgx4], { z13.b - z16.b }, z8.b
// CHECK-ENCODING: [0xaf,0x75,0x38,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13875af <unknown>

usdot   za.s[w11, 7], {z13.b - z16.b}, z8.b  // 11000001-00111000-01110101-10101111
// CHECK-INST: usdot   za.s[w11, 7, vgx4], { z13.b - z16.b }, z8.b
// CHECK-ENCODING: [0xaf,0x75,0x38,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13875af <unknown>

usdot   za.s[w11, 7, vgx4], {z31.b - z2.b}, z15.b  // 11000001-00111111-01110111-11101111
// CHECK-INST: usdot   za.s[w11, 7, vgx4], { z31.b, z0.b, z1.b, z2.b }, z15.b
// CHECK-ENCODING: [0xef,0x77,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13f77ef <unknown>

usdot   za.s[w11, 7], {z31.b - z2.b}, z15.b  // 11000001-00111111-01110111-11101111
// CHECK-INST: usdot   za.s[w11, 7, vgx4], { z31.b, z0.b, z1.b, z2.b }, z15.b
// CHECK-ENCODING: [0xef,0x77,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13f77ef <unknown>

usdot   za.s[w8, 5, vgx4], {z17.b - z20.b}, z0.b  // 11000001-00110000-00010110-00101101
// CHECK-INST: usdot   za.s[w8, 5, vgx4], { z17.b - z20.b }, z0.b
// CHECK-ENCODING: [0x2d,0x16,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c130162d <unknown>

usdot   za.s[w8, 5], {z17.b - z20.b}, z0.b  // 11000001-00110000-00010110-00101101
// CHECK-INST: usdot   za.s[w8, 5, vgx4], { z17.b - z20.b }, z0.b
// CHECK-ENCODING: [0x2d,0x16,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c130162d <unknown>

usdot   za.s[w8, 1, vgx4], {z1.b - z4.b}, z14.b  // 11000001-00111110-00010100-00101001
// CHECK-INST: usdot   za.s[w8, 1, vgx4], { z1.b - z4.b }, z14.b
// CHECK-ENCODING: [0x29,0x14,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e1429 <unknown>

usdot   za.s[w8, 1], {z1.b - z4.b}, z14.b  // 11000001-00111110-00010100-00101001
// CHECK-INST: usdot   za.s[w8, 1, vgx4], { z1.b - z4.b }, z14.b
// CHECK-ENCODING: [0x29,0x14,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e1429 <unknown>

usdot   za.s[w10, 0, vgx4], {z19.b - z22.b}, z4.b  // 11000001-00110100-01010110-01101000
// CHECK-INST: usdot   za.s[w10, 0, vgx4], { z19.b - z22.b }, z4.b
// CHECK-ENCODING: [0x68,0x56,0x34,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1345668 <unknown>

usdot   za.s[w10, 0], {z19.b - z22.b}, z4.b  // 11000001-00110100-01010110-01101000
// CHECK-INST: usdot   za.s[w10, 0, vgx4], { z19.b - z22.b }, z4.b
// CHECK-ENCODING: [0x68,0x56,0x34,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1345668 <unknown>

usdot   za.s[w8, 0, vgx4], {z12.b - z15.b}, z2.b  // 11000001-00110010-00010101-10001000
// CHECK-INST: usdot   za.s[w8, 0, vgx4], { z12.b - z15.b }, z2.b
// CHECK-ENCODING: [0x88,0x15,0x32,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1321588 <unknown>

usdot   za.s[w8, 0], {z12.b - z15.b}, z2.b  // 11000001-00110010-00010101-10001000
// CHECK-INST: usdot   za.s[w8, 0, vgx4], { z12.b - z15.b }, z2.b
// CHECK-ENCODING: [0x88,0x15,0x32,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1321588 <unknown>

usdot   za.s[w10, 1, vgx4], {z1.b - z4.b}, z10.b  // 11000001-00111010-01010100-00101001
// CHECK-INST: usdot   za.s[w10, 1, vgx4], { z1.b - z4.b }, z10.b
// CHECK-ENCODING: [0x29,0x54,0x3a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13a5429 <unknown>

usdot   za.s[w10, 1], {z1.b - z4.b}, z10.b  // 11000001-00111010-01010100-00101001
// CHECK-INST: usdot   za.s[w10, 1, vgx4], { z1.b - z4.b }, z10.b
// CHECK-ENCODING: [0x29,0x54,0x3a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13a5429 <unknown>

usdot   za.s[w8, 5, vgx4], {z22.b - z25.b}, z14.b  // 11000001-00111110-00010110-11001101
// CHECK-INST: usdot   za.s[w8, 5, vgx4], { z22.b - z25.b }, z14.b
// CHECK-ENCODING: [0xcd,0x16,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e16cd <unknown>

usdot   za.s[w8, 5], {z22.b - z25.b}, z14.b  // 11000001-00111110-00010110-11001101
// CHECK-INST: usdot   za.s[w8, 5, vgx4], { z22.b - z25.b }, z14.b
// CHECK-ENCODING: [0xcd,0x16,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e16cd <unknown>

usdot   za.s[w11, 2, vgx4], {z9.b - z12.b}, z1.b  // 11000001-00110001-01110101-00101010
// CHECK-INST: usdot   za.s[w11, 2, vgx4], { z9.b - z12.b }, z1.b
// CHECK-ENCODING: [0x2a,0x75,0x31,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c131752a <unknown>

usdot   za.s[w11, 2], {z9.b - z12.b}, z1.b  // 11000001-00110001-01110101-00101010
// CHECK-INST: usdot   za.s[w11, 2, vgx4], { z9.b - z12.b }, z1.b
// CHECK-ENCODING: [0x2a,0x75,0x31,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c131752a <unknown>

usdot   za.s[w9, 7, vgx4], {z12.b - z15.b}, z11.b  // 11000001-00111011-00110101-10001111
// CHECK-INST: usdot   za.s[w9, 7, vgx4], { z12.b - z15.b }, z11.b
// CHECK-ENCODING: [0x8f,0x35,0x3b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13b358f <unknown>

usdot   za.s[w9, 7], {z12.b - z15.b}, z11.b  // 11000001-00111011-00110101-10001111
// CHECK-INST: usdot   za.s[w9, 7, vgx4], { z12.b - z15.b }, z11.b
// CHECK-ENCODING: [0x8f,0x35,0x3b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13b358f <unknown>


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


usdot   za.s[w8, 0, vgx4], {z0.b - z3.b}, {z0.b - z3.b}  // 11000001-10100001-00010100-00001000
// CHECK-INST: usdot   za.s[w8, 0, vgx4], { z0.b - z3.b }, { z0.b - z3.b }
// CHECK-ENCODING: [0x08,0x14,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11408 <unknown>

usdot   za.s[w8, 0], {z0.b - z3.b}, {z0.b - z3.b}  // 11000001-10100001-00010100-00001000
// CHECK-INST: usdot   za.s[w8, 0, vgx4], { z0.b - z3.b }, { z0.b - z3.b }
// CHECK-ENCODING: [0x08,0x14,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11408 <unknown>

usdot   za.s[w10, 5, vgx4], {z8.b - z11.b}, {z20.b - z23.b}  // 11000001-10110101-01010101-00001101
// CHECK-INST: usdot   za.s[w10, 5, vgx4], { z8.b - z11.b }, { z20.b - z23.b }
// CHECK-ENCODING: [0x0d,0x55,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b5550d <unknown>

usdot   za.s[w10, 5], {z8.b - z11.b}, {z20.b - z23.b}  // 11000001-10110101-01010101-00001101
// CHECK-INST: usdot   za.s[w10, 5, vgx4], { z8.b - z11.b }, { z20.b - z23.b }
// CHECK-ENCODING: [0x0d,0x55,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b5550d <unknown>

usdot   za.s[w11, 7, vgx4], {z12.b - z15.b}, {z8.b - z11.b}  // 11000001-10101001-01110101-10001111
// CHECK-INST: usdot   za.s[w11, 7, vgx4], { z12.b - z15.b }, { z8.b - z11.b }
// CHECK-ENCODING: [0x8f,0x75,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a9758f <unknown>

usdot   za.s[w11, 7], {z12.b - z15.b}, {z8.b - z11.b}  // 11000001-10101001-01110101-10001111
// CHECK-INST: usdot   za.s[w11, 7, vgx4], { z12.b - z15.b }, { z8.b - z11.b }
// CHECK-ENCODING: [0x8f,0x75,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a9758f <unknown>

usdot   za.s[w11, 7, vgx4], {z28.b - z31.b}, {z28.b - z31.b}  // 11000001-10111101-01110111-10001111
// CHECK-INST: usdot   za.s[w11, 7, vgx4], { z28.b - z31.b }, { z28.b - z31.b }
// CHECK-ENCODING: [0x8f,0x77,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd778f <unknown>

usdot   za.s[w11, 7], {z28.b - z31.b}, {z28.b - z31.b}  // 11000001-10111101-01110111-10001111
// CHECK-INST: usdot   za.s[w11, 7, vgx4], { z28.b - z31.b }, { z28.b - z31.b }
// CHECK-ENCODING: [0x8f,0x77,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd778f <unknown>

usdot   za.s[w8, 5, vgx4], {z16.b - z19.b}, {z16.b - z19.b}  // 11000001-10110001-00010110-00001101
// CHECK-INST: usdot   za.s[w8, 5, vgx4], { z16.b - z19.b }, { z16.b - z19.b }
// CHECK-ENCODING: [0x0d,0x16,0xb1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b1160d <unknown>

usdot   za.s[w8, 5], {z16.b - z19.b}, {z16.b - z19.b}  // 11000001-10110001-00010110-00001101
// CHECK-INST: usdot   za.s[w8, 5, vgx4], { z16.b - z19.b }, { z16.b - z19.b }
// CHECK-ENCODING: [0x0d,0x16,0xb1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b1160d <unknown>

usdot   za.s[w8, 1, vgx4], {z0.b - z3.b}, {z28.b - z31.b}  // 11000001-10111101-00010100-00001001
// CHECK-INST: usdot   za.s[w8, 1, vgx4], { z0.b - z3.b }, { z28.b - z31.b }
// CHECK-ENCODING: [0x09,0x14,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd1409 <unknown>

usdot   za.s[w8, 1], {z0.b - z3.b}, {z28.b - z31.b}  // 11000001-10111101-00010100-00001001
// CHECK-INST: usdot   za.s[w8, 1, vgx4], { z0.b - z3.b }, { z28.b - z31.b }
// CHECK-ENCODING: [0x09,0x14,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd1409 <unknown>

usdot   za.s[w10, 0, vgx4], {z16.b - z19.b}, {z20.b - z23.b}  // 11000001-10110101-01010110-00001000
// CHECK-INST: usdot   za.s[w10, 0, vgx4], { z16.b - z19.b }, { z20.b - z23.b }
// CHECK-ENCODING: [0x08,0x56,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b55608 <unknown>

usdot   za.s[w10, 0], {z16.b - z19.b}, {z20.b - z23.b}  // 11000001-10110101-01010110-00001000
// CHECK-INST: usdot   za.s[w10, 0, vgx4], { z16.b - z19.b }, { z20.b - z23.b }
// CHECK-ENCODING: [0x08,0x56,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b55608 <unknown>

usdot   za.s[w8, 0, vgx4], {z12.b - z15.b}, {z0.b - z3.b}  // 11000001-10100001-00010101-10001000
// CHECK-INST: usdot   za.s[w8, 0, vgx4], { z12.b - z15.b }, { z0.b - z3.b }
// CHECK-ENCODING: [0x88,0x15,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11588 <unknown>

usdot   za.s[w8, 0], {z12.b - z15.b}, {z0.b - z3.b}  // 11000001-10100001-00010101-10001000
// CHECK-INST: usdot   za.s[w8, 0, vgx4], { z12.b - z15.b }, { z0.b - z3.b }
// CHECK-ENCODING: [0x88,0x15,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11588 <unknown>

usdot   za.s[w10, 1, vgx4], {z0.b - z3.b}, {z24.b - z27.b}  // 11000001-10111001-01010100-00001001
// CHECK-INST: usdot   za.s[w10, 1, vgx4], { z0.b - z3.b }, { z24.b - z27.b }
// CHECK-ENCODING: [0x09,0x54,0xb9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b95409 <unknown>

usdot   za.s[w10, 1], {z0.b - z3.b}, {z24.b - z27.b}  // 11000001-10111001-01010100-00001001
// CHECK-INST: usdot   za.s[w10, 1, vgx4], { z0.b - z3.b }, { z24.b - z27.b }
// CHECK-ENCODING: [0x09,0x54,0xb9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b95409 <unknown>

usdot   za.s[w8, 5, vgx4], {z20.b - z23.b}, {z28.b - z31.b}  // 11000001-10111101-00010110-10001101
// CHECK-INST: usdot   za.s[w8, 5, vgx4], { z20.b - z23.b }, { z28.b - z31.b }
// CHECK-ENCODING: [0x8d,0x16,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd168d <unknown>

usdot   za.s[w8, 5], {z20.b - z23.b}, {z28.b - z31.b}  // 11000001-10111101-00010110-10001101
// CHECK-INST: usdot   za.s[w8, 5, vgx4], { z20.b - z23.b }, { z28.b - z31.b }
// CHECK-ENCODING: [0x8d,0x16,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd168d <unknown>

usdot   za.s[w11, 2, vgx4], {z8.b - z11.b}, {z0.b - z3.b}  // 11000001-10100001-01110101-00001010
// CHECK-INST: usdot   za.s[w11, 2, vgx4], { z8.b - z11.b }, { z0.b - z3.b }
// CHECK-ENCODING: [0x0a,0x75,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a1750a <unknown>

usdot   za.s[w11, 2], {z8.b - z11.b}, {z0.b - z3.b}  // 11000001-10100001-01110101-00001010
// CHECK-INST: usdot   za.s[w11, 2, vgx4], { z8.b - z11.b }, { z0.b - z3.b }
// CHECK-ENCODING: [0x0a,0x75,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a1750a <unknown>

usdot   za.s[w9, 7, vgx4], {z12.b - z15.b}, {z8.b - z11.b}  // 11000001-10101001-00110101-10001111
// CHECK-INST: usdot   za.s[w9, 7, vgx4], { z12.b - z15.b }, { z8.b - z11.b }
// CHECK-ENCODING: [0x8f,0x35,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a9358f <unknown>

usdot   za.s[w9, 7], {z12.b - z15.b}, {z8.b - z11.b}  // 11000001-10101001-00110101-10001111
// CHECK-INST: usdot   za.s[w9, 7, vgx4], { z12.b - z15.b }, { z8.b - z11.b }
// CHECK-ENCODING: [0x8f,0x35,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a9358f <unknown>

