// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+sme-i16i64 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2,+sme-i16i64 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2,+sme-i16i64 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2,+sme-i16i64 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+sme-i16i64 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2,+sme-i16i64 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


sdot    za.s[w8, 0, vgx2], {z0.h, z1.h}, z0.h  // 11000001-01100000-00010100-00001000
// CHECK-INST: sdot    za.s[w8, 0, vgx2], { z0.h, z1.h }, z0.h
// CHECK-ENCODING: [0x08,0x14,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1601408 <unknown>

sdot    za.s[w8, 0], {z0.h, z1.h}, z0.h  // 11000001-01100000-00010100-00001000
// CHECK-INST: sdot    za.s[w8, 0, vgx2], { z0.h, z1.h }, z0.h
// CHECK-ENCODING: [0x08,0x14,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1601408 <unknown>

sdot    za.s[w10, 5, vgx2], {z10.h, z11.h}, z5.h  // 11000001-01100101-01010101-01001101
// CHECK-INST: sdot    za.s[w10, 5, vgx2], { z10.h, z11.h }, z5.h
// CHECK-ENCODING: [0x4d,0x55,0x65,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c165554d <unknown>

sdot    za.s[w10, 5], {z10.h, z11.h}, z5.h  // 11000001-01100101-01010101-01001101
// CHECK-INST: sdot    za.s[w10, 5, vgx2], { z10.h, z11.h }, z5.h
// CHECK-ENCODING: [0x4d,0x55,0x65,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c165554d <unknown>

sdot    za.s[w11, 7, vgx2], {z13.h, z14.h}, z8.h  // 11000001-01101000-01110101-10101111
// CHECK-INST: sdot    za.s[w11, 7, vgx2], { z13.h, z14.h }, z8.h
// CHECK-ENCODING: [0xaf,0x75,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16875af <unknown>

sdot    za.s[w11, 7], {z13.h, z14.h}, z8.h  // 11000001-01101000-01110101-10101111
// CHECK-INST: sdot    za.s[w11, 7, vgx2], { z13.h, z14.h }, z8.h
// CHECK-ENCODING: [0xaf,0x75,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16875af <unknown>

sdot    za.s[w11, 7, vgx2], {z31.h, z0.h}, z15.h  // 11000001-01101111-01110111-11101111
// CHECK-INST: sdot    za.s[w11, 7, vgx2], { z31.h, z0.h }, z15.h
// CHECK-ENCODING: [0xef,0x77,0x6f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16f77ef <unknown>

sdot    za.s[w11, 7], {z31.h, z0.h}, z15.h  // 11000001-01101111-01110111-11101111
// CHECK-INST: sdot    za.s[w11, 7, vgx2], { z31.h, z0.h }, z15.h
// CHECK-ENCODING: [0xef,0x77,0x6f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16f77ef <unknown>

sdot    za.s[w8, 5, vgx2], {z17.h, z18.h}, z0.h  // 11000001-01100000-00010110-00101101
// CHECK-INST: sdot    za.s[w8, 5, vgx2], { z17.h, z18.h }, z0.h
// CHECK-ENCODING: [0x2d,0x16,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c160162d <unknown>

sdot    za.s[w8, 5], {z17.h, z18.h}, z0.h  // 11000001-01100000-00010110-00101101
// CHECK-INST: sdot    za.s[w8, 5, vgx2], { z17.h, z18.h }, z0.h
// CHECK-ENCODING: [0x2d,0x16,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c160162d <unknown>

sdot    za.s[w8, 1, vgx2], {z1.h, z2.h}, z14.h  // 11000001-01101110-00010100-00101001
// CHECK-INST: sdot    za.s[w8, 1, vgx2], { z1.h, z2.h }, z14.h
// CHECK-ENCODING: [0x29,0x14,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e1429 <unknown>

sdot    za.s[w8, 1], {z1.h, z2.h}, z14.h  // 11000001-01101110-00010100-00101001
// CHECK-INST: sdot    za.s[w8, 1, vgx2], { z1.h, z2.h }, z14.h
// CHECK-ENCODING: [0x29,0x14,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e1429 <unknown>

sdot    za.s[w10, 0, vgx2], {z19.h, z20.h}, z4.h  // 11000001-01100100-01010110-01101000
// CHECK-INST: sdot    za.s[w10, 0, vgx2], { z19.h, z20.h }, z4.h
// CHECK-ENCODING: [0x68,0x56,0x64,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1645668 <unknown>

sdot    za.s[w10, 0], {z19.h, z20.h}, z4.h  // 11000001-01100100-01010110-01101000
// CHECK-INST: sdot    za.s[w10, 0, vgx2], { z19.h, z20.h }, z4.h
// CHECK-ENCODING: [0x68,0x56,0x64,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1645668 <unknown>

sdot    za.s[w8, 0, vgx2], {z12.h, z13.h}, z2.h  // 11000001-01100010-00010101-10001000
// CHECK-INST: sdot    za.s[w8, 0, vgx2], { z12.h, z13.h }, z2.h
// CHECK-ENCODING: [0x88,0x15,0x62,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1621588 <unknown>

sdot    za.s[w8, 0], {z12.h, z13.h}, z2.h  // 11000001-01100010-00010101-10001000
// CHECK-INST: sdot    za.s[w8, 0, vgx2], { z12.h, z13.h }, z2.h
// CHECK-ENCODING: [0x88,0x15,0x62,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1621588 <unknown>

sdot    za.s[w10, 1, vgx2], {z1.h, z2.h}, z10.h  // 11000001-01101010-01010100-00101001
// CHECK-INST: sdot    za.s[w10, 1, vgx2], { z1.h, z2.h }, z10.h
// CHECK-ENCODING: [0x29,0x54,0x6a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16a5429 <unknown>

sdot    za.s[w10, 1], {z1.h, z2.h}, z10.h  // 11000001-01101010-01010100-00101001
// CHECK-INST: sdot    za.s[w10, 1, vgx2], { z1.h, z2.h }, z10.h
// CHECK-ENCODING: [0x29,0x54,0x6a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16a5429 <unknown>

sdot    za.s[w8, 5, vgx2], {z22.h, z23.h}, z14.h  // 11000001-01101110-00010110-11001101
// CHECK-INST: sdot    za.s[w8, 5, vgx2], { z22.h, z23.h }, z14.h
// CHECK-ENCODING: [0xcd,0x16,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e16cd <unknown>

sdot    za.s[w8, 5], {z22.h, z23.h}, z14.h  // 11000001-01101110-00010110-11001101
// CHECK-INST: sdot    za.s[w8, 5, vgx2], { z22.h, z23.h }, z14.h
// CHECK-ENCODING: [0xcd,0x16,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e16cd <unknown>

sdot    za.s[w11, 2, vgx2], {z9.h, z10.h}, z1.h  // 11000001-01100001-01110101-00101010
// CHECK-INST: sdot    za.s[w11, 2, vgx2], { z9.h, z10.h }, z1.h
// CHECK-ENCODING: [0x2a,0x75,0x61,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c161752a <unknown>

sdot    za.s[w11, 2], {z9.h, z10.h}, z1.h  // 11000001-01100001-01110101-00101010
// CHECK-INST: sdot    za.s[w11, 2, vgx2], { z9.h, z10.h }, z1.h
// CHECK-ENCODING: [0x2a,0x75,0x61,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c161752a <unknown>

sdot    za.s[w9, 7, vgx2], {z12.h, z13.h}, z11.h  // 11000001-01101011-00110101-10001111
// CHECK-INST: sdot    za.s[w9, 7, vgx2], { z12.h, z13.h }, z11.h
// CHECK-ENCODING: [0x8f,0x35,0x6b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16b358f <unknown>

sdot    za.s[w9, 7], {z12.h, z13.h}, z11.h  // 11000001-01101011-00110101-10001111
// CHECK-INST: sdot    za.s[w9, 7, vgx2], { z12.h, z13.h }, z11.h
// CHECK-ENCODING: [0x8f,0x35,0x6b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16b358f <unknown>


sdot    za.s[w8, 0, vgx2], {z0.h, z1.h}, z0.h[0]  // 11000001-01010000-00010000-00000000
// CHECK-INST: sdot    za.s[w8, 0, vgx2], { z0.h, z1.h }, z0.h[0]
// CHECK-ENCODING: [0x00,0x10,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1501000 <unknown>

sdot    za.s[w8, 0], {z0.h, z1.h}, z0.h[0]  // 11000001-01010000-00010000-00000000
// CHECK-INST: sdot    za.s[w8, 0, vgx2], { z0.h, z1.h }, z0.h[0]
// CHECK-ENCODING: [0x00,0x10,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1501000 <unknown>

sdot    za.s[w10, 5, vgx2], {z10.h, z11.h}, z5.h[1]  // 11000001-01010101-01010101-01000101
// CHECK-INST: sdot    za.s[w10, 5, vgx2], { z10.h, z11.h }, z5.h[1]
// CHECK-ENCODING: [0x45,0x55,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1555545 <unknown>

sdot    za.s[w10, 5], {z10.h, z11.h}, z5.h[1]  // 11000001-01010101-01010101-01000101
// CHECK-INST: sdot    za.s[w10, 5, vgx2], { z10.h, z11.h }, z5.h[1]
// CHECK-ENCODING: [0x45,0x55,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1555545 <unknown>

sdot    za.s[w11, 7, vgx2], {z12.h, z13.h}, z8.h[3]  // 11000001-01011000-01111101-10000111
// CHECK-INST: sdot    za.s[w11, 7, vgx2], { z12.h, z13.h }, z8.h[3]
// CHECK-ENCODING: [0x87,0x7d,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1587d87 <unknown>

sdot    za.s[w11, 7], {z12.h, z13.h}, z8.h[3]  // 11000001-01011000-01111101-10000111
// CHECK-INST: sdot    za.s[w11, 7, vgx2], { z12.h, z13.h }, z8.h[3]
// CHECK-ENCODING: [0x87,0x7d,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1587d87 <unknown>

sdot    za.s[w11, 7, vgx2], {z30.h, z31.h}, z15.h[3]  // 11000001-01011111-01111111-11000111
// CHECK-INST: sdot    za.s[w11, 7, vgx2], { z30.h, z31.h }, z15.h[3]
// CHECK-ENCODING: [0xc7,0x7f,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15f7fc7 <unknown>

sdot    za.s[w11, 7], {z30.h, z31.h}, z15.h[3]  // 11000001-01011111-01111111-11000111
// CHECK-INST: sdot    za.s[w11, 7, vgx2], { z30.h, z31.h }, z15.h[3]
// CHECK-ENCODING: [0xc7,0x7f,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15f7fc7 <unknown>

sdot    za.s[w8, 5, vgx2], {z16.h, z17.h}, z0.h[3]  // 11000001-01010000-00011110-00000101
// CHECK-INST: sdot    za.s[w8, 5, vgx2], { z16.h, z17.h }, z0.h[3]
// CHECK-ENCODING: [0x05,0x1e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1501e05 <unknown>

sdot    za.s[w8, 5], {z16.h, z17.h}, z0.h[3]  // 11000001-01010000-00011110-00000101
// CHECK-INST: sdot    za.s[w8, 5, vgx2], { z16.h, z17.h }, z0.h[3]
// CHECK-ENCODING: [0x05,0x1e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1501e05 <unknown>

sdot    za.s[w8, 1, vgx2], {z0.h, z1.h}, z14.h[1]  // 11000001-01011110-00010100-00000001
// CHECK-INST: sdot    za.s[w8, 1, vgx2], { z0.h, z1.h }, z14.h[1]
// CHECK-ENCODING: [0x01,0x14,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e1401 <unknown>

sdot    za.s[w8, 1], {z0.h, z1.h}, z14.h[1]  // 11000001-01011110-00010100-00000001
// CHECK-INST: sdot    za.s[w8, 1, vgx2], { z0.h, z1.h }, z14.h[1]
// CHECK-ENCODING: [0x01,0x14,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e1401 <unknown>

sdot    za.s[w10, 0, vgx2], {z18.h, z19.h}, z4.h[1]  // 11000001-01010100-01010110-01000000
// CHECK-INST: sdot    za.s[w10, 0, vgx2], { z18.h, z19.h }, z4.h[1]
// CHECK-ENCODING: [0x40,0x56,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1545640 <unknown>

sdot    za.s[w10, 0], {z18.h, z19.h}, z4.h[1]  // 11000001-01010100-01010110-01000000
// CHECK-INST: sdot    za.s[w10, 0, vgx2], { z18.h, z19.h }, z4.h[1]
// CHECK-ENCODING: [0x40,0x56,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1545640 <unknown>

sdot    za.s[w8, 0, vgx2], {z12.h, z13.h}, z2.h[2]  // 11000001-01010010-00011001-10000000
// CHECK-INST: sdot    za.s[w8, 0, vgx2], { z12.h, z13.h }, z2.h[2]
// CHECK-ENCODING: [0x80,0x19,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1521980 <unknown>

sdot    za.s[w8, 0], {z12.h, z13.h}, z2.h[2]  // 11000001-01010010-00011001-10000000
// CHECK-INST: sdot    za.s[w8, 0, vgx2], { z12.h, z13.h }, z2.h[2]
// CHECK-ENCODING: [0x80,0x19,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1521980 <unknown>

sdot    za.s[w10, 1, vgx2], {z0.h, z1.h}, z10.h[2]  // 11000001-01011010-01011000-00000001
// CHECK-INST: sdot    za.s[w10, 1, vgx2], { z0.h, z1.h }, z10.h[2]
// CHECK-ENCODING: [0x01,0x58,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15a5801 <unknown>

sdot    za.s[w10, 1], {z0.h, z1.h}, z10.h[2]  // 11000001-01011010-01011000-00000001
// CHECK-INST: sdot    za.s[w10, 1, vgx2], { z0.h, z1.h }, z10.h[2]
// CHECK-ENCODING: [0x01,0x58,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15a5801 <unknown>

sdot    za.s[w8, 5, vgx2], {z22.h, z23.h}, z14.h[2]  // 11000001-01011110-00011010-11000101
// CHECK-INST: sdot    za.s[w8, 5, vgx2], { z22.h, z23.h }, z14.h[2]
// CHECK-ENCODING: [0xc5,0x1a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e1ac5 <unknown>

sdot    za.s[w8, 5], {z22.h, z23.h}, z14.h[2]  // 11000001-01011110-00011010-11000101
// CHECK-INST: sdot    za.s[w8, 5, vgx2], { z22.h, z23.h }, z14.h[2]
// CHECK-ENCODING: [0xc5,0x1a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e1ac5 <unknown>

sdot    za.s[w11, 2, vgx2], {z8.h, z9.h}, z1.h[1]  // 11000001-01010001-01110101-00000010
// CHECK-INST: sdot    za.s[w11, 2, vgx2], { z8.h, z9.h }, z1.h[1]
// CHECK-ENCODING: [0x02,0x75,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1517502 <unknown>

sdot    za.s[w11, 2], {z8.h, z9.h}, z1.h[1]  // 11000001-01010001-01110101-00000010
// CHECK-INST: sdot    za.s[w11, 2, vgx2], { z8.h, z9.h }, z1.h[1]
// CHECK-ENCODING: [0x02,0x75,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1517502 <unknown>

sdot    za.s[w9, 7, vgx2], {z12.h, z13.h}, z11.h[2]  // 11000001-01011011-00111001-10000111
// CHECK-INST: sdot    za.s[w9, 7, vgx2], { z12.h, z13.h }, z11.h[2]
// CHECK-ENCODING: [0x87,0x39,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15b3987 <unknown>

sdot    za.s[w9, 7], {z12.h, z13.h}, z11.h[2]  // 11000001-01011011-00111001-10000111
// CHECK-INST: sdot    za.s[w9, 7, vgx2], { z12.h, z13.h }, z11.h[2]
// CHECK-ENCODING: [0x87,0x39,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15b3987 <unknown>


sdot    za.s[w8, 0, vgx2], {z0.h, z1.h}, {z0.h, z1.h}  // 11000001-11100000-00010100-00001000
// CHECK-INST: sdot    za.s[w8, 0, vgx2], { z0.h, z1.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x08,0x14,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e01408 <unknown>

sdot    za.s[w8, 0], {z0.h, z1.h}, {z0.h, z1.h}  // 11000001-11100000-00010100-00001000
// CHECK-INST: sdot    za.s[w8, 0, vgx2], { z0.h, z1.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x08,0x14,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e01408 <unknown>

sdot    za.s[w10, 5, vgx2], {z10.h, z11.h}, {z20.h, z21.h}  // 11000001-11110100-01010101-01001101
// CHECK-INST: sdot    za.s[w10, 5, vgx2], { z10.h, z11.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x4d,0x55,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f4554d <unknown>

sdot    za.s[w10, 5], {z10.h, z11.h}, {z20.h, z21.h}  // 11000001-11110100-01010101-01001101
// CHECK-INST: sdot    za.s[w10, 5, vgx2], { z10.h, z11.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x4d,0x55,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f4554d <unknown>

sdot    za.s[w11, 7, vgx2], {z12.h, z13.h}, {z8.h, z9.h}  // 11000001-11101000-01110101-10001111
// CHECK-INST: sdot    za.s[w11, 7, vgx2], { z12.h, z13.h }, { z8.h, z9.h }
// CHECK-ENCODING: [0x8f,0x75,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e8758f <unknown>

sdot    za.s[w11, 7], {z12.h, z13.h}, {z8.h, z9.h}  // 11000001-11101000-01110101-10001111
// CHECK-INST: sdot    za.s[w11, 7, vgx2], { z12.h, z13.h }, { z8.h, z9.h }
// CHECK-ENCODING: [0x8f,0x75,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e8758f <unknown>

sdot    za.s[w11, 7, vgx2], {z30.h, z31.h}, {z30.h, z31.h}  // 11000001-11111110-01110111-11001111
// CHECK-INST: sdot    za.s[w11, 7, vgx2], { z30.h, z31.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xcf,0x77,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe77cf <unknown>

sdot    za.s[w11, 7], {z30.h, z31.h}, {z30.h, z31.h}  // 11000001-11111110-01110111-11001111
// CHECK-INST: sdot    za.s[w11, 7, vgx2], { z30.h, z31.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xcf,0x77,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe77cf <unknown>

sdot    za.s[w8, 5, vgx2], {z16.h, z17.h}, {z16.h, z17.h}  // 11000001-11110000-00010110-00001101
// CHECK-INST: sdot    za.s[w8, 5, vgx2], { z16.h, z17.h }, { z16.h, z17.h }
// CHECK-ENCODING: [0x0d,0x16,0xf0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f0160d <unknown>

sdot    za.s[w8, 5], {z16.h, z17.h}, {z16.h, z17.h}  // 11000001-11110000-00010110-00001101
// CHECK-INST: sdot    za.s[w8, 5, vgx2], { z16.h, z17.h }, { z16.h, z17.h }
// CHECK-ENCODING: [0x0d,0x16,0xf0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f0160d <unknown>

sdot    za.s[w8, 1, vgx2], {z0.h, z1.h}, {z30.h, z31.h}  // 11000001-11111110-00010100-00001001
// CHECK-INST: sdot    za.s[w8, 1, vgx2], { z0.h, z1.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0x09,0x14,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe1409 <unknown>

sdot    za.s[w8, 1], {z0.h, z1.h}, {z30.h, z31.h}  // 11000001-11111110-00010100-00001001
// CHECK-INST: sdot    za.s[w8, 1, vgx2], { z0.h, z1.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0x09,0x14,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe1409 <unknown>

sdot    za.s[w10, 0, vgx2], {z18.h, z19.h}, {z20.h, z21.h}  // 11000001-11110100-01010110-01001000
// CHECK-INST: sdot    za.s[w10, 0, vgx2], { z18.h, z19.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x48,0x56,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f45648 <unknown>

sdot    za.s[w10, 0], {z18.h, z19.h}, {z20.h, z21.h}  // 11000001-11110100-01010110-01001000
// CHECK-INST: sdot    za.s[w10, 0, vgx2], { z18.h, z19.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x48,0x56,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f45648 <unknown>

sdot    za.s[w8, 0, vgx2], {z12.h, z13.h}, {z2.h, z3.h}  // 11000001-11100010-00010101-10001000
// CHECK-INST: sdot    za.s[w8, 0, vgx2], { z12.h, z13.h }, { z2.h, z3.h }
// CHECK-ENCODING: [0x88,0x15,0xe2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e21588 <unknown>

sdot    za.s[w8, 0], {z12.h, z13.h}, {z2.h, z3.h}  // 11000001-11100010-00010101-10001000
// CHECK-INST: sdot    za.s[w8, 0, vgx2], { z12.h, z13.h }, { z2.h, z3.h }
// CHECK-ENCODING: [0x88,0x15,0xe2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e21588 <unknown>

sdot    za.s[w10, 1, vgx2], {z0.h, z1.h}, {z26.h, z27.h}  // 11000001-11111010-01010100-00001001
// CHECK-INST: sdot    za.s[w10, 1, vgx2], { z0.h, z1.h }, { z26.h, z27.h }
// CHECK-ENCODING: [0x09,0x54,0xfa,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fa5409 <unknown>

sdot    za.s[w10, 1], {z0.h, z1.h}, {z26.h, z27.h}  // 11000001-11111010-01010100-00001001
// CHECK-INST: sdot    za.s[w10, 1, vgx2], { z0.h, z1.h }, { z26.h, z27.h }
// CHECK-ENCODING: [0x09,0x54,0xfa,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fa5409 <unknown>

sdot    za.s[w8, 5, vgx2], {z22.h, z23.h}, {z30.h, z31.h}  // 11000001-11111110-00010110-11001101
// CHECK-INST: sdot    za.s[w8, 5, vgx2], { z22.h, z23.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xcd,0x16,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe16cd <unknown>

sdot    za.s[w8, 5], {z22.h, z23.h}, {z30.h, z31.h}  // 11000001-11111110-00010110-11001101
// CHECK-INST: sdot    za.s[w8, 5, vgx2], { z22.h, z23.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xcd,0x16,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe16cd <unknown>

sdot    za.s[w11, 2, vgx2], {z8.h, z9.h}, {z0.h, z1.h}  // 11000001-11100000-01110101-00001010
// CHECK-INST: sdot    za.s[w11, 2, vgx2], { z8.h, z9.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x0a,0x75,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e0750a <unknown>

sdot    za.s[w11, 2], {z8.h, z9.h}, {z0.h, z1.h}  // 11000001-11100000-01110101-00001010
// CHECK-INST: sdot    za.s[w11, 2, vgx2], { z8.h, z9.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x0a,0x75,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e0750a <unknown>

sdot    za.s[w9, 7, vgx2], {z12.h, z13.h}, {z10.h, z11.h}  // 11000001-11101010-00110101-10001111
// CHECK-INST: sdot    za.s[w9, 7, vgx2], { z12.h, z13.h }, { z10.h, z11.h }
// CHECK-ENCODING: [0x8f,0x35,0xea,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ea358f <unknown>

sdot    za.s[w9, 7], {z12.h, z13.h}, {z10.h, z11.h}  // 11000001-11101010-00110101-10001111
// CHECK-INST: sdot    za.s[w9, 7, vgx2], { z12.h, z13.h }, { z10.h, z11.h }
// CHECK-ENCODING: [0x8f,0x35,0xea,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ea358f <unknown>



sdot    za.s[w8, 0, vgx2], {z0.b, z1.b}, z0.b  // 11000001-00100000-00010100-00000000
// CHECK-INST: sdot    za.s[w8, 0, vgx2], { z0.b, z1.b }, z0.b
// CHECK-ENCODING: [0x00,0x14,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1201400 <unknown>

sdot    za.s[w8, 0], {z0.b, z1.b}, z0.b  // 11000001-00100000-00010100-00000000
// CHECK-INST: sdot    za.s[w8, 0, vgx2], { z0.b, z1.b }, z0.b
// CHECK-ENCODING: [0x00,0x14,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1201400 <unknown>

sdot    za.s[w10, 5, vgx2], {z10.b, z11.b}, z5.b  // 11000001-00100101-01010101-01000101
// CHECK-INST: sdot    za.s[w10, 5, vgx2], { z10.b, z11.b }, z5.b
// CHECK-ENCODING: [0x45,0x55,0x25,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1255545 <unknown>

sdot    za.s[w10, 5], {z10.b, z11.b}, z5.b  // 11000001-00100101-01010101-01000101
// CHECK-INST: sdot    za.s[w10, 5, vgx2], { z10.b, z11.b }, z5.b
// CHECK-ENCODING: [0x45,0x55,0x25,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1255545 <unknown>

sdot    za.s[w11, 7, vgx2], {z13.b, z14.b}, z8.b  // 11000001-00101000-01110101-10100111
// CHECK-INST: sdot    za.s[w11, 7, vgx2], { z13.b, z14.b }, z8.b
// CHECK-ENCODING: [0xa7,0x75,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12875a7 <unknown>

sdot    za.s[w11, 7], {z13.b, z14.b}, z8.b  // 11000001-00101000-01110101-10100111
// CHECK-INST: sdot    za.s[w11, 7, vgx2], { z13.b, z14.b }, z8.b
// CHECK-ENCODING: [0xa7,0x75,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12875a7 <unknown>

sdot    za.s[w11, 7, vgx2], {z31.b, z0.b}, z15.b  // 11000001-00101111-01110111-11100111
// CHECK-INST: sdot    za.s[w11, 7, vgx2], { z31.b, z0.b }, z15.b
// CHECK-ENCODING: [0xe7,0x77,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12f77e7 <unknown>

sdot    za.s[w11, 7], {z31.b, z0.b}, z15.b  // 11000001-00101111-01110111-11100111
// CHECK-INST: sdot    za.s[w11, 7, vgx2], { z31.b, z0.b }, z15.b
// CHECK-ENCODING: [0xe7,0x77,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12f77e7 <unknown>

sdot    za.s[w8, 5, vgx2], {z17.b, z18.b}, z0.b  // 11000001-00100000-00010110-00100101
// CHECK-INST: sdot    za.s[w8, 5, vgx2], { z17.b, z18.b }, z0.b
// CHECK-ENCODING: [0x25,0x16,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1201625 <unknown>

sdot    za.s[w8, 5], {z17.b, z18.b}, z0.b  // 11000001-00100000-00010110-00100101
// CHECK-INST: sdot    za.s[w8, 5, vgx2], { z17.b, z18.b }, z0.b
// CHECK-ENCODING: [0x25,0x16,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1201625 <unknown>

sdot    za.s[w8, 1, vgx2], {z1.b, z2.b}, z14.b  // 11000001-00101110-00010100-00100001
// CHECK-INST: sdot    za.s[w8, 1, vgx2], { z1.b, z2.b }, z14.b
// CHECK-ENCODING: [0x21,0x14,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e1421 <unknown>

sdot    za.s[w8, 1], {z1.b, z2.b}, z14.b  // 11000001-00101110-00010100-00100001
// CHECK-INST: sdot    za.s[w8, 1, vgx2], { z1.b, z2.b }, z14.b
// CHECK-ENCODING: [0x21,0x14,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e1421 <unknown>

sdot    za.s[w10, 0, vgx2], {z19.b, z20.b}, z4.b  // 11000001-00100100-01010110-01100000
// CHECK-INST: sdot    za.s[w10, 0, vgx2], { z19.b, z20.b }, z4.b
// CHECK-ENCODING: [0x60,0x56,0x24,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1245660 <unknown>

sdot    za.s[w10, 0], {z19.b, z20.b}, z4.b  // 11000001-00100100-01010110-01100000
// CHECK-INST: sdot    za.s[w10, 0, vgx2], { z19.b, z20.b }, z4.b
// CHECK-ENCODING: [0x60,0x56,0x24,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1245660 <unknown>

sdot    za.s[w8, 0, vgx2], {z12.b, z13.b}, z2.b  // 11000001-00100010-00010101-10000000
// CHECK-INST: sdot    za.s[w8, 0, vgx2], { z12.b, z13.b }, z2.b
// CHECK-ENCODING: [0x80,0x15,0x22,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1221580 <unknown>

sdot    za.s[w8, 0], {z12.b, z13.b}, z2.b  // 11000001-00100010-00010101-10000000
// CHECK-INST: sdot    za.s[w8, 0, vgx2], { z12.b, z13.b }, z2.b
// CHECK-ENCODING: [0x80,0x15,0x22,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1221580 <unknown>

sdot    za.s[w10, 1, vgx2], {z1.b, z2.b}, z10.b  // 11000001-00101010-01010100-00100001
// CHECK-INST: sdot    za.s[w10, 1, vgx2], { z1.b, z2.b }, z10.b
// CHECK-ENCODING: [0x21,0x54,0x2a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12a5421 <unknown>

sdot    za.s[w10, 1], {z1.b, z2.b}, z10.b  // 11000001-00101010-01010100-00100001
// CHECK-INST: sdot    za.s[w10, 1, vgx2], { z1.b, z2.b }, z10.b
// CHECK-ENCODING: [0x21,0x54,0x2a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12a5421 <unknown>

sdot    za.s[w8, 5, vgx2], {z22.b, z23.b}, z14.b  // 11000001-00101110-00010110-11000101
// CHECK-INST: sdot    za.s[w8, 5, vgx2], { z22.b, z23.b }, z14.b
// CHECK-ENCODING: [0xc5,0x16,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e16c5 <unknown>

sdot    za.s[w8, 5], {z22.b, z23.b}, z14.b  // 11000001-00101110-00010110-11000101
// CHECK-INST: sdot    za.s[w8, 5, vgx2], { z22.b, z23.b }, z14.b
// CHECK-ENCODING: [0xc5,0x16,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e16c5 <unknown>

sdot    za.s[w11, 2, vgx2], {z9.b, z10.b}, z1.b  // 11000001-00100001-01110101-00100010
// CHECK-INST: sdot    za.s[w11, 2, vgx2], { z9.b, z10.b }, z1.b
// CHECK-ENCODING: [0x22,0x75,0x21,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1217522 <unknown>

sdot    za.s[w11, 2], {z9.b, z10.b}, z1.b  // 11000001-00100001-01110101-00100010
// CHECK-INST: sdot    za.s[w11, 2, vgx2], { z9.b, z10.b }, z1.b
// CHECK-ENCODING: [0x22,0x75,0x21,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1217522 <unknown>

sdot    za.s[w9, 7, vgx2], {z12.b, z13.b}, z11.b  // 11000001-00101011-00110101-10000111
// CHECK-INST: sdot    za.s[w9, 7, vgx2], { z12.b, z13.b }, z11.b
// CHECK-ENCODING: [0x87,0x35,0x2b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12b3587 <unknown>

sdot    za.s[w9, 7], {z12.b, z13.b}, z11.b  // 11000001-00101011-00110101-10000111
// CHECK-INST: sdot    za.s[w9, 7, vgx2], { z12.b, z13.b }, z11.b
// CHECK-ENCODING: [0x87,0x35,0x2b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12b3587 <unknown>


sdot    za.s[w8, 0, vgx2], {z0.b, z1.b}, z0.b[0]  // 11000001-01010000-00010000-00100000
// CHECK-INST: sdot    za.s[w8, 0, vgx2], { z0.b, z1.b }, z0.b[0]
// CHECK-ENCODING: [0x20,0x10,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1501020 <unknown>

sdot    za.s[w8, 0], {z0.b, z1.b}, z0.b[0]  // 11000001-01010000-00010000-00100000
// CHECK-INST: sdot    za.s[w8, 0, vgx2], { z0.b, z1.b }, z0.b[0]
// CHECK-ENCODING: [0x20,0x10,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1501020 <unknown>

sdot    za.s[w10, 5, vgx2], {z10.b, z11.b}, z5.b[1]  // 11000001-01010101-01010101-01100101
// CHECK-INST: sdot    za.s[w10, 5, vgx2], { z10.b, z11.b }, z5.b[1]
// CHECK-ENCODING: [0x65,0x55,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1555565 <unknown>

sdot    za.s[w10, 5], {z10.b, z11.b}, z5.b[1]  // 11000001-01010101-01010101-01100101
// CHECK-INST: sdot    za.s[w10, 5, vgx2], { z10.b, z11.b }, z5.b[1]
// CHECK-ENCODING: [0x65,0x55,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1555565 <unknown>

sdot    za.s[w11, 7, vgx2], {z12.b, z13.b}, z8.b[3]  // 11000001-01011000-01111101-10100111
// CHECK-INST: sdot    za.s[w11, 7, vgx2], { z12.b, z13.b }, z8.b[3]
// CHECK-ENCODING: [0xa7,0x7d,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1587da7 <unknown>

sdot    za.s[w11, 7], {z12.b, z13.b}, z8.b[3]  // 11000001-01011000-01111101-10100111
// CHECK-INST: sdot    za.s[w11, 7, vgx2], { z12.b, z13.b }, z8.b[3]
// CHECK-ENCODING: [0xa7,0x7d,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1587da7 <unknown>

sdot    za.s[w11, 7, vgx2], {z30.b, z31.b}, z15.b[3]  // 11000001-01011111-01111111-11100111
// CHECK-INST: sdot    za.s[w11, 7, vgx2], { z30.b, z31.b }, z15.b[3]
// CHECK-ENCODING: [0xe7,0x7f,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15f7fe7 <unknown>

sdot    za.s[w11, 7], {z30.b, z31.b}, z15.b[3]  // 11000001-01011111-01111111-11100111
// CHECK-INST: sdot    za.s[w11, 7, vgx2], { z30.b, z31.b }, z15.b[3]
// CHECK-ENCODING: [0xe7,0x7f,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15f7fe7 <unknown>

sdot    za.s[w8, 5, vgx2], {z16.b, z17.b}, z0.b[3]  // 11000001-01010000-00011110-00100101
// CHECK-INST: sdot    za.s[w8, 5, vgx2], { z16.b, z17.b }, z0.b[3]
// CHECK-ENCODING: [0x25,0x1e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1501e25 <unknown>

sdot    za.s[w8, 5], {z16.b, z17.b}, z0.b[3]  // 11000001-01010000-00011110-00100101
// CHECK-INST: sdot    za.s[w8, 5, vgx2], { z16.b, z17.b }, z0.b[3]
// CHECK-ENCODING: [0x25,0x1e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1501e25 <unknown>

sdot    za.s[w8, 1, vgx2], {z0.b, z1.b}, z14.b[1]  // 11000001-01011110-00010100-00100001
// CHECK-INST: sdot    za.s[w8, 1, vgx2], { z0.b, z1.b }, z14.b[1]
// CHECK-ENCODING: [0x21,0x14,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e1421 <unknown>

sdot    za.s[w8, 1], {z0.b, z1.b}, z14.b[1]  // 11000001-01011110-00010100-00100001
// CHECK-INST: sdot    za.s[w8, 1, vgx2], { z0.b, z1.b }, z14.b[1]
// CHECK-ENCODING: [0x21,0x14,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e1421 <unknown>

sdot    za.s[w10, 0, vgx2], {z18.b, z19.b}, z4.b[1]  // 11000001-01010100-01010110-01100000
// CHECK-INST: sdot    za.s[w10, 0, vgx2], { z18.b, z19.b }, z4.b[1]
// CHECK-ENCODING: [0x60,0x56,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1545660 <unknown>

sdot    za.s[w10, 0], {z18.b, z19.b}, z4.b[1]  // 11000001-01010100-01010110-01100000
// CHECK-INST: sdot    za.s[w10, 0, vgx2], { z18.b, z19.b }, z4.b[1]
// CHECK-ENCODING: [0x60,0x56,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1545660 <unknown>

sdot    za.s[w8, 0, vgx2], {z12.b, z13.b}, z2.b[2]  // 11000001-01010010-00011001-10100000
// CHECK-INST: sdot    za.s[w8, 0, vgx2], { z12.b, z13.b }, z2.b[2]
// CHECK-ENCODING: [0xa0,0x19,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15219a0 <unknown>

sdot    za.s[w8, 0], {z12.b, z13.b}, z2.b[2]  // 11000001-01010010-00011001-10100000
// CHECK-INST: sdot    za.s[w8, 0, vgx2], { z12.b, z13.b }, z2.b[2]
// CHECK-ENCODING: [0xa0,0x19,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15219a0 <unknown>

sdot    za.s[w10, 1, vgx2], {z0.b, z1.b}, z10.b[2]  // 11000001-01011010-01011000-00100001
// CHECK-INST: sdot    za.s[w10, 1, vgx2], { z0.b, z1.b }, z10.b[2]
// CHECK-ENCODING: [0x21,0x58,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15a5821 <unknown>

sdot    za.s[w10, 1], {z0.b, z1.b}, z10.b[2]  // 11000001-01011010-01011000-00100001
// CHECK-INST: sdot    za.s[w10, 1, vgx2], { z0.b, z1.b }, z10.b[2]
// CHECK-ENCODING: [0x21,0x58,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15a5821 <unknown>

sdot    za.s[w8, 5, vgx2], {z22.b, z23.b}, z14.b[2]  // 11000001-01011110-00011010-11100101
// CHECK-INST: sdot    za.s[w8, 5, vgx2], { z22.b, z23.b }, z14.b[2]
// CHECK-ENCODING: [0xe5,0x1a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e1ae5 <unknown>

sdot    za.s[w8, 5], {z22.b, z23.b}, z14.b[2]  // 11000001-01011110-00011010-11100101
// CHECK-INST: sdot    za.s[w8, 5, vgx2], { z22.b, z23.b }, z14.b[2]
// CHECK-ENCODING: [0xe5,0x1a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e1ae5 <unknown>

sdot    za.s[w11, 2, vgx2], {z8.b, z9.b}, z1.b[1]  // 11000001-01010001-01110101-00100010
// CHECK-INST: sdot    za.s[w11, 2, vgx2], { z8.b, z9.b }, z1.b[1]
// CHECK-ENCODING: [0x22,0x75,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1517522 <unknown>

sdot    za.s[w11, 2], {z8.b, z9.b}, z1.b[1]  // 11000001-01010001-01110101-00100010
// CHECK-INST: sdot    za.s[w11, 2, vgx2], { z8.b, z9.b }, z1.b[1]
// CHECK-ENCODING: [0x22,0x75,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1517522 <unknown>

sdot    za.s[w9, 7, vgx2], {z12.b, z13.b}, z11.b[2]  // 11000001-01011011-00111001-10100111
// CHECK-INST: sdot    za.s[w9, 7, vgx2], { z12.b, z13.b }, z11.b[2]
// CHECK-ENCODING: [0xa7,0x39,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15b39a7 <unknown>

sdot    za.s[w9, 7], {z12.b, z13.b}, z11.b[2]  // 11000001-01011011-00111001-10100111
// CHECK-INST: sdot    za.s[w9, 7, vgx2], { z12.b, z13.b }, z11.b[2]
// CHECK-ENCODING: [0xa7,0x39,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15b39a7 <unknown>



sdot    za.s[w8, 0, vgx2], {z0.b, z1.b}, {z0.b, z1.b}  // 11000001-10100000-00010100-00000000
// CHECK-INST: sdot    za.s[w8, 0, vgx2], { z0.b, z1.b }, { z0.b, z1.b }
// CHECK-ENCODING: [0x00,0x14,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a01400 <unknown>

sdot    za.s[w8, 0], {z0.b, z1.b}, {z0.b, z1.b}  // 11000001-10100000-00010100-00000000
// CHECK-INST: sdot    za.s[w8, 0, vgx2], { z0.b, z1.b }, { z0.b, z1.b }
// CHECK-ENCODING: [0x00,0x14,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a01400 <unknown>

sdot    za.s[w10, 5, vgx2], {z10.b, z11.b}, {z20.b, z21.b}  // 11000001-10110100-01010101-01000101
// CHECK-INST: sdot    za.s[w10, 5, vgx2], { z10.b, z11.b }, { z20.b, z21.b }
// CHECK-ENCODING: [0x45,0x55,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b45545 <unknown>

sdot    za.s[w10, 5], {z10.b, z11.b}, {z20.b, z21.b}  // 11000001-10110100-01010101-01000101
// CHECK-INST: sdot    za.s[w10, 5, vgx2], { z10.b, z11.b }, { z20.b, z21.b }
// CHECK-ENCODING: [0x45,0x55,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b45545 <unknown>

sdot    za.s[w11, 7, vgx2], {z12.b, z13.b}, {z8.b, z9.b}  // 11000001-10101000-01110101-10000111
// CHECK-INST: sdot    za.s[w11, 7, vgx2], { z12.b, z13.b }, { z8.b, z9.b }
// CHECK-ENCODING: [0x87,0x75,0xa8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a87587 <unknown>

sdot    za.s[w11, 7], {z12.b, z13.b}, {z8.b, z9.b}  // 11000001-10101000-01110101-10000111
// CHECK-INST: sdot    za.s[w11, 7, vgx2], { z12.b, z13.b }, { z8.b, z9.b }
// CHECK-ENCODING: [0x87,0x75,0xa8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a87587 <unknown>

sdot    za.s[w11, 7, vgx2], {z30.b, z31.b}, {z30.b, z31.b}  // 11000001-10111110-01110111-11000111
// CHECK-INST: sdot    za.s[w11, 7, vgx2], { z30.b, z31.b }, { z30.b, z31.b }
// CHECK-ENCODING: [0xc7,0x77,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be77c7 <unknown>

sdot    za.s[w11, 7], {z30.b, z31.b}, {z30.b, z31.b}  // 11000001-10111110-01110111-11000111
// CHECK-INST: sdot    za.s[w11, 7, vgx2], { z30.b, z31.b }, { z30.b, z31.b }
// CHECK-ENCODING: [0xc7,0x77,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be77c7 <unknown>

sdot    za.s[w8, 5, vgx2], {z16.b, z17.b}, {z16.b, z17.b}  // 11000001-10110000-00010110-00000101
// CHECK-INST: sdot    za.s[w8, 5, vgx2], { z16.b, z17.b }, { z16.b, z17.b }
// CHECK-ENCODING: [0x05,0x16,0xb0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b01605 <unknown>

sdot    za.s[w8, 5], {z16.b, z17.b}, {z16.b, z17.b}  // 11000001-10110000-00010110-00000101
// CHECK-INST: sdot    za.s[w8, 5, vgx2], { z16.b, z17.b }, { z16.b, z17.b }
// CHECK-ENCODING: [0x05,0x16,0xb0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b01605 <unknown>

sdot    za.s[w8, 1, vgx2], {z0.b, z1.b}, {z30.b, z31.b}  // 11000001-10111110-00010100-00000001
// CHECK-INST: sdot    za.s[w8, 1, vgx2], { z0.b, z1.b }, { z30.b, z31.b }
// CHECK-ENCODING: [0x01,0x14,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be1401 <unknown>

sdot    za.s[w8, 1], {z0.b, z1.b}, {z30.b, z31.b}  // 11000001-10111110-00010100-00000001
// CHECK-INST: sdot    za.s[w8, 1, vgx2], { z0.b, z1.b }, { z30.b, z31.b }
// CHECK-ENCODING: [0x01,0x14,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be1401 <unknown>

sdot    za.s[w10, 0, vgx2], {z18.b, z19.b}, {z20.b, z21.b}  // 11000001-10110100-01010110-01000000
// CHECK-INST: sdot    za.s[w10, 0, vgx2], { z18.b, z19.b }, { z20.b, z21.b }
// CHECK-ENCODING: [0x40,0x56,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b45640 <unknown>

sdot    za.s[w10, 0], {z18.b, z19.b}, {z20.b, z21.b}  // 11000001-10110100-01010110-01000000
// CHECK-INST: sdot    za.s[w10, 0, vgx2], { z18.b, z19.b }, { z20.b, z21.b }
// CHECK-ENCODING: [0x40,0x56,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b45640 <unknown>

sdot    za.s[w8, 0, vgx2], {z12.b, z13.b}, {z2.b, z3.b}  // 11000001-10100010-00010101-10000000
// CHECK-INST: sdot    za.s[w8, 0, vgx2], { z12.b, z13.b }, { z2.b, z3.b }
// CHECK-ENCODING: [0x80,0x15,0xa2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a21580 <unknown>

sdot    za.s[w8, 0], {z12.b, z13.b}, {z2.b, z3.b}  // 11000001-10100010-00010101-10000000
// CHECK-INST: sdot    za.s[w8, 0, vgx2], { z12.b, z13.b }, { z2.b, z3.b }
// CHECK-ENCODING: [0x80,0x15,0xa2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a21580 <unknown>

sdot    za.s[w10, 1, vgx2], {z0.b, z1.b}, {z26.b, z27.b}  // 11000001-10111010-01010100-00000001
// CHECK-INST: sdot    za.s[w10, 1, vgx2], { z0.b, z1.b }, { z26.b, z27.b }
// CHECK-ENCODING: [0x01,0x54,0xba,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ba5401 <unknown>

sdot    za.s[w10, 1], {z0.b, z1.b}, {z26.b, z27.b}  // 11000001-10111010-01010100-00000001
// CHECK-INST: sdot    za.s[w10, 1, vgx2], { z0.b, z1.b }, { z26.b, z27.b }
// CHECK-ENCODING: [0x01,0x54,0xba,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ba5401 <unknown>

sdot    za.s[w8, 5, vgx2], {z22.b, z23.b}, {z30.b, z31.b}  // 11000001-10111110-00010110-11000101
// CHECK-INST: sdot    za.s[w8, 5, vgx2], { z22.b, z23.b }, { z30.b, z31.b }
// CHECK-ENCODING: [0xc5,0x16,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be16c5 <unknown>

sdot    za.s[w8, 5], {z22.b, z23.b}, {z30.b, z31.b}  // 11000001-10111110-00010110-11000101
// CHECK-INST: sdot    za.s[w8, 5, vgx2], { z22.b, z23.b }, { z30.b, z31.b }
// CHECK-ENCODING: [0xc5,0x16,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be16c5 <unknown>

sdot    za.s[w11, 2, vgx2], {z8.b, z9.b}, {z0.b, z1.b}  // 11000001-10100000-01110101-00000010
// CHECK-INST: sdot    za.s[w11, 2, vgx2], { z8.b, z9.b }, { z0.b, z1.b }
// CHECK-ENCODING: [0x02,0x75,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a07502 <unknown>

sdot    za.s[w11, 2], {z8.b, z9.b}, {z0.b, z1.b}  // 11000001-10100000-01110101-00000010
// CHECK-INST: sdot    za.s[w11, 2, vgx2], { z8.b, z9.b }, { z0.b, z1.b }
// CHECK-ENCODING: [0x02,0x75,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a07502 <unknown>

sdot    za.s[w9, 7, vgx2], {z12.b, z13.b}, {z10.b, z11.b}  // 11000001-10101010-00110101-10000111
// CHECK-INST: sdot    za.s[w9, 7, vgx2], { z12.b, z13.b }, { z10.b, z11.b }
// CHECK-ENCODING: [0x87,0x35,0xaa,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1aa3587 <unknown>

sdot    za.s[w9, 7], {z12.b, z13.b}, {z10.b, z11.b}  // 11000001-10101010-00110101-10000111
// CHECK-INST: sdot    za.s[w9, 7, vgx2], { z12.b, z13.b }, { z10.b, z11.b }
// CHECK-ENCODING: [0x87,0x35,0xaa,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1aa3587 <unknown>


sdot    za.d[w8, 0, vgx2], {z0.h, z1.h}, z0.h  // 11000001-01100000-00010100-00000000
// CHECK-INST: sdot    za.d[w8, 0, vgx2], { z0.h, z1.h }, z0.h
// CHECK-ENCODING: [0x00,0x14,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1601400 <unknown>

sdot    za.d[w8, 0], {z0.h, z1.h}, z0.h  // 11000001-01100000-00010100-00000000
// CHECK-INST: sdot    za.d[w8, 0, vgx2], { z0.h, z1.h }, z0.h
// CHECK-ENCODING: [0x00,0x14,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1601400 <unknown>

sdot    za.d[w10, 5, vgx2], {z10.h, z11.h}, z5.h  // 11000001-01100101-01010101-01000101
// CHECK-INST: sdot    za.d[w10, 5, vgx2], { z10.h, z11.h }, z5.h
// CHECK-ENCODING: [0x45,0x55,0x65,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1655545 <unknown>

sdot    za.d[w10, 5], {z10.h, z11.h}, z5.h  // 11000001-01100101-01010101-01000101
// CHECK-INST: sdot    za.d[w10, 5, vgx2], { z10.h, z11.h }, z5.h
// CHECK-ENCODING: [0x45,0x55,0x65,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1655545 <unknown>

sdot    za.d[w11, 7, vgx2], {z13.h, z14.h}, z8.h  // 11000001-01101000-01110101-10100111
// CHECK-INST: sdot    za.d[w11, 7, vgx2], { z13.h, z14.h }, z8.h
// CHECK-ENCODING: [0xa7,0x75,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16875a7 <unknown>

sdot    za.d[w11, 7], {z13.h, z14.h}, z8.h  // 11000001-01101000-01110101-10100111
// CHECK-INST: sdot    za.d[w11, 7, vgx2], { z13.h, z14.h }, z8.h
// CHECK-ENCODING: [0xa7,0x75,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16875a7 <unknown>

sdot    za.d[w11, 7, vgx2], {z31.h, z0.h}, z15.h  // 11000001-01101111-01110111-11100111
// CHECK-INST: sdot    za.d[w11, 7, vgx2], { z31.h, z0.h }, z15.h
// CHECK-ENCODING: [0xe7,0x77,0x6f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16f77e7 <unknown>

sdot    za.d[w11, 7], {z31.h, z0.h}, z15.h  // 11000001-01101111-01110111-11100111
// CHECK-INST: sdot    za.d[w11, 7, vgx2], { z31.h, z0.h }, z15.h
// CHECK-ENCODING: [0xe7,0x77,0x6f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16f77e7 <unknown>

sdot    za.d[w8, 5, vgx2], {z17.h, z18.h}, z0.h  // 11000001-01100000-00010110-00100101
// CHECK-INST: sdot    za.d[w8, 5, vgx2], { z17.h, z18.h }, z0.h
// CHECK-ENCODING: [0x25,0x16,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1601625 <unknown>

sdot    za.d[w8, 5], {z17.h, z18.h}, z0.h  // 11000001-01100000-00010110-00100101
// CHECK-INST: sdot    za.d[w8, 5, vgx2], { z17.h, z18.h }, z0.h
// CHECK-ENCODING: [0x25,0x16,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1601625 <unknown>

sdot    za.d[w8, 1, vgx2], {z1.h, z2.h}, z14.h  // 11000001-01101110-00010100-00100001
// CHECK-INST: sdot    za.d[w8, 1, vgx2], { z1.h, z2.h }, z14.h
// CHECK-ENCODING: [0x21,0x14,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e1421 <unknown>

sdot    za.d[w8, 1], {z1.h, z2.h}, z14.h  // 11000001-01101110-00010100-00100001
// CHECK-INST: sdot    za.d[w8, 1, vgx2], { z1.h, z2.h }, z14.h
// CHECK-ENCODING: [0x21,0x14,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e1421 <unknown>

sdot    za.d[w10, 0, vgx2], {z19.h, z20.h}, z4.h  // 11000001-01100100-01010110-01100000
// CHECK-INST: sdot    za.d[w10, 0, vgx2], { z19.h, z20.h }, z4.h
// CHECK-ENCODING: [0x60,0x56,0x64,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1645660 <unknown>

sdot    za.d[w10, 0], {z19.h, z20.h}, z4.h  // 11000001-01100100-01010110-01100000
// CHECK-INST: sdot    za.d[w10, 0, vgx2], { z19.h, z20.h }, z4.h
// CHECK-ENCODING: [0x60,0x56,0x64,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1645660 <unknown>

sdot    za.d[w8, 0, vgx2], {z12.h, z13.h}, z2.h  // 11000001-01100010-00010101-10000000
// CHECK-INST: sdot    za.d[w8, 0, vgx2], { z12.h, z13.h }, z2.h
// CHECK-ENCODING: [0x80,0x15,0x62,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1621580 <unknown>

sdot    za.d[w8, 0], {z12.h, z13.h}, z2.h  // 11000001-01100010-00010101-10000000
// CHECK-INST: sdot    za.d[w8, 0, vgx2], { z12.h, z13.h }, z2.h
// CHECK-ENCODING: [0x80,0x15,0x62,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1621580 <unknown>

sdot    za.d[w10, 1, vgx2], {z1.h, z2.h}, z10.h  // 11000001-01101010-01010100-00100001
// CHECK-INST: sdot    za.d[w10, 1, vgx2], { z1.h, z2.h }, z10.h
// CHECK-ENCODING: [0x21,0x54,0x6a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16a5421 <unknown>

sdot    za.d[w10, 1], {z1.h, z2.h}, z10.h  // 11000001-01101010-01010100-00100001
// CHECK-INST: sdot    za.d[w10, 1, vgx2], { z1.h, z2.h }, z10.h
// CHECK-ENCODING: [0x21,0x54,0x6a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16a5421 <unknown>

sdot    za.d[w8, 5, vgx2], {z22.h, z23.h}, z14.h  // 11000001-01101110-00010110-11000101
// CHECK-INST: sdot    za.d[w8, 5, vgx2], { z22.h, z23.h }, z14.h
// CHECK-ENCODING: [0xc5,0x16,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e16c5 <unknown>

sdot    za.d[w8, 5], {z22.h, z23.h}, z14.h  // 11000001-01101110-00010110-11000101
// CHECK-INST: sdot    za.d[w8, 5, vgx2], { z22.h, z23.h }, z14.h
// CHECK-ENCODING: [0xc5,0x16,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e16c5 <unknown>

sdot    za.d[w11, 2, vgx2], {z9.h, z10.h}, z1.h  // 11000001-01100001-01110101-00100010
// CHECK-INST: sdot    za.d[w11, 2, vgx2], { z9.h, z10.h }, z1.h
// CHECK-ENCODING: [0x22,0x75,0x61,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1617522 <unknown>

sdot    za.d[w11, 2], {z9.h, z10.h}, z1.h  // 11000001-01100001-01110101-00100010
// CHECK-INST: sdot    za.d[w11, 2, vgx2], { z9.h, z10.h }, z1.h
// CHECK-ENCODING: [0x22,0x75,0x61,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1617522 <unknown>

sdot    za.d[w9, 7, vgx2], {z12.h, z13.h}, z11.h  // 11000001-01101011-00110101-10000111
// CHECK-INST: sdot    za.d[w9, 7, vgx2], { z12.h, z13.h }, z11.h
// CHECK-ENCODING: [0x87,0x35,0x6b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16b3587 <unknown>

sdot    za.d[w9, 7], {z12.h, z13.h}, z11.h  // 11000001-01101011-00110101-10000111
// CHECK-INST: sdot    za.d[w9, 7, vgx2], { z12.h, z13.h }, z11.h
// CHECK-ENCODING: [0x87,0x35,0x6b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16b3587 <unknown>


sdot    za.d[w8, 0, vgx2], {z0.h, z1.h}, z0.h[0]  // 11000001-11010000-00000000-00001000
// CHECK-INST: sdot    za.d[w8, 0, vgx2], { z0.h, z1.h }, z0.h[0]
// CHECK-ENCODING: [0x08,0x00,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d00008 <unknown>

sdot    za.d[w8, 0], {z0.h, z1.h}, z0.h[0]  // 11000001-11010000-00000000-00001000
// CHECK-INST: sdot    za.d[w8, 0, vgx2], { z0.h, z1.h }, z0.h[0]
// CHECK-ENCODING: [0x08,0x00,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d00008 <unknown>

sdot    za.d[w10, 5, vgx2], {z10.h, z11.h}, z5.h[1]  // 11000001-11010101-01000101-01001101
// CHECK-INST: sdot    za.d[w10, 5, vgx2], { z10.h, z11.h }, z5.h[1]
// CHECK-ENCODING: [0x4d,0x45,0xd5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d5454d <unknown>

sdot    za.d[w10, 5], {z10.h, z11.h}, z5.h[1]  // 11000001-11010101-01000101-01001101
// CHECK-INST: sdot    za.d[w10, 5, vgx2], { z10.h, z11.h }, z5.h[1]
// CHECK-ENCODING: [0x4d,0x45,0xd5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d5454d <unknown>

sdot    za.d[w11, 7, vgx2], {z12.h, z13.h}, z8.h[1]  // 11000001-11011000-01100101-10001111
// CHECK-INST: sdot    za.d[w11, 7, vgx2], { z12.h, z13.h }, z8.h[1]
// CHECK-ENCODING: [0x8f,0x65,0xd8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d8658f <unknown>

sdot    za.d[w11, 7], {z12.h, z13.h}, z8.h[1]  // 11000001-11011000-01100101-10001111
// CHECK-INST: sdot    za.d[w11, 7, vgx2], { z12.h, z13.h }, z8.h[1]
// CHECK-ENCODING: [0x8f,0x65,0xd8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d8658f <unknown>

sdot    za.d[w11, 7, vgx2], {z30.h, z31.h}, z15.h[1]  // 11000001-11011111-01100111-11001111
// CHECK-INST: sdot    za.d[w11, 7, vgx2], { z30.h, z31.h }, z15.h[1]
// CHECK-ENCODING: [0xcf,0x67,0xdf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1df67cf <unknown>

sdot    za.d[w11, 7], {z30.h, z31.h}, z15.h[1]  // 11000001-11011111-01100111-11001111
// CHECK-INST: sdot    za.d[w11, 7, vgx2], { z30.h, z31.h }, z15.h[1]
// CHECK-ENCODING: [0xcf,0x67,0xdf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1df67cf <unknown>

sdot    za.d[w8, 5, vgx2], {z16.h, z17.h}, z0.h[1]  // 11000001-11010000-00000110-00001101
// CHECK-INST: sdot    za.d[w8, 5, vgx2], { z16.h, z17.h }, z0.h[1]
// CHECK-ENCODING: [0x0d,0x06,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d0060d <unknown>

sdot    za.d[w8, 5], {z16.h, z17.h}, z0.h[1]  // 11000001-11010000-00000110-00001101
// CHECK-INST: sdot    za.d[w8, 5, vgx2], { z16.h, z17.h }, z0.h[1]
// CHECK-ENCODING: [0x0d,0x06,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d0060d <unknown>

sdot    za.d[w8, 1, vgx2], {z0.h, z1.h}, z14.h[1]  // 11000001-11011110-00000100-00001001
// CHECK-INST: sdot    za.d[w8, 1, vgx2], { z0.h, z1.h }, z14.h[1]
// CHECK-ENCODING: [0x09,0x04,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de0409 <unknown>

sdot    za.d[w8, 1], {z0.h, z1.h}, z14.h[1]  // 11000001-11011110-00000100-00001001
// CHECK-INST: sdot    za.d[w8, 1, vgx2], { z0.h, z1.h }, z14.h[1]
// CHECK-ENCODING: [0x09,0x04,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de0409 <unknown>

sdot    za.d[w10, 0, vgx2], {z18.h, z19.h}, z4.h[1]  // 11000001-11010100-01000110-01001000
// CHECK-INST: sdot    za.d[w10, 0, vgx2], { z18.h, z19.h }, z4.h[1]
// CHECK-ENCODING: [0x48,0x46,0xd4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d44648 <unknown>

sdot    za.d[w10, 0], {z18.h, z19.h}, z4.h[1]  // 11000001-11010100-01000110-01001000
// CHECK-INST: sdot    za.d[w10, 0, vgx2], { z18.h, z19.h }, z4.h[1]
// CHECK-ENCODING: [0x48,0x46,0xd4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d44648 <unknown>

sdot    za.d[w8, 0, vgx2], {z12.h, z13.h}, z2.h[0]  // 11000001-11010010-00000001-10001000
// CHECK-INST: sdot    za.d[w8, 0, vgx2], { z12.h, z13.h }, z2.h[0]
// CHECK-ENCODING: [0x88,0x01,0xd2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d20188 <unknown>

sdot    za.d[w8, 0], {z12.h, z13.h}, z2.h[0]  // 11000001-11010010-00000001-10001000
// CHECK-INST: sdot    za.d[w8, 0, vgx2], { z12.h, z13.h }, z2.h[0]
// CHECK-ENCODING: [0x88,0x01,0xd2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d20188 <unknown>

sdot    za.d[w10, 1, vgx2], {z0.h, z1.h}, z10.h[0]  // 11000001-11011010-01000000-00001001
// CHECK-INST: sdot    za.d[w10, 1, vgx2], { z0.h, z1.h }, z10.h[0]
// CHECK-ENCODING: [0x09,0x40,0xda,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1da4009 <unknown>

sdot    za.d[w10, 1], {z0.h, z1.h}, z10.h[0]  // 11000001-11011010-01000000-00001001
// CHECK-INST: sdot    za.d[w10, 1, vgx2], { z0.h, z1.h }, z10.h[0]
// CHECK-ENCODING: [0x09,0x40,0xda,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1da4009 <unknown>

sdot    za.d[w8, 5, vgx2], {z22.h, z23.h}, z14.h[0]  // 11000001-11011110-00000010-11001101
// CHECK-INST: sdot    za.d[w8, 5, vgx2], { z22.h, z23.h }, z14.h[0]
// CHECK-ENCODING: [0xcd,0x02,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de02cd <unknown>

sdot    za.d[w8, 5], {z22.h, z23.h}, z14.h[0]  // 11000001-11011110-00000010-11001101
// CHECK-INST: sdot    za.d[w8, 5, vgx2], { z22.h, z23.h }, z14.h[0]
// CHECK-ENCODING: [0xcd,0x02,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de02cd <unknown>

sdot    za.d[w11, 2, vgx2], {z8.h, z9.h}, z1.h[1]  // 11000001-11010001-01100101-00001010
// CHECK-INST: sdot    za.d[w11, 2, vgx2], { z8.h, z9.h }, z1.h[1]
// CHECK-ENCODING: [0x0a,0x65,0xd1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d1650a <unknown>

sdot    za.d[w11, 2], {z8.h, z9.h}, z1.h[1]  // 11000001-11010001-01100101-00001010
// CHECK-INST: sdot    za.d[w11, 2, vgx2], { z8.h, z9.h }, z1.h[1]
// CHECK-ENCODING: [0x0a,0x65,0xd1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d1650a <unknown>

sdot    za.d[w9, 7, vgx2], {z12.h, z13.h}, z11.h[0]  // 11000001-11011011-00100001-10001111
// CHECK-INST: sdot    za.d[w9, 7, vgx2], { z12.h, z13.h }, z11.h[0]
// CHECK-ENCODING: [0x8f,0x21,0xdb,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1db218f <unknown>

sdot    za.d[w9, 7], {z12.h, z13.h}, z11.h[0]  // 11000001-11011011-00100001-10001111
// CHECK-INST: sdot    za.d[w9, 7, vgx2], { z12.h, z13.h }, z11.h[0]
// CHECK-ENCODING: [0x8f,0x21,0xdb,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1db218f <unknown>


sdot    za.d[w8, 0, vgx2], {z0.h, z1.h}, {z0.h, z1.h}  // 11000001-11100000-00010100-00000000
// CHECK-INST: sdot    za.d[w8, 0, vgx2], { z0.h, z1.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x00,0x14,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e01400 <unknown>

sdot    za.d[w8, 0], {z0.h, z1.h}, {z0.h, z1.h}  // 11000001-11100000-00010100-00000000
// CHECK-INST: sdot    za.d[w8, 0, vgx2], { z0.h, z1.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x00,0x14,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e01400 <unknown>

sdot    za.d[w10, 5, vgx2], {z10.h, z11.h}, {z20.h, z21.h}  // 11000001-11110100-01010101-01000101
// CHECK-INST: sdot    za.d[w10, 5, vgx2], { z10.h, z11.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x45,0x55,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f45545 <unknown>

sdot    za.d[w10, 5], {z10.h, z11.h}, {z20.h, z21.h}  // 11000001-11110100-01010101-01000101
// CHECK-INST: sdot    za.d[w10, 5, vgx2], { z10.h, z11.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x45,0x55,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f45545 <unknown>

sdot    za.d[w11, 7, vgx2], {z12.h, z13.h}, {z8.h, z9.h}  // 11000001-11101000-01110101-10000111
// CHECK-INST: sdot    za.d[w11, 7, vgx2], { z12.h, z13.h }, { z8.h, z9.h }
// CHECK-ENCODING: [0x87,0x75,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e87587 <unknown>

sdot    za.d[w11, 7], {z12.h, z13.h}, {z8.h, z9.h}  // 11000001-11101000-01110101-10000111
// CHECK-INST: sdot    za.d[w11, 7, vgx2], { z12.h, z13.h }, { z8.h, z9.h }
// CHECK-ENCODING: [0x87,0x75,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e87587 <unknown>

sdot    za.d[w11, 7, vgx2], {z30.h, z31.h}, {z30.h, z31.h}  // 11000001-11111110-01110111-11000111
// CHECK-INST: sdot    za.d[w11, 7, vgx2], { z30.h, z31.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xc7,0x77,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe77c7 <unknown>

sdot    za.d[w11, 7], {z30.h, z31.h}, {z30.h, z31.h}  // 11000001-11111110-01110111-11000111
// CHECK-INST: sdot    za.d[w11, 7, vgx2], { z30.h, z31.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xc7,0x77,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe77c7 <unknown>

sdot    za.d[w8, 5, vgx2], {z16.h, z17.h}, {z16.h, z17.h}  // 11000001-11110000-00010110-00000101
// CHECK-INST: sdot    za.d[w8, 5, vgx2], { z16.h, z17.h }, { z16.h, z17.h }
// CHECK-ENCODING: [0x05,0x16,0xf0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f01605 <unknown>

sdot    za.d[w8, 5], {z16.h, z17.h}, {z16.h, z17.h}  // 11000001-11110000-00010110-00000101
// CHECK-INST: sdot    za.d[w8, 5, vgx2], { z16.h, z17.h }, { z16.h, z17.h }
// CHECK-ENCODING: [0x05,0x16,0xf0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f01605 <unknown>

sdot    za.d[w8, 1, vgx2], {z0.h, z1.h}, {z30.h, z31.h}  // 11000001-11111110-00010100-00000001
// CHECK-INST: sdot    za.d[w8, 1, vgx2], { z0.h, z1.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0x01,0x14,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe1401 <unknown>

sdot    za.d[w8, 1], {z0.h, z1.h}, {z30.h, z31.h}  // 11000001-11111110-00010100-00000001
// CHECK-INST: sdot    za.d[w8, 1, vgx2], { z0.h, z1.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0x01,0x14,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe1401 <unknown>

sdot    za.d[w10, 0, vgx2], {z18.h, z19.h}, {z20.h, z21.h}  // 11000001-11110100-01010110-01000000
// CHECK-INST: sdot    za.d[w10, 0, vgx2], { z18.h, z19.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x40,0x56,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f45640 <unknown>

sdot    za.d[w10, 0], {z18.h, z19.h}, {z20.h, z21.h}  // 11000001-11110100-01010110-01000000
// CHECK-INST: sdot    za.d[w10, 0, vgx2], { z18.h, z19.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x40,0x56,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f45640 <unknown>

sdot    za.d[w8, 0, vgx2], {z12.h, z13.h}, {z2.h, z3.h}  // 11000001-11100010-00010101-10000000
// CHECK-INST: sdot    za.d[w8, 0, vgx2], { z12.h, z13.h }, { z2.h, z3.h }
// CHECK-ENCODING: [0x80,0x15,0xe2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e21580 <unknown>

sdot    za.d[w8, 0], {z12.h, z13.h}, {z2.h, z3.h}  // 11000001-11100010-00010101-10000000
// CHECK-INST: sdot    za.d[w8, 0, vgx2], { z12.h, z13.h }, { z2.h, z3.h }
// CHECK-ENCODING: [0x80,0x15,0xe2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e21580 <unknown>

sdot    za.d[w10, 1, vgx2], {z0.h, z1.h}, {z26.h, z27.h}  // 11000001-11111010-01010100-00000001
// CHECK-INST: sdot    za.d[w10, 1, vgx2], { z0.h, z1.h }, { z26.h, z27.h }
// CHECK-ENCODING: [0x01,0x54,0xfa,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fa5401 <unknown>

sdot    za.d[w10, 1], {z0.h, z1.h}, {z26.h, z27.h}  // 11000001-11111010-01010100-00000001
// CHECK-INST: sdot    za.d[w10, 1, vgx2], { z0.h, z1.h }, { z26.h, z27.h }
// CHECK-ENCODING: [0x01,0x54,0xfa,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fa5401 <unknown>

sdot    za.d[w8, 5, vgx2], {z22.h, z23.h}, {z30.h, z31.h}  // 11000001-11111110-00010110-11000101
// CHECK-INST: sdot    za.d[w8, 5, vgx2], { z22.h, z23.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xc5,0x16,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe16c5 <unknown>

sdot    za.d[w8, 5], {z22.h, z23.h}, {z30.h, z31.h}  // 11000001-11111110-00010110-11000101
// CHECK-INST: sdot    za.d[w8, 5, vgx2], { z22.h, z23.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xc5,0x16,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe16c5 <unknown>

sdot    za.d[w11, 2, vgx2], {z8.h, z9.h}, {z0.h, z1.h}  // 11000001-11100000-01110101-00000010
// CHECK-INST: sdot    za.d[w11, 2, vgx2], { z8.h, z9.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x02,0x75,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e07502 <unknown>

sdot    za.d[w11, 2], {z8.h, z9.h}, {z0.h, z1.h}  // 11000001-11100000-01110101-00000010
// CHECK-INST: sdot    za.d[w11, 2, vgx2], { z8.h, z9.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x02,0x75,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e07502 <unknown>

sdot    za.d[w9, 7, vgx2], {z12.h, z13.h}, {z10.h, z11.h}  // 11000001-11101010-00110101-10000111
// CHECK-INST: sdot    za.d[w9, 7, vgx2], { z12.h, z13.h }, { z10.h, z11.h }
// CHECK-ENCODING: [0x87,0x35,0xea,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ea3587 <unknown>

sdot    za.d[w9, 7], {z12.h, z13.h}, {z10.h, z11.h}  // 11000001-11101010-00110101-10000111
// CHECK-INST: sdot    za.d[w9, 7, vgx2], { z12.h, z13.h }, { z10.h, z11.h }
// CHECK-ENCODING: [0x87,0x35,0xea,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ea3587 <unknown>


sdot    za.s[w8, 0, vgx4], {z0.h - z3.h}, z0.h  // 11000001-01110000-00010100-00001000
// CHECK-INST: sdot    za.s[w8, 0, vgx4], { z0.h - z3.h }, z0.h
// CHECK-ENCODING: [0x08,0x14,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1701408 <unknown>

sdot    za.s[w8, 0], {z0.h - z3.h}, z0.h  // 11000001-01110000-00010100-00001000
// CHECK-INST: sdot    za.s[w8, 0, vgx4], { z0.h - z3.h }, z0.h
// CHECK-ENCODING: [0x08,0x14,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1701408 <unknown>

sdot    za.s[w10, 5, vgx4], {z10.h - z13.h}, z5.h  // 11000001-01110101-01010101-01001101
// CHECK-INST: sdot    za.s[w10, 5, vgx4], { z10.h - z13.h }, z5.h
// CHECK-ENCODING: [0x4d,0x55,0x75,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c175554d <unknown>

sdot    za.s[w10, 5], {z10.h - z13.h}, z5.h  // 11000001-01110101-01010101-01001101
// CHECK-INST: sdot    za.s[w10, 5, vgx4], { z10.h - z13.h }, z5.h
// CHECK-ENCODING: [0x4d,0x55,0x75,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c175554d <unknown>

sdot    za.s[w11, 7, vgx4], {z13.h - z16.h}, z8.h  // 11000001-01111000-01110101-10101111
// CHECK-INST: sdot    za.s[w11, 7, vgx4], { z13.h - z16.h }, z8.h
// CHECK-ENCODING: [0xaf,0x75,0x78,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17875af <unknown>

sdot    za.s[w11, 7], {z13.h - z16.h}, z8.h  // 11000001-01111000-01110101-10101111
// CHECK-INST: sdot    za.s[w11, 7, vgx4], { z13.h - z16.h }, z8.h
// CHECK-ENCODING: [0xaf,0x75,0x78,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17875af <unknown>

sdot    za.s[w11, 7, vgx4], {z31.h - z2.h}, z15.h  // 11000001-01111111-01110111-11101111
// CHECK-INST: sdot    za.s[w11, 7, vgx4], { z31.h, z0.h, z1.h, z2.h }, z15.h
// CHECK-ENCODING: [0xef,0x77,0x7f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17f77ef <unknown>

sdot    za.s[w11, 7], {z31.h - z2.h}, z15.h  // 11000001-01111111-01110111-11101111
// CHECK-INST: sdot    za.s[w11, 7, vgx4], { z31.h, z0.h, z1.h, z2.h }, z15.h
// CHECK-ENCODING: [0xef,0x77,0x7f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17f77ef <unknown>

sdot    za.s[w8, 5, vgx4], {z17.h - z20.h}, z0.h  // 11000001-01110000-00010110-00101101
// CHECK-INST: sdot    za.s[w8, 5, vgx4], { z17.h - z20.h }, z0.h
// CHECK-ENCODING: [0x2d,0x16,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c170162d <unknown>

sdot    za.s[w8, 5], {z17.h - z20.h}, z0.h  // 11000001-01110000-00010110-00101101
// CHECK-INST: sdot    za.s[w8, 5, vgx4], { z17.h - z20.h }, z0.h
// CHECK-ENCODING: [0x2d,0x16,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c170162d <unknown>

sdot    za.s[w8, 1, vgx4], {z1.h - z4.h}, z14.h  // 11000001-01111110-00010100-00101001
// CHECK-INST: sdot    za.s[w8, 1, vgx4], { z1.h - z4.h }, z14.h
// CHECK-ENCODING: [0x29,0x14,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17e1429 <unknown>

sdot    za.s[w8, 1], {z1.h - z4.h}, z14.h  // 11000001-01111110-00010100-00101001
// CHECK-INST: sdot    za.s[w8, 1, vgx4], { z1.h - z4.h }, z14.h
// CHECK-ENCODING: [0x29,0x14,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17e1429 <unknown>

sdot    za.s[w10, 0, vgx4], {z19.h - z22.h}, z4.h  // 11000001-01110100-01010110-01101000
// CHECK-INST: sdot    za.s[w10, 0, vgx4], { z19.h - z22.h }, z4.h
// CHECK-ENCODING: [0x68,0x56,0x74,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1745668 <unknown>

sdot    za.s[w10, 0], {z19.h - z22.h}, z4.h  // 11000001-01110100-01010110-01101000
// CHECK-INST: sdot    za.s[w10, 0, vgx4], { z19.h - z22.h }, z4.h
// CHECK-ENCODING: [0x68,0x56,0x74,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1745668 <unknown>

sdot    za.s[w8, 0, vgx4], {z12.h - z15.h}, z2.h  // 11000001-01110010-00010101-10001000
// CHECK-INST: sdot    za.s[w8, 0, vgx4], { z12.h - z15.h }, z2.h
// CHECK-ENCODING: [0x88,0x15,0x72,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1721588 <unknown>

sdot    za.s[w8, 0], {z12.h - z15.h}, z2.h  // 11000001-01110010-00010101-10001000
// CHECK-INST: sdot    za.s[w8, 0, vgx4], { z12.h - z15.h }, z2.h
// CHECK-ENCODING: [0x88,0x15,0x72,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1721588 <unknown>

sdot    za.s[w10, 1, vgx4], {z1.h - z4.h}, z10.h  // 11000001-01111010-01010100-00101001
// CHECK-INST: sdot    za.s[w10, 1, vgx4], { z1.h - z4.h }, z10.h
// CHECK-ENCODING: [0x29,0x54,0x7a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17a5429 <unknown>

sdot    za.s[w10, 1], {z1.h - z4.h}, z10.h  // 11000001-01111010-01010100-00101001
// CHECK-INST: sdot    za.s[w10, 1, vgx4], { z1.h - z4.h }, z10.h
// CHECK-ENCODING: [0x29,0x54,0x7a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17a5429 <unknown>

sdot    za.s[w8, 5, vgx4], {z22.h - z25.h}, z14.h  // 11000001-01111110-00010110-11001101
// CHECK-INST: sdot    za.s[w8, 5, vgx4], { z22.h - z25.h }, z14.h
// CHECK-ENCODING: [0xcd,0x16,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17e16cd <unknown>

sdot    za.s[w8, 5], {z22.h - z25.h}, z14.h  // 11000001-01111110-00010110-11001101
// CHECK-INST: sdot    za.s[w8, 5, vgx4], { z22.h - z25.h }, z14.h
// CHECK-ENCODING: [0xcd,0x16,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17e16cd <unknown>

sdot    za.s[w11, 2, vgx4], {z9.h - z12.h}, z1.h  // 11000001-01110001-01110101-00101010
// CHECK-INST: sdot    za.s[w11, 2, vgx4], { z9.h - z12.h }, z1.h
// CHECK-ENCODING: [0x2a,0x75,0x71,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c171752a <unknown>

sdot    za.s[w11, 2], {z9.h - z12.h}, z1.h  // 11000001-01110001-01110101-00101010
// CHECK-INST: sdot    za.s[w11, 2, vgx4], { z9.h - z12.h }, z1.h
// CHECK-ENCODING: [0x2a,0x75,0x71,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c171752a <unknown>

sdot    za.s[w9, 7, vgx4], {z12.h - z15.h}, z11.h  // 11000001-01111011-00110101-10001111
// CHECK-INST: sdot    za.s[w9, 7, vgx4], { z12.h - z15.h }, z11.h
// CHECK-ENCODING: [0x8f,0x35,0x7b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17b358f <unknown>

sdot    za.s[w9, 7], {z12.h - z15.h}, z11.h  // 11000001-01111011-00110101-10001111
// CHECK-INST: sdot    za.s[w9, 7, vgx4], { z12.h - z15.h }, z11.h
// CHECK-ENCODING: [0x8f,0x35,0x7b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17b358f <unknown>


sdot    za.s[w8, 0, vgx4], {z0.h - z3.h}, z0.h[0]  // 11000001-01010000-10010000-00000000
// CHECK-INST: sdot    za.s[w8, 0, vgx4], { z0.h - z3.h }, z0.h[0]
// CHECK-ENCODING: [0x00,0x90,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1509000 <unknown>

sdot    za.s[w8, 0], {z0.h - z3.h}, z0.h[0]  // 11000001-01010000-10010000-00000000
// CHECK-INST: sdot    za.s[w8, 0, vgx4], { z0.h - z3.h }, z0.h[0]
// CHECK-ENCODING: [0x00,0x90,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1509000 <unknown>

sdot    za.s[w10, 5, vgx4], {z8.h - z11.h}, z5.h[1]  // 11000001-01010101-11010101-00000101
// CHECK-INST: sdot    za.s[w10, 5, vgx4], { z8.h - z11.h }, z5.h[1]
// CHECK-ENCODING: [0x05,0xd5,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c155d505 <unknown>

sdot    za.s[w10, 5], {z8.h - z11.h}, z5.h[1]  // 11000001-01010101-11010101-00000101
// CHECK-INST: sdot    za.s[w10, 5, vgx4], { z8.h - z11.h }, z5.h[1]
// CHECK-ENCODING: [0x05,0xd5,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c155d505 <unknown>

sdot    za.s[w11, 7, vgx4], {z12.h - z15.h}, z8.h[3]  // 11000001-01011000-11111101-10000111
// CHECK-INST: sdot    za.s[w11, 7, vgx4], { z12.h - z15.h }, z8.h[3]
// CHECK-ENCODING: [0x87,0xfd,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c158fd87 <unknown>

sdot    za.s[w11, 7], {z12.h - z15.h}, z8.h[3]  // 11000001-01011000-11111101-10000111
// CHECK-INST: sdot    za.s[w11, 7, vgx4], { z12.h - z15.h }, z8.h[3]
// CHECK-ENCODING: [0x87,0xfd,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c158fd87 <unknown>

sdot    za.s[w11, 7, vgx4], {z28.h - z31.h}, z15.h[3]  // 11000001-01011111-11111111-10000111
// CHECK-INST: sdot    za.s[w11, 7, vgx4], { z28.h - z31.h }, z15.h[3]
// CHECK-ENCODING: [0x87,0xff,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15fff87 <unknown>

sdot    za.s[w11, 7], {z28.h - z31.h}, z15.h[3]  // 11000001-01011111-11111111-10000111
// CHECK-INST: sdot    za.s[w11, 7, vgx4], { z28.h - z31.h }, z15.h[3]
// CHECK-ENCODING: [0x87,0xff,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15fff87 <unknown>

sdot    za.s[w8, 5, vgx4], {z16.h - z19.h}, z0.h[3]  // 11000001-01010000-10011110-00000101
// CHECK-INST: sdot    za.s[w8, 5, vgx4], { z16.h - z19.h }, z0.h[3]
// CHECK-ENCODING: [0x05,0x9e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1509e05 <unknown>

sdot    za.s[w8, 5], {z16.h - z19.h}, z0.h[3]  // 11000001-01010000-10011110-00000101
// CHECK-INST: sdot    za.s[w8, 5, vgx4], { z16.h - z19.h }, z0.h[3]
// CHECK-ENCODING: [0x05,0x9e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1509e05 <unknown>

sdot    za.s[w8, 1, vgx4], {z0.h - z3.h}, z14.h[1]  // 11000001-01011110-10010100-00000001
// CHECK-INST: sdot    za.s[w8, 1, vgx4], { z0.h - z3.h }, z14.h[1]
// CHECK-ENCODING: [0x01,0x94,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e9401 <unknown>

sdot    za.s[w8, 1], {z0.h - z3.h}, z14.h[1]  // 11000001-01011110-10010100-00000001
// CHECK-INST: sdot    za.s[w8, 1, vgx4], { z0.h - z3.h }, z14.h[1]
// CHECK-ENCODING: [0x01,0x94,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e9401 <unknown>

sdot    za.s[w10, 0, vgx4], {z16.h - z19.h}, z4.h[1]  // 11000001-01010100-11010110-00000000
// CHECK-INST: sdot    za.s[w10, 0, vgx4], { z16.h - z19.h }, z4.h[1]
// CHECK-ENCODING: [0x00,0xd6,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c154d600 <unknown>

sdot    za.s[w10, 0], {z16.h - z19.h}, z4.h[1]  // 11000001-01010100-11010110-00000000
// CHECK-INST: sdot    za.s[w10, 0, vgx4], { z16.h - z19.h }, z4.h[1]
// CHECK-ENCODING: [0x00,0xd6,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c154d600 <unknown>

sdot    za.s[w8, 0, vgx4], {z12.h - z15.h}, z2.h[2]  // 11000001-01010010-10011001-10000000
// CHECK-INST: sdot    za.s[w8, 0, vgx4], { z12.h - z15.h }, z2.h[2]
// CHECK-ENCODING: [0x80,0x99,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1529980 <unknown>

sdot    za.s[w8, 0], {z12.h - z15.h}, z2.h[2]  // 11000001-01010010-10011001-10000000
// CHECK-INST: sdot    za.s[w8, 0, vgx4], { z12.h - z15.h }, z2.h[2]
// CHECK-ENCODING: [0x80,0x99,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1529980 <unknown>

sdot    za.s[w10, 1, vgx4], {z0.h - z3.h}, z10.h[2]  // 11000001-01011010-11011000-00000001
// CHECK-INST: sdot    za.s[w10, 1, vgx4], { z0.h - z3.h }, z10.h[2]
// CHECK-ENCODING: [0x01,0xd8,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15ad801 <unknown>

sdot    za.s[w10, 1], {z0.h - z3.h}, z10.h[2]  // 11000001-01011010-11011000-00000001
// CHECK-INST: sdot    za.s[w10, 1, vgx4], { z0.h - z3.h }, z10.h[2]
// CHECK-ENCODING: [0x01,0xd8,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15ad801 <unknown>

sdot    za.s[w8, 5, vgx4], {z20.h - z23.h}, z14.h[2]  // 11000001-01011110-10011010-10000101
// CHECK-INST: sdot    za.s[w8, 5, vgx4], { z20.h - z23.h }, z14.h[2]
// CHECK-ENCODING: [0x85,0x9a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e9a85 <unknown>

sdot    za.s[w8, 5], {z20.h - z23.h}, z14.h[2]  // 11000001-01011110-10011010-10000101
// CHECK-INST: sdot    za.s[w8, 5, vgx4], { z20.h - z23.h }, z14.h[2]
// CHECK-ENCODING: [0x85,0x9a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e9a85 <unknown>

sdot    za.s[w11, 2, vgx4], {z8.h - z11.h}, z1.h[1]  // 11000001-01010001-11110101-00000010
// CHECK-INST: sdot    za.s[w11, 2, vgx4], { z8.h - z11.h }, z1.h[1]
// CHECK-ENCODING: [0x02,0xf5,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c151f502 <unknown>

sdot    za.s[w11, 2], {z8.h - z11.h}, z1.h[1]  // 11000001-01010001-11110101-00000010
// CHECK-INST: sdot    za.s[w11, 2, vgx4], { z8.h - z11.h }, z1.h[1]
// CHECK-ENCODING: [0x02,0xf5,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c151f502 <unknown>

sdot    za.s[w9, 7, vgx4], {z12.h - z15.h}, z11.h[2]  // 11000001-01011011-10111001-10000111
// CHECK-INST: sdot    za.s[w9, 7, vgx4], { z12.h - z15.h }, z11.h[2]
// CHECK-ENCODING: [0x87,0xb9,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15bb987 <unknown>

sdot    za.s[w9, 7], {z12.h - z15.h}, z11.h[2]  // 11000001-01011011-10111001-10000111
// CHECK-INST: sdot    za.s[w9, 7, vgx4], { z12.h - z15.h }, z11.h[2]
// CHECK-ENCODING: [0x87,0xb9,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15bb987 <unknown>

sdot    za.s[w8, 0, vgx4], {z0.h - z3.h}, {z0.h - z3.h}  // 11000001-11100001-00010100-00001000
// CHECK-INST: sdot    za.s[w8, 0, vgx4], { z0.h - z3.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x08,0x14,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11408 <unknown>

sdot    za.s[w8, 0], {z0.h - z3.h}, {z0.h - z3.h}  // 11000001-11100001-00010100-00001000
// CHECK-INST: sdot    za.s[w8, 0, vgx4], { z0.h - z3.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x08,0x14,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11408 <unknown>

sdot    za.s[w10, 5, vgx4], {z8.h - z11.h}, {z20.h - z23.h}  // 11000001-11110101-01010101-00001101
// CHECK-INST: sdot    za.s[w10, 5, vgx4], { z8.h - z11.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x0d,0x55,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f5550d <unknown>

sdot    za.s[w10, 5], {z8.h - z11.h}, {z20.h - z23.h}  // 11000001-11110101-01010101-00001101
// CHECK-INST: sdot    za.s[w10, 5, vgx4], { z8.h - z11.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x0d,0x55,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f5550d <unknown>

sdot    za.s[w11, 7, vgx4], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-11101001-01110101-10001111
// CHECK-INST: sdot    za.s[w11, 7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x8f,0x75,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e9758f <unknown>

sdot    za.s[w11, 7], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-11101001-01110101-10001111
// CHECK-INST: sdot    za.s[w11, 7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x8f,0x75,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e9758f <unknown>

sdot    za.s[w11, 7, vgx4], {z28.h - z31.h}, {z28.h - z31.h}  // 11000001-11111101-01110111-10001111
// CHECK-INST: sdot    za.s[w11, 7, vgx4], { z28.h - z31.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x8f,0x77,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd778f <unknown>

sdot    za.s[w11, 7], {z28.h - z31.h}, {z28.h - z31.h}  // 11000001-11111101-01110111-10001111
// CHECK-INST: sdot    za.s[w11, 7, vgx4], { z28.h - z31.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x8f,0x77,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd778f <unknown>

sdot    za.s[w8, 5, vgx4], {z16.h - z19.h}, {z16.h - z19.h}  // 11000001-11110001-00010110-00001101
// CHECK-INST: sdot    za.s[w8, 5, vgx4], { z16.h - z19.h }, { z16.h - z19.h }
// CHECK-ENCODING: [0x0d,0x16,0xf1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f1160d <unknown>

sdot    za.s[w8, 5], {z16.h - z19.h}, {z16.h - z19.h}  // 11000001-11110001-00010110-00001101
// CHECK-INST: sdot    za.s[w8, 5, vgx4], { z16.h - z19.h }, { z16.h - z19.h }
// CHECK-ENCODING: [0x0d,0x16,0xf1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f1160d <unknown>

sdot    za.s[w8, 1, vgx4], {z0.h - z3.h}, {z28.h - z31.h}  // 11000001-11111101-00010100-00001001
// CHECK-INST: sdot    za.s[w8, 1, vgx4], { z0.h - z3.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x09,0x14,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd1409 <unknown>

sdot    za.s[w8, 1], {z0.h - z3.h}, {z28.h - z31.h}  // 11000001-11111101-00010100-00001001
// CHECK-INST: sdot    za.s[w8, 1, vgx4], { z0.h - z3.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x09,0x14,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd1409 <unknown>


sdot    za.s[w10, 0, vgx4], {z16.h - z19.h}, {z20.h - z23.h}  // 11000001-11110101-01010110-00001000
// CHECK-INST: sdot    za.s[w10, 0, vgx4], { z16.h - z19.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x08,0x56,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f55608 <unknown>

sdot    za.s[w10, 0], {z16.h - z19.h}, {z20.h - z23.h}  // 11000001-11110101-01010110-00001000
// CHECK-INST: sdot    za.s[w10, 0, vgx4], { z16.h - z19.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x08,0x56,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f55608 <unknown>

sdot    za.s[w8, 0, vgx4], {z12.h - z15.h}, {z0.h - z3.h}  // 11000001-11100001-00010101-10001000
// CHECK-INST: sdot    za.s[w8, 0, vgx4], { z12.h - z15.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x88,0x15,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11588 <unknown>

sdot    za.s[w8, 0], {z12.h - z15.h}, {z0.h - z3.h}  // 11000001-11100001-00010101-10001000
// CHECK-INST: sdot    za.s[w8, 0, vgx4], { z12.h - z15.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x88,0x15,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11588 <unknown>

sdot    za.s[w10, 1, vgx4], {z0.h - z3.h}, {z24.h - z27.h}  // 11000001-11111001-01010100-00001001
// CHECK-INST: sdot    za.s[w10, 1, vgx4], { z0.h - z3.h }, { z24.h - z27.h }
// CHECK-ENCODING: [0x09,0x54,0xf9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f95409 <unknown>

sdot    za.s[w10, 1], {z0.h - z3.h}, {z24.h - z27.h}  // 11000001-11111001-01010100-00001001
// CHECK-INST: sdot    za.s[w10, 1, vgx4], { z0.h - z3.h }, { z24.h - z27.h }
// CHECK-ENCODING: [0x09,0x54,0xf9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f95409 <unknown>

sdot    za.s[w8, 5, vgx4], {z20.h - z23.h}, {z28.h - z31.h}  // 11000001-11111101-00010110-10001101
// CHECK-INST: sdot    za.s[w8, 5, vgx4], { z20.h - z23.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x8d,0x16,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd168d <unknown>

sdot    za.s[w8, 5], {z20.h - z23.h}, {z28.h - z31.h}  // 11000001-11111101-00010110-10001101
// CHECK-INST: sdot    za.s[w8, 5, vgx4], { z20.h - z23.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x8d,0x16,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd168d <unknown>

sdot    za.s[w11, 2, vgx4], {z8.h - z11.h}, {z0.h - z3.h}  // 11000001-11100001-01110101-00001010
// CHECK-INST: sdot    za.s[w11, 2, vgx4], { z8.h - z11.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x0a,0x75,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e1750a <unknown>

sdot    za.s[w11, 2], {z8.h - z11.h}, {z0.h - z3.h}  // 11000001-11100001-01110101-00001010
// CHECK-INST: sdot    za.s[w11, 2, vgx4], { z8.h - z11.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x0a,0x75,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e1750a <unknown>

sdot    za.s[w9, 7, vgx4], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-11101001-00110101-10001111
// CHECK-INST: sdot    za.s[w9, 7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x8f,0x35,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e9358f <unknown>

sdot    za.s[w9, 7], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-11101001-00110101-10001111
// CHECK-INST: sdot    za.s[w9, 7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x8f,0x35,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e9358f <unknown>


sdot    za.s[w8, 0, vgx4], {z0.b - z3.b}, z0.b  // 11000001-00110000-00010100-00000000
// CHECK-INST: sdot    za.s[w8, 0, vgx4], { z0.b - z3.b }, z0.b
// CHECK-ENCODING: [0x00,0x14,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1301400 <unknown>

sdot    za.s[w8, 0], {z0.b - z3.b}, z0.b  // 11000001-00110000-00010100-00000000
// CHECK-INST: sdot    za.s[w8, 0, vgx4], { z0.b - z3.b }, z0.b
// CHECK-ENCODING: [0x00,0x14,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1301400 <unknown>

sdot    za.s[w10, 5, vgx4], {z10.b - z13.b}, z5.b  // 11000001-00110101-01010101-01000101
// CHECK-INST: sdot    za.s[w10, 5, vgx4], { z10.b - z13.b }, z5.b
// CHECK-ENCODING: [0x45,0x55,0x35,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1355545 <unknown>

sdot    za.s[w10, 5], {z10.b - z13.b}, z5.b  // 11000001-00110101-01010101-01000101
// CHECK-INST: sdot    za.s[w10, 5, vgx4], { z10.b - z13.b }, z5.b
// CHECK-ENCODING: [0x45,0x55,0x35,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1355545 <unknown>

sdot    za.s[w11, 7, vgx4], {z13.b - z16.b}, z8.b  // 11000001-00111000-01110101-10100111
// CHECK-INST: sdot    za.s[w11, 7, vgx4], { z13.b - z16.b }, z8.b
// CHECK-ENCODING: [0xa7,0x75,0x38,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13875a7 <unknown>

sdot    za.s[w11, 7], {z13.b - z16.b}, z8.b  // 11000001-00111000-01110101-10100111
// CHECK-INST: sdot    za.s[w11, 7, vgx4], { z13.b - z16.b }, z8.b
// CHECK-ENCODING: [0xa7,0x75,0x38,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13875a7 <unknown>

sdot    za.s[w11, 7, vgx4], {z31.b - z2.b}, z15.b  // 11000001-00111111-01110111-11100111
// CHECK-INST: sdot    za.s[w11, 7, vgx4], { z31.b, z0.b, z1.b, z2.b }, z15.b
// CHECK-ENCODING: [0xe7,0x77,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13f77e7 <unknown>

sdot    za.s[w11, 7], {z31.b - z2.b}, z15.b  // 11000001-00111111-01110111-11100111
// CHECK-INST: sdot    za.s[w11, 7, vgx4], { z31.b, z0.b, z1.b, z2.b }, z15.b
// CHECK-ENCODING: [0xe7,0x77,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13f77e7 <unknown>

sdot    za.s[w8, 5, vgx4], {z17.b - z20.b}, z0.b  // 11000001-00110000-00010110-00100101
// CHECK-INST: sdot    za.s[w8, 5, vgx4], { z17.b - z20.b }, z0.b
// CHECK-ENCODING: [0x25,0x16,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1301625 <unknown>

sdot    za.s[w8, 5], {z17.b - z20.b}, z0.b  // 11000001-00110000-00010110-00100101
// CHECK-INST: sdot    za.s[w8, 5, vgx4], { z17.b - z20.b }, z0.b
// CHECK-ENCODING: [0x25,0x16,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1301625 <unknown>

sdot    za.s[w8, 1, vgx4], {z1.b - z4.b}, z14.b  // 11000001-00111110-00010100-00100001
// CHECK-INST: sdot    za.s[w8, 1, vgx4], { z1.b - z4.b }, z14.b
// CHECK-ENCODING: [0x21,0x14,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e1421 <unknown>

sdot    za.s[w8, 1], {z1.b - z4.b}, z14.b  // 11000001-00111110-00010100-00100001
// CHECK-INST: sdot    za.s[w8, 1, vgx4], { z1.b - z4.b }, z14.b
// CHECK-ENCODING: [0x21,0x14,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e1421 <unknown>

sdot    za.s[w10, 0, vgx4], {z19.b - z22.b}, z4.b  // 11000001-00110100-01010110-01100000
// CHECK-INST: sdot    za.s[w10, 0, vgx4], { z19.b - z22.b }, z4.b
// CHECK-ENCODING: [0x60,0x56,0x34,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1345660 <unknown>

sdot    za.s[w10, 0], {z19.b - z22.b}, z4.b  // 11000001-00110100-01010110-01100000
// CHECK-INST: sdot    za.s[w10, 0, vgx4], { z19.b - z22.b }, z4.b
// CHECK-ENCODING: [0x60,0x56,0x34,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1345660 <unknown>

sdot    za.s[w8, 0, vgx4], {z12.b - z15.b}, z2.b  // 11000001-00110010-00010101-10000000
// CHECK-INST: sdot    za.s[w8, 0, vgx4], { z12.b - z15.b }, z2.b
// CHECK-ENCODING: [0x80,0x15,0x32,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1321580 <unknown>

sdot    za.s[w8, 0], {z12.b - z15.b}, z2.b  // 11000001-00110010-00010101-10000000
// CHECK-INST: sdot    za.s[w8, 0, vgx4], { z12.b - z15.b }, z2.b
// CHECK-ENCODING: [0x80,0x15,0x32,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1321580 <unknown>

sdot    za.s[w10, 1, vgx4], {z1.b - z4.b}, z10.b  // 11000001-00111010-01010100-00100001
// CHECK-INST: sdot    za.s[w10, 1, vgx4], { z1.b - z4.b }, z10.b
// CHECK-ENCODING: [0x21,0x54,0x3a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13a5421 <unknown>

sdot    za.s[w10, 1], {z1.b - z4.b}, z10.b  // 11000001-00111010-01010100-00100001
// CHECK-INST: sdot    za.s[w10, 1, vgx4], { z1.b - z4.b }, z10.b
// CHECK-ENCODING: [0x21,0x54,0x3a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13a5421 <unknown>

sdot    za.s[w8, 5, vgx4], {z22.b - z25.b}, z14.b  // 11000001-00111110-00010110-11000101
// CHECK-INST: sdot    za.s[w8, 5, vgx4], { z22.b - z25.b }, z14.b
// CHECK-ENCODING: [0xc5,0x16,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e16c5 <unknown>

sdot    za.s[w8, 5], {z22.b - z25.b}, z14.b  // 11000001-00111110-00010110-11000101
// CHECK-INST: sdot    za.s[w8, 5, vgx4], { z22.b - z25.b }, z14.b
// CHECK-ENCODING: [0xc5,0x16,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e16c5 <unknown>

sdot    za.s[w11, 2, vgx4], {z9.b - z12.b}, z1.b  // 11000001-00110001-01110101-00100010
// CHECK-INST: sdot    za.s[w11, 2, vgx4], { z9.b - z12.b }, z1.b
// CHECK-ENCODING: [0x22,0x75,0x31,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1317522 <unknown>

sdot    za.s[w11, 2], {z9.b - z12.b}, z1.b  // 11000001-00110001-01110101-00100010
// CHECK-INST: sdot    za.s[w11, 2, vgx4], { z9.b - z12.b }, z1.b
// CHECK-ENCODING: [0x22,0x75,0x31,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1317522 <unknown>

sdot    za.s[w9, 7, vgx4], {z12.b - z15.b}, z11.b  // 11000001-00111011-00110101-10000111
// CHECK-INST: sdot    za.s[w9, 7, vgx4], { z12.b - z15.b }, z11.b
// CHECK-ENCODING: [0x87,0x35,0x3b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13b3587 <unknown>

sdot    za.s[w9, 7], {z12.b - z15.b}, z11.b  // 11000001-00111011-00110101-10000111
// CHECK-INST: sdot    za.s[w9, 7, vgx4], { z12.b - z15.b }, z11.b
// CHECK-ENCODING: [0x87,0x35,0x3b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13b3587 <unknown>


sdot    za.s[w8, 0, vgx4], {z0.b - z3.b}, z0.b[0]  // 11000001-01010000-10010000-00100000
// CHECK-INST: sdot    za.s[w8, 0, vgx4], { z0.b - z3.b }, z0.b[0]
// CHECK-ENCODING: [0x20,0x90,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1509020 <unknown>

sdot    za.s[w8, 0], {z0.b - z3.b}, z0.b[0]  // 11000001-01010000-10010000-00100000
// CHECK-INST: sdot    za.s[w8, 0, vgx4], { z0.b - z3.b }, z0.b[0]
// CHECK-ENCODING: [0x20,0x90,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1509020 <unknown>

sdot    za.s[w10, 5, vgx4], {z8.b - z11.b}, z5.b[1]  // 11000001-01010101-11010101-00100101
// CHECK-INST: sdot    za.s[w10, 5, vgx4], { z8.b - z11.b }, z5.b[1]
// CHECK-ENCODING: [0x25,0xd5,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c155d525 <unknown>

sdot    za.s[w10, 5], {z8.b - z11.b}, z5.b[1]  // 11000001-01010101-11010101-00100101
// CHECK-INST: sdot    za.s[w10, 5, vgx4], { z8.b - z11.b }, z5.b[1]
// CHECK-ENCODING: [0x25,0xd5,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c155d525 <unknown>

sdot    za.s[w11, 7, vgx4], {z12.b - z15.b}, z8.b[3]  // 11000001-01011000-11111101-10100111
// CHECK-INST: sdot    za.s[w11, 7, vgx4], { z12.b - z15.b }, z8.b[3]
// CHECK-ENCODING: [0xa7,0xfd,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c158fda7 <unknown>

sdot    za.s[w11, 7], {z12.b - z15.b}, z8.b[3]  // 11000001-01011000-11111101-10100111
// CHECK-INST: sdot    za.s[w11, 7, vgx4], { z12.b - z15.b }, z8.b[3]
// CHECK-ENCODING: [0xa7,0xfd,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c158fda7 <unknown>

sdot    za.s[w11, 7, vgx4], {z28.b - z31.b}, z15.b[3]  // 11000001-01011111-11111111-10100111
// CHECK-INST: sdot    za.s[w11, 7, vgx4], { z28.b - z31.b }, z15.b[3]
// CHECK-ENCODING: [0xa7,0xff,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15fffa7 <unknown>

sdot    za.s[w11, 7], {z28.b - z31.b}, z15.b[3]  // 11000001-01011111-11111111-10100111
// CHECK-INST: sdot    za.s[w11, 7, vgx4], { z28.b - z31.b }, z15.b[3]
// CHECK-ENCODING: [0xa7,0xff,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15fffa7 <unknown>

sdot    za.s[w8, 5, vgx4], {z16.b - z19.b}, z0.b[3]  // 11000001-01010000-10011110-00100101
// CHECK-INST: sdot    za.s[w8, 5, vgx4], { z16.b - z19.b }, z0.b[3]
// CHECK-ENCODING: [0x25,0x9e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1509e25 <unknown>

sdot    za.s[w8, 5], {z16.b - z19.b}, z0.b[3]  // 11000001-01010000-10011110-00100101
// CHECK-INST: sdot    za.s[w8, 5, vgx4], { z16.b - z19.b }, z0.b[3]
// CHECK-ENCODING: [0x25,0x9e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1509e25 <unknown>

sdot    za.s[w8, 1, vgx4], {z0.b - z3.b}, z14.b[1]  // 11000001-01011110-10010100-00100001
// CHECK-INST: sdot    za.s[w8, 1, vgx4], { z0.b - z3.b }, z14.b[1]
// CHECK-ENCODING: [0x21,0x94,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e9421 <unknown>

sdot    za.s[w8, 1], {z0.b - z3.b}, z14.b[1]  // 11000001-01011110-10010100-00100001
// CHECK-INST: sdot    za.s[w8, 1, vgx4], { z0.b - z3.b }, z14.b[1]
// CHECK-ENCODING: [0x21,0x94,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e9421 <unknown>

sdot    za.s[w10, 0, vgx4], {z16.b - z19.b}, z4.b[1]  // 11000001-01010100-11010110-00100000
// CHECK-INST: sdot    za.s[w10, 0, vgx4], { z16.b - z19.b }, z4.b[1]
// CHECK-ENCODING: [0x20,0xd6,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c154d620 <unknown>

sdot    za.s[w10, 0], {z16.b - z19.b}, z4.b[1]  // 11000001-01010100-11010110-00100000
// CHECK-INST: sdot    za.s[w10, 0, vgx4], { z16.b - z19.b }, z4.b[1]
// CHECK-ENCODING: [0x20,0xd6,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c154d620 <unknown>

sdot    za.s[w8, 0, vgx4], {z12.b - z15.b}, z2.b[2]  // 11000001-01010010-10011001-10100000
// CHECK-INST: sdot    za.s[w8, 0, vgx4], { z12.b - z15.b }, z2.b[2]
// CHECK-ENCODING: [0xa0,0x99,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15299a0 <unknown>

sdot    za.s[w8, 0], {z12.b - z15.b}, z2.b[2]  // 11000001-01010010-10011001-10100000
// CHECK-INST: sdot    za.s[w8, 0, vgx4], { z12.b - z15.b }, z2.b[2]
// CHECK-ENCODING: [0xa0,0x99,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15299a0 <unknown>

sdot    za.s[w10, 1, vgx4], {z0.b - z3.b}, z10.b[2]  // 11000001-01011010-11011000-00100001
// CHECK-INST: sdot    za.s[w10, 1, vgx4], { z0.b - z3.b }, z10.b[2]
// CHECK-ENCODING: [0x21,0xd8,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15ad821 <unknown>

sdot    za.s[w10, 1], {z0.b - z3.b}, z10.b[2]  // 11000001-01011010-11011000-00100001
// CHECK-INST: sdot    za.s[w10, 1, vgx4], { z0.b - z3.b }, z10.b[2]
// CHECK-ENCODING: [0x21,0xd8,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15ad821 <unknown>

sdot    za.s[w8, 5, vgx4], {z20.b - z23.b}, z14.b[2]  // 11000001-01011110-10011010-10100101
// CHECK-INST: sdot    za.s[w8, 5, vgx4], { z20.b - z23.b }, z14.b[2]
// CHECK-ENCODING: [0xa5,0x9a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e9aa5 <unknown>

sdot    za.s[w8, 5], {z20.b - z23.b}, z14.b[2]  // 11000001-01011110-10011010-10100101
// CHECK-INST: sdot    za.s[w8, 5, vgx4], { z20.b - z23.b }, z14.b[2]
// CHECK-ENCODING: [0xa5,0x9a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e9aa5 <unknown>

sdot    za.s[w11, 2, vgx4], {z8.b - z11.b}, z1.b[1]  // 11000001-01010001-11110101-00100010
// CHECK-INST: sdot    za.s[w11, 2, vgx4], { z8.b - z11.b }, z1.b[1]
// CHECK-ENCODING: [0x22,0xf5,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c151f522 <unknown>

sdot    za.s[w11, 2], {z8.b - z11.b}, z1.b[1]  // 11000001-01010001-11110101-00100010
// CHECK-INST: sdot    za.s[w11, 2, vgx4], { z8.b - z11.b }, z1.b[1]
// CHECK-ENCODING: [0x22,0xf5,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c151f522 <unknown>

sdot    za.s[w9, 7, vgx4], {z12.b - z15.b}, z11.b[2]  // 11000001-01011011-10111001-10100111
// CHECK-INST: sdot    za.s[w9, 7, vgx4], { z12.b - z15.b }, z11.b[2]
// CHECK-ENCODING: [0xa7,0xb9,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15bb9a7 <unknown>

sdot    za.s[w9, 7], {z12.b - z15.b}, z11.b[2]  // 11000001-01011011-10111001-10100111
// CHECK-INST: sdot    za.s[w9, 7, vgx4], { z12.b - z15.b }, z11.b[2]
// CHECK-ENCODING: [0xa7,0xb9,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15bb9a7 <unknown>


sdot    za.s[w8, 0, vgx4], {z0.b - z3.b}, {z0.b - z3.b}  // 11000001-10100001-00010100-00000000
// CHECK-INST: sdot    za.s[w8, 0, vgx4], { z0.b - z3.b }, { z0.b - z3.b }
// CHECK-ENCODING: [0x00,0x14,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11400 <unknown>

sdot    za.s[w8, 0], {z0.b - z3.b}, {z0.b - z3.b}  // 11000001-10100001-00010100-00000000
// CHECK-INST: sdot    za.s[w8, 0, vgx4], { z0.b - z3.b }, { z0.b - z3.b }
// CHECK-ENCODING: [0x00,0x14,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11400 <unknown>

sdot    za.s[w10, 5, vgx4], {z8.b - z11.b}, {z20.b - z23.b}  // 11000001-10110101-01010101-00000101
// CHECK-INST: sdot    za.s[w10, 5, vgx4], { z8.b - z11.b }, { z20.b - z23.b }
// CHECK-ENCODING: [0x05,0x55,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b55505 <unknown>

sdot    za.s[w10, 5], {z8.b - z11.b}, {z20.b - z23.b}  // 11000001-10110101-01010101-00000101
// CHECK-INST: sdot    za.s[w10, 5, vgx4], { z8.b - z11.b }, { z20.b - z23.b }
// CHECK-ENCODING: [0x05,0x55,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b55505 <unknown>

sdot    za.s[w11, 7, vgx4], {z12.b - z15.b}, {z8.b - z11.b}  // 11000001-10101001-01110101-10000111
// CHECK-INST: sdot    za.s[w11, 7, vgx4], { z12.b - z15.b }, { z8.b - z11.b }
// CHECK-ENCODING: [0x87,0x75,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a97587 <unknown>

sdot    za.s[w11, 7], {z12.b - z15.b}, {z8.b - z11.b}  // 11000001-10101001-01110101-10000111
// CHECK-INST: sdot    za.s[w11, 7, vgx4], { z12.b - z15.b }, { z8.b - z11.b }
// CHECK-ENCODING: [0x87,0x75,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a97587 <unknown>

sdot    za.s[w11, 7, vgx4], {z28.b - z31.b}, {z28.b - z31.b}  // 11000001-10111101-01110111-10000111
// CHECK-INST: sdot    za.s[w11, 7, vgx4], { z28.b - z31.b }, { z28.b - z31.b }
// CHECK-ENCODING: [0x87,0x77,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd7787 <unknown>

sdot    za.s[w11, 7], {z28.b - z31.b}, {z28.b - z31.b}  // 11000001-10111101-01110111-10000111
// CHECK-INST: sdot    za.s[w11, 7, vgx4], { z28.b - z31.b }, { z28.b - z31.b }
// CHECK-ENCODING: [0x87,0x77,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd7787 <unknown>

sdot    za.s[w8, 5, vgx4], {z16.b - z19.b}, {z16.b - z19.b}  // 11000001-10110001-00010110-00000101
// CHECK-INST: sdot    za.s[w8, 5, vgx4], { z16.b - z19.b }, { z16.b - z19.b }
// CHECK-ENCODING: [0x05,0x16,0xb1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b11605 <unknown>

sdot    za.s[w8, 5], {z16.b - z19.b}, {z16.b - z19.b}  // 11000001-10110001-00010110-00000101
// CHECK-INST: sdot    za.s[w8, 5, vgx4], { z16.b - z19.b }, { z16.b - z19.b }
// CHECK-ENCODING: [0x05,0x16,0xb1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b11605 <unknown>

sdot    za.s[w8, 1, vgx4], {z0.b - z3.b}, {z28.b - z31.b}  // 11000001-10111101-00010100-00000001
// CHECK-INST: sdot    za.s[w8, 1, vgx4], { z0.b - z3.b }, { z28.b - z31.b }
// CHECK-ENCODING: [0x01,0x14,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd1401 <unknown>

sdot    za.s[w8, 1], {z0.b - z3.b}, {z28.b - z31.b}  // 11000001-10111101-00010100-00000001
// CHECK-INST: sdot    za.s[w8, 1, vgx4], { z0.b - z3.b }, { z28.b - z31.b }
// CHECK-ENCODING: [0x01,0x14,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd1401 <unknown>

sdot    za.s[w10, 0, vgx4], {z16.b - z19.b}, {z20.b - z23.b}  // 11000001-10110101-01010110-00000000
// CHECK-INST: sdot    za.s[w10, 0, vgx4], { z16.b - z19.b }, { z20.b - z23.b }
// CHECK-ENCODING: [0x00,0x56,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b55600 <unknown>

sdot    za.s[w10, 0], {z16.b - z19.b}, {z20.b - z23.b}  // 11000001-10110101-01010110-00000000
// CHECK-INST: sdot    za.s[w10, 0, vgx4], { z16.b - z19.b }, { z20.b - z23.b }
// CHECK-ENCODING: [0x00,0x56,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b55600 <unknown>

sdot    za.s[w8, 0, vgx4], {z12.b - z15.b}, {z0.b - z3.b}  // 11000001-10100001-00010101-10000000
// CHECK-INST: sdot    za.s[w8, 0, vgx4], { z12.b - z15.b }, { z0.b - z3.b }
// CHECK-ENCODING: [0x80,0x15,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11580 <unknown>

sdot    za.s[w8, 0], {z12.b - z15.b}, {z0.b - z3.b}  // 11000001-10100001-00010101-10000000
// CHECK-INST: sdot    za.s[w8, 0, vgx4], { z12.b - z15.b }, { z0.b - z3.b }
// CHECK-ENCODING: [0x80,0x15,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11580 <unknown>

sdot    za.s[w10, 1, vgx4], {z0.b - z3.b}, {z24.b - z27.b}  // 11000001-10111001-01010100-00000001
// CHECK-INST: sdot    za.s[w10, 1, vgx4], { z0.b - z3.b }, { z24.b - z27.b }
// CHECK-ENCODING: [0x01,0x54,0xb9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b95401 <unknown>

sdot    za.s[w10, 1], {z0.b - z3.b}, {z24.b - z27.b}  // 11000001-10111001-01010100-00000001
// CHECK-INST: sdot    za.s[w10, 1, vgx4], { z0.b - z3.b }, { z24.b - z27.b }
// CHECK-ENCODING: [0x01,0x54,0xb9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b95401 <unknown>

sdot    za.s[w8, 5, vgx4], {z20.b - z23.b}, {z28.b - z31.b}  // 11000001-10111101-00010110-10000101
// CHECK-INST: sdot    za.s[w8, 5, vgx4], { z20.b - z23.b }, { z28.b - z31.b }
// CHECK-ENCODING: [0x85,0x16,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd1685 <unknown>

sdot    za.s[w8, 5], {z20.b - z23.b}, {z28.b - z31.b}  // 11000001-10111101-00010110-10000101
// CHECK-INST: sdot    za.s[w8, 5, vgx4], { z20.b - z23.b }, { z28.b - z31.b }
// CHECK-ENCODING: [0x85,0x16,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd1685 <unknown>

sdot    za.s[w11, 2, vgx4], {z8.b - z11.b}, {z0.b - z3.b}  // 11000001-10100001-01110101-00000010
// CHECK-INST: sdot    za.s[w11, 2, vgx4], { z8.b - z11.b }, { z0.b - z3.b }
// CHECK-ENCODING: [0x02,0x75,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a17502 <unknown>

sdot    za.s[w11, 2], {z8.b - z11.b}, {z0.b - z3.b}  // 11000001-10100001-01110101-00000010
// CHECK-INST: sdot    za.s[w11, 2, vgx4], { z8.b - z11.b }, { z0.b - z3.b }
// CHECK-ENCODING: [0x02,0x75,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a17502 <unknown>

sdot    za.s[w9, 7, vgx4], {z12.b - z15.b}, {z8.b - z11.b}  // 11000001-10101001-00110101-10000111
// CHECK-INST: sdot    za.s[w9, 7, vgx4], { z12.b - z15.b }, { z8.b - z11.b }
// CHECK-ENCODING: [0x87,0x35,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a93587 <unknown>

sdot    za.s[w9, 7], {z12.b - z15.b}, {z8.b - z11.b}  // 11000001-10101001-00110101-10000111
// CHECK-INST: sdot    za.s[w9, 7, vgx4], { z12.b - z15.b }, { z8.b - z11.b }
// CHECK-ENCODING: [0x87,0x35,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a93587 <unknown>


sdot    za.d[w8, 0, vgx4], {z0.h - z3.h}, z0.h  // 11000001-01110000-00010100-00000000
// CHECK-INST: sdot    za.d[w8, 0, vgx4], { z0.h - z3.h }, z0.h
// CHECK-ENCODING: [0x00,0x14,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1701400 <unknown>

sdot    za.d[w8, 0], {z0.h - z3.h}, z0.h  // 11000001-01110000-00010100-00000000
// CHECK-INST: sdot    za.d[w8, 0, vgx4], { z0.h - z3.h }, z0.h
// CHECK-ENCODING: [0x00,0x14,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1701400 <unknown>

sdot    za.d[w10, 5, vgx4], {z10.h - z13.h}, z5.h  // 11000001-01110101-01010101-01000101
// CHECK-INST: sdot    za.d[w10, 5, vgx4], { z10.h - z13.h }, z5.h
// CHECK-ENCODING: [0x45,0x55,0x75,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1755545 <unknown>

sdot    za.d[w10, 5], {z10.h - z13.h}, z5.h  // 11000001-01110101-01010101-01000101
// CHECK-INST: sdot    za.d[w10, 5, vgx4], { z10.h - z13.h }, z5.h
// CHECK-ENCODING: [0x45,0x55,0x75,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1755545 <unknown>

sdot    za.d[w11, 7, vgx4], {z13.h - z16.h}, z8.h  // 11000001-01111000-01110101-10100111
// CHECK-INST: sdot    za.d[w11, 7, vgx4], { z13.h - z16.h }, z8.h
// CHECK-ENCODING: [0xa7,0x75,0x78,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17875a7 <unknown>

sdot    za.d[w11, 7], {z13.h - z16.h}, z8.h  // 11000001-01111000-01110101-10100111
// CHECK-INST: sdot    za.d[w11, 7, vgx4], { z13.h - z16.h }, z8.h
// CHECK-ENCODING: [0xa7,0x75,0x78,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17875a7 <unknown>

sdot    za.d[w11, 7, vgx4], {z31.h - z2.h}, z15.h  // 11000001-01111111-01110111-11100111
// CHECK-INST: sdot    za.d[w11, 7, vgx4], { z31.h, z0.h, z1.h, z2.h }, z15.h
// CHECK-ENCODING: [0xe7,0x77,0x7f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17f77e7 <unknown>

sdot    za.d[w11, 7], {z31.h - z2.h}, z15.h  // 11000001-01111111-01110111-11100111
// CHECK-INST: sdot    za.d[w11, 7, vgx4], { z31.h, z0.h, z1.h, z2.h }, z15.h
// CHECK-ENCODING: [0xe7,0x77,0x7f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17f77e7 <unknown>

sdot    za.d[w8, 5, vgx4], {z17.h - z20.h}, z0.h  // 11000001-01110000-00010110-00100101
// CHECK-INST: sdot    za.d[w8, 5, vgx4], { z17.h - z20.h }, z0.h
// CHECK-ENCODING: [0x25,0x16,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1701625 <unknown>

sdot    za.d[w8, 5], {z17.h - z20.h}, z0.h  // 11000001-01110000-00010110-00100101
// CHECK-INST: sdot    za.d[w8, 5, vgx4], { z17.h - z20.h }, z0.h
// CHECK-ENCODING: [0x25,0x16,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1701625 <unknown>

sdot    za.d[w8, 1, vgx4], {z1.h - z4.h}, z14.h  // 11000001-01111110-00010100-00100001
// CHECK-INST: sdot    za.d[w8, 1, vgx4], { z1.h - z4.h }, z14.h
// CHECK-ENCODING: [0x21,0x14,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17e1421 <unknown>

sdot    za.d[w8, 1], {z1.h - z4.h}, z14.h  // 11000001-01111110-00010100-00100001
// CHECK-INST: sdot    za.d[w8, 1, vgx4], { z1.h - z4.h }, z14.h
// CHECK-ENCODING: [0x21,0x14,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17e1421 <unknown>

sdot    za.d[w10, 0, vgx4], {z19.h - z22.h}, z4.h  // 11000001-01110100-01010110-01100000
// CHECK-INST: sdot    za.d[w10, 0, vgx4], { z19.h - z22.h }, z4.h
// CHECK-ENCODING: [0x60,0x56,0x74,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1745660 <unknown>

sdot    za.d[w10, 0], {z19.h - z22.h}, z4.h  // 11000001-01110100-01010110-01100000
// CHECK-INST: sdot    za.d[w10, 0, vgx4], { z19.h - z22.h }, z4.h
// CHECK-ENCODING: [0x60,0x56,0x74,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1745660 <unknown>

sdot    za.d[w8, 0, vgx4], {z12.h - z15.h}, z2.h  // 11000001-01110010-00010101-10000000
// CHECK-INST: sdot    za.d[w8, 0, vgx4], { z12.h - z15.h }, z2.h
// CHECK-ENCODING: [0x80,0x15,0x72,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1721580 <unknown>

sdot    za.d[w8, 0], {z12.h - z15.h}, z2.h  // 11000001-01110010-00010101-10000000
// CHECK-INST: sdot    za.d[w8, 0, vgx4], { z12.h - z15.h }, z2.h
// CHECK-ENCODING: [0x80,0x15,0x72,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1721580 <unknown>

sdot    za.d[w10, 1, vgx4], {z1.h - z4.h}, z10.h  // 11000001-01111010-01010100-00100001
// CHECK-INST: sdot    za.d[w10, 1, vgx4], { z1.h - z4.h }, z10.h
// CHECK-ENCODING: [0x21,0x54,0x7a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17a5421 <unknown>

sdot    za.d[w10, 1], {z1.h - z4.h}, z10.h  // 11000001-01111010-01010100-00100001
// CHECK-INST: sdot    za.d[w10, 1, vgx4], { z1.h - z4.h }, z10.h
// CHECK-ENCODING: [0x21,0x54,0x7a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17a5421 <unknown>

sdot    za.d[w8, 5, vgx4], {z22.h - z25.h}, z14.h  // 11000001-01111110-00010110-11000101
// CHECK-INST: sdot    za.d[w8, 5, vgx4], { z22.h - z25.h }, z14.h
// CHECK-ENCODING: [0xc5,0x16,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17e16c5 <unknown>

sdot    za.d[w8, 5], {z22.h - z25.h}, z14.h  // 11000001-01111110-00010110-11000101
// CHECK-INST: sdot    za.d[w8, 5, vgx4], { z22.h - z25.h }, z14.h
// CHECK-ENCODING: [0xc5,0x16,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17e16c5 <unknown>

sdot    za.d[w11, 2, vgx4], {z9.h - z12.h}, z1.h  // 11000001-01110001-01110101-00100010
// CHECK-INST: sdot    za.d[w11, 2, vgx4], { z9.h - z12.h }, z1.h
// CHECK-ENCODING: [0x22,0x75,0x71,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1717522 <unknown>

sdot    za.d[w11, 2], {z9.h - z12.h}, z1.h  // 11000001-01110001-01110101-00100010
// CHECK-INST: sdot    za.d[w11, 2, vgx4], { z9.h - z12.h }, z1.h
// CHECK-ENCODING: [0x22,0x75,0x71,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1717522 <unknown>

sdot    za.d[w9, 7, vgx4], {z12.h - z15.h}, z11.h  // 11000001-01111011-00110101-10000111
// CHECK-INST: sdot    za.d[w9, 7, vgx4], { z12.h - z15.h }, z11.h
// CHECK-ENCODING: [0x87,0x35,0x7b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17b3587 <unknown>

sdot    za.d[w9, 7], {z12.h - z15.h}, z11.h  // 11000001-01111011-00110101-10000111
// CHECK-INST: sdot    za.d[w9, 7, vgx4], { z12.h - z15.h }, z11.h
// CHECK-ENCODING: [0x87,0x35,0x7b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17b3587 <unknown>


sdot    za.d[w8, 0, vgx4], {z0.h - z3.h}, z0.h[0]  // 11000001-11010000-10000000-00001000
// CHECK-INST: sdot    za.d[w8, 0, vgx4], { z0.h - z3.h }, z0.h[0]
// CHECK-ENCODING: [0x08,0x80,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d08008 <unknown>

sdot    za.d[w8, 0], {z0.h - z3.h}, z0.h[0]  // 11000001-11010000-10000000-00001000
// CHECK-INST: sdot    za.d[w8, 0, vgx4], { z0.h - z3.h }, z0.h[0]
// CHECK-ENCODING: [0x08,0x80,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d08008 <unknown>

sdot    za.d[w10, 5, vgx4], {z8.h - z11.h}, z5.h[1]  // 11000001-11010101-11000101-00001101
// CHECK-INST: sdot    za.d[w10, 5, vgx4], { z8.h - z11.h }, z5.h[1]
// CHECK-ENCODING: [0x0d,0xc5,0xd5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d5c50d <unknown>

sdot    za.d[w10, 5], {z8.h - z11.h}, z5.h[1]  // 11000001-11010101-11000101-00001101
// CHECK-INST: sdot    za.d[w10, 5, vgx4], { z8.h - z11.h }, z5.h[1]
// CHECK-ENCODING: [0x0d,0xc5,0xd5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d5c50d <unknown>

sdot    za.d[w11, 7, vgx4], {z12.h - z15.h}, z8.h[1]  // 11000001-11011000-11100101-10001111
// CHECK-INST: sdot    za.d[w11, 7, vgx4], { z12.h - z15.h }, z8.h[1]
// CHECK-ENCODING: [0x8f,0xe5,0xd8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d8e58f <unknown>

sdot    za.d[w11, 7], {z12.h - z15.h}, z8.h[1]  // 11000001-11011000-11100101-10001111
// CHECK-INST: sdot    za.d[w11, 7, vgx4], { z12.h - z15.h }, z8.h[1]
// CHECK-ENCODING: [0x8f,0xe5,0xd8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d8e58f <unknown>

sdot    za.d[w11, 7, vgx4], {z28.h - z31.h}, z15.h[1]  // 11000001-11011111-11100111-10001111
// CHECK-INST: sdot    za.d[w11, 7, vgx4], { z28.h - z31.h }, z15.h[1]
// CHECK-ENCODING: [0x8f,0xe7,0xdf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dfe78f <unknown>

sdot    za.d[w11, 7], {z28.h - z31.h}, z15.h[1]  // 11000001-11011111-11100111-10001111
// CHECK-INST: sdot    za.d[w11, 7, vgx4], { z28.h - z31.h }, z15.h[1]
// CHECK-ENCODING: [0x8f,0xe7,0xdf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dfe78f <unknown>

sdot    za.d[w8, 5, vgx4], {z16.h - z19.h}, z0.h[1]  // 11000001-11010000-10000110-00001101
// CHECK-INST: sdot    za.d[w8, 5, vgx4], { z16.h - z19.h }, z0.h[1]
// CHECK-ENCODING: [0x0d,0x86,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d0860d <unknown>

sdot    za.d[w8, 5], {z16.h - z19.h}, z0.h[1]  // 11000001-11010000-10000110-00001101
// CHECK-INST: sdot    za.d[w8, 5, vgx4], { z16.h - z19.h }, z0.h[1]
// CHECK-ENCODING: [0x0d,0x86,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d0860d <unknown>

sdot    za.d[w8, 1, vgx4], {z0.h - z3.h}, z14.h[1]  // 11000001-11011110-10000100-00001001
// CHECK-INST: sdot    za.d[w8, 1, vgx4], { z0.h - z3.h }, z14.h[1]
// CHECK-ENCODING: [0x09,0x84,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de8409 <unknown>

sdot    za.d[w8, 1], {z0.h - z3.h}, z14.h[1]  // 11000001-11011110-10000100-00001001
// CHECK-INST: sdot    za.d[w8, 1, vgx4], { z0.h - z3.h }, z14.h[1]
// CHECK-ENCODING: [0x09,0x84,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de8409 <unknown>

sdot    za.d[w10, 0, vgx4], {z16.h - z19.h}, z4.h[1]  // 11000001-11010100-11000110-00001000
// CHECK-INST: sdot    za.d[w10, 0, vgx4], { z16.h - z19.h }, z4.h[1]
// CHECK-ENCODING: [0x08,0xc6,0xd4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d4c608 <unknown>

sdot    za.d[w10, 0], {z16.h - z19.h}, z4.h[1]  // 11000001-11010100-11000110-00001000
// CHECK-INST: sdot    za.d[w10, 0, vgx4], { z16.h - z19.h }, z4.h[1]
// CHECK-ENCODING: [0x08,0xc6,0xd4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d4c608 <unknown>

sdot    za.d[w8, 0, vgx4], {z12.h - z15.h}, z2.h[0]  // 11000001-11010010-10000001-10001000
// CHECK-INST: sdot    za.d[w8, 0, vgx4], { z12.h - z15.h }, z2.h[0]
// CHECK-ENCODING: [0x88,0x81,0xd2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d28188 <unknown>

sdot    za.d[w8, 0], {z12.h - z15.h}, z2.h[0]  // 11000001-11010010-10000001-10001000
// CHECK-INST: sdot    za.d[w8, 0, vgx4], { z12.h - z15.h }, z2.h[0]
// CHECK-ENCODING: [0x88,0x81,0xd2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d28188 <unknown>

sdot    za.d[w10, 1, vgx4], {z0.h - z3.h}, z10.h[0]  // 11000001-11011010-11000000-00001001
// CHECK-INST: sdot    za.d[w10, 1, vgx4], { z0.h - z3.h }, z10.h[0]
// CHECK-ENCODING: [0x09,0xc0,0xda,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dac009 <unknown>

sdot    za.d[w10, 1], {z0.h - z3.h}, z10.h[0]  // 11000001-11011010-11000000-00001001
// CHECK-INST: sdot    za.d[w10, 1, vgx4], { z0.h - z3.h }, z10.h[0]
// CHECK-ENCODING: [0x09,0xc0,0xda,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dac009 <unknown>

sdot    za.d[w8, 5, vgx4], {z20.h - z23.h}, z14.h[0]  // 11000001-11011110-10000010-10001101
// CHECK-INST: sdot    za.d[w8, 5, vgx4], { z20.h - z23.h }, z14.h[0]
// CHECK-ENCODING: [0x8d,0x82,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de828d <unknown>

sdot    za.d[w8, 5], {z20.h - z23.h}, z14.h[0]  // 11000001-11011110-10000010-10001101
// CHECK-INST: sdot    za.d[w8, 5, vgx4], { z20.h - z23.h }, z14.h[0]
// CHECK-ENCODING: [0x8d,0x82,0xde,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1de828d <unknown>

sdot    za.d[w11, 2, vgx4], {z8.h - z11.h}, z1.h[1]  // 11000001-11010001-11100101-00001010
// CHECK-INST: sdot    za.d[w11, 2, vgx4], { z8.h - z11.h }, z1.h[1]
// CHECK-ENCODING: [0x0a,0xe5,0xd1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d1e50a <unknown>

sdot    za.d[w11, 2], {z8.h - z11.h}, z1.h[1]  // 11000001-11010001-11100101-00001010
// CHECK-INST: sdot    za.d[w11, 2, vgx4], { z8.h - z11.h }, z1.h[1]
// CHECK-ENCODING: [0x0a,0xe5,0xd1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1d1e50a <unknown>

sdot    za.d[w9, 7, vgx4], {z12.h - z15.h}, z11.h[0]  // 11000001-11011011-10100001-10001111
// CHECK-INST: sdot    za.d[w9, 7, vgx4], { z12.h - z15.h }, z11.h[0]
// CHECK-ENCODING: [0x8f,0xa1,0xdb,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dba18f <unknown>

sdot    za.d[w9, 7], {z12.h - z15.h}, z11.h[0]  // 11000001-11011011-10100001-10001111
// CHECK-INST: sdot    za.d[w9, 7, vgx4], { z12.h - z15.h }, z11.h[0]
// CHECK-ENCODING: [0x8f,0xa1,0xdb,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1dba18f <unknown>


sdot    za.d[w8, 0, vgx4], {z0.h - z3.h}, {z0.h - z3.h}  // 11000001-11100001-00010100-00000000
// CHECK-INST: sdot    za.d[w8, 0, vgx4], { z0.h - z3.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x00,0x14,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11400 <unknown>

sdot    za.d[w8, 0], {z0.h - z3.h}, {z0.h - z3.h}  // 11000001-11100001-00010100-00000000
// CHECK-INST: sdot    za.d[w8, 0, vgx4], { z0.h - z3.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x00,0x14,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11400 <unknown>

sdot    za.d[w10, 5, vgx4], {z8.h - z11.h}, {z20.h - z23.h}  // 11000001-11110101-01010101-00000101
// CHECK-INST: sdot    za.d[w10, 5, vgx4], { z8.h - z11.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x05,0x55,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f55505 <unknown>

sdot    za.d[w10, 5], {z8.h - z11.h}, {z20.h - z23.h}  // 11000001-11110101-01010101-00000101
// CHECK-INST: sdot    za.d[w10, 5, vgx4], { z8.h - z11.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x05,0x55,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f55505 <unknown>

sdot    za.d[w11, 7, vgx4], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-11101001-01110101-10000111
// CHECK-INST: sdot    za.d[w11, 7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x87,0x75,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e97587 <unknown>

sdot    za.d[w11, 7], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-11101001-01110101-10000111
// CHECK-INST: sdot    za.d[w11, 7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x87,0x75,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e97587 <unknown>

sdot    za.d[w11, 7, vgx4], {z28.h - z31.h}, {z28.h - z31.h}  // 11000001-11111101-01110111-10000111
// CHECK-INST: sdot    za.d[w11, 7, vgx4], { z28.h - z31.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x87,0x77,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd7787 <unknown>

sdot    za.d[w11, 7], {z28.h - z31.h}, {z28.h - z31.h}  // 11000001-11111101-01110111-10000111
// CHECK-INST: sdot    za.d[w11, 7, vgx4], { z28.h - z31.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x87,0x77,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd7787 <unknown>

sdot    za.d[w8, 5, vgx4], {z16.h - z19.h}, {z16.h - z19.h}  // 11000001-11110001-00010110-00000101
// CHECK-INST: sdot    za.d[w8, 5, vgx4], { z16.h - z19.h }, { z16.h - z19.h }
// CHECK-ENCODING: [0x05,0x16,0xf1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f11605 <unknown>

sdot    za.d[w8, 5], {z16.h - z19.h}, {z16.h - z19.h}  // 11000001-11110001-00010110-00000101
// CHECK-INST: sdot    za.d[w8, 5, vgx4], { z16.h - z19.h }, { z16.h - z19.h }
// CHECK-ENCODING: [0x05,0x16,0xf1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f11605 <unknown>

sdot    za.d[w8, 1, vgx4], {z0.h - z3.h}, {z28.h - z31.h}  // 11000001-11111101-00010100-00000001
// CHECK-INST: sdot    za.d[w8, 1, vgx4], { z0.h - z3.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x01,0x14,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd1401 <unknown>

sdot    za.d[w8, 1], {z0.h - z3.h}, {z28.h - z31.h}  // 11000001-11111101-00010100-00000001
// CHECK-INST: sdot    za.d[w8, 1, vgx4], { z0.h - z3.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x01,0x14,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd1401 <unknown>

sdot    za.d[w10, 0, vgx4], {z16.h - z19.h}, {z20.h - z23.h}  // 11000001-11110101-01010110-00000000
// CHECK-INST: sdot    za.d[w10, 0, vgx4], { z16.h - z19.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x00,0x56,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f55600 <unknown>

sdot    za.d[w10, 0], {z16.h - z19.h}, {z20.h - z23.h}  // 11000001-11110101-01010110-00000000
// CHECK-INST: sdot    za.d[w10, 0, vgx4], { z16.h - z19.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x00,0x56,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f55600 <unknown>

sdot    za.d[w8, 0, vgx4], {z12.h - z15.h}, {z0.h - z3.h}  // 11000001-11100001-00010101-10000000
// CHECK-INST: sdot    za.d[w8, 0, vgx4], { z12.h - z15.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x80,0x15,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11580 <unknown>

sdot    za.d[w8, 0], {z12.h - z15.h}, {z0.h - z3.h}  // 11000001-11100001-00010101-10000000
// CHECK-INST: sdot    za.d[w8, 0, vgx4], { z12.h - z15.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x80,0x15,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11580 <unknown>

sdot    za.d[w10, 1, vgx4], {z0.h - z3.h}, {z24.h - z27.h}  // 11000001-11111001-01010100-00000001
// CHECK-INST: sdot    za.d[w10, 1, vgx4], { z0.h - z3.h }, { z24.h - z27.h }
// CHECK-ENCODING: [0x01,0x54,0xf9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f95401 <unknown>

sdot    za.d[w10, 1], {z0.h - z3.h}, {z24.h - z27.h}  // 11000001-11111001-01010100-00000001
// CHECK-INST: sdot    za.d[w10, 1, vgx4], { z0.h - z3.h }, { z24.h - z27.h }
// CHECK-ENCODING: [0x01,0x54,0xf9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f95401 <unknown>

sdot    za.d[w8, 5, vgx4], {z20.h - z23.h}, {z28.h - z31.h}  // 11000001-11111101-00010110-10000101
// CHECK-INST: sdot    za.d[w8, 5, vgx4], { z20.h - z23.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x85,0x16,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd1685 <unknown>

sdot    za.d[w8, 5], {z20.h - z23.h}, {z28.h - z31.h}  // 11000001-11111101-00010110-10000101
// CHECK-INST: sdot    za.d[w8, 5, vgx4], { z20.h - z23.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x85,0x16,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd1685 <unknown>

sdot    za.d[w11, 2, vgx4], {z8.h - z11.h}, {z0.h - z3.h}  // 11000001-11100001-01110101-00000010
// CHECK-INST: sdot    za.d[w11, 2, vgx4], { z8.h - z11.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x02,0x75,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e17502 <unknown>

sdot    za.d[w11, 2], {z8.h - z11.h}, {z0.h - z3.h}  // 11000001-11100001-01110101-00000010
// CHECK-INST: sdot    za.d[w11, 2, vgx4], { z8.h - z11.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x02,0x75,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e17502 <unknown>

sdot    za.d[w9, 7, vgx4], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-11101001-00110101-10000111
// CHECK-INST: sdot    za.d[w9, 7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x87,0x35,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e93587 <unknown>

sdot    za.d[w9, 7], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-11101001-00110101-10000111
// CHECK-INST: sdot    za.d[w9, 7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x87,0x35,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e93587 <unknown>

