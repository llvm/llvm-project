// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+sme-i16i64 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2,+sme-i16i64 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2,+sme-i16i64 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2,+sme-i16i64 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+sme-i16i64 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2,+sme-i16i64 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


add     {z0.h, z1.h}, {z0.h, z1.h}, z0.h  // 11000001-01100000-10100011-00000000
// CHECK-INST: add     { z0.h, z1.h }, { z0.h, z1.h }, z0.h
// CHECK-ENCODING: [0x00,0xa3,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c160a300 <unknown>

add     {z20.h, z21.h}, {z20.h, z21.h}, z5.h  // 11000001-01100101-10100011-00010100
// CHECK-INST: add     { z20.h, z21.h }, { z20.h, z21.h }, z5.h
// CHECK-ENCODING: [0x14,0xa3,0x65,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c165a314 <unknown>

add     {z22.h, z23.h}, {z22.h, z23.h}, z8.h  // 11000001-01101000-10100011-00010110
// CHECK-INST: add     { z22.h, z23.h }, { z22.h, z23.h }, z8.h
// CHECK-ENCODING: [0x16,0xa3,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c168a316 <unknown>

add     {z30.h, z31.h}, {z30.h, z31.h}, z15.h  // 11000001-01101111-10100011-00011110
// CHECK-INST: add     { z30.h, z31.h }, { z30.h, z31.h }, z15.h
// CHECK-ENCODING: [0x1e,0xa3,0x6f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16fa31e <unknown>


add     za.s[w8, 0, vgx2], {z0.s, z1.s}  // 11000001-10100000-00011100-00010000
// CHECK-INST: add     za.s[w8, 0, vgx2], { z0.s, z1.s }
// CHECK-ENCODING: [0x10,0x1c,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a01c10 <unknown>

add     za.s[w8, 0], {z0.s, z1.s}  // 11000001-10100000-00011100-00010000
// CHECK-INST: add     za.s[w8, 0, vgx2], { z0.s, z1.s }
// CHECK-ENCODING: [0x10,0x1c,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a01c10 <unknown>

add     za.s[w10, 5, vgx2], {z10.s, z11.s}  // 11000001-10100000-01011101-01010101
// CHECK-INST: add     za.s[w10, 5, vgx2], { z10.s, z11.s }
// CHECK-ENCODING: [0x55,0x5d,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a05d55 <unknown>

add     za.s[w10, 5], {z10.s, z11.s}  // 11000001-10100000-01011101-01010101
// CHECK-INST: add     za.s[w10, 5, vgx2], { z10.s, z11.s }
// CHECK-ENCODING: [0x55,0x5d,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a05d55 <unknown>

add     za.s[w11, 7, vgx2], {z12.s, z13.s}  // 11000001-10100000-01111101-10010111
// CHECK-INST: add     za.s[w11, 7, vgx2], { z12.s, z13.s }
// CHECK-ENCODING: [0x97,0x7d,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a07d97 <unknown>

add     za.s[w11, 7], {z12.s, z13.s}  // 11000001-10100000-01111101-10010111
// CHECK-INST: add     za.s[w11, 7, vgx2], { z12.s, z13.s }
// CHECK-ENCODING: [0x97,0x7d,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a07d97 <unknown>

add     za.s[w11, 7, vgx2], {z30.s, z31.s}  // 11000001-10100000-01111111-11010111
// CHECK-INST: add     za.s[w11, 7, vgx2], { z30.s, z31.s }
// CHECK-ENCODING: [0xd7,0x7f,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a07fd7 <unknown>

add     za.s[w11, 7], {z30.s, z31.s}  // 11000001-10100000-01111111-11010111
// CHECK-INST: add     za.s[w11, 7, vgx2], { z30.s, z31.s }
// CHECK-ENCODING: [0xd7,0x7f,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a07fd7 <unknown>

add     za.s[w8, 5, vgx2], {z16.s, z17.s}  // 11000001-10100000-00011110-00010101
// CHECK-INST: add     za.s[w8, 5, vgx2], { z16.s, z17.s }
// CHECK-ENCODING: [0x15,0x1e,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a01e15 <unknown>

add     za.s[w8, 5], {z16.s, z17.s}  // 11000001-10100000-00011110-00010101
// CHECK-INST: add     za.s[w8, 5, vgx2], { z16.s, z17.s }
// CHECK-ENCODING: [0x15,0x1e,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a01e15 <unknown>

add     za.s[w8, 1, vgx2], {z0.s, z1.s}  // 11000001-10100000-00011100-00010001
// CHECK-INST: add     za.s[w8, 1, vgx2], { z0.s, z1.s }
// CHECK-ENCODING: [0x11,0x1c,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a01c11 <unknown>

add     za.s[w8, 1], {z0.s, z1.s}  // 11000001-10100000-00011100-00010001
// CHECK-INST: add     za.s[w8, 1, vgx2], { z0.s, z1.s }
// CHECK-ENCODING: [0x11,0x1c,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a01c11 <unknown>

add     za.s[w10, 0, vgx2], {z18.s, z19.s}  // 11000001-10100000-01011110-01010000
// CHECK-INST: add     za.s[w10, 0, vgx2], { z18.s, z19.s }
// CHECK-ENCODING: [0x50,0x5e,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a05e50 <unknown>

add     za.s[w10, 0], {z18.s, z19.s}  // 11000001-10100000-01011110-01010000
// CHECK-INST: add     za.s[w10, 0, vgx2], { z18.s, z19.s }
// CHECK-ENCODING: [0x50,0x5e,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a05e50 <unknown>

add     za.s[w8, 0], {z12.s, z13.s}  // 11000001-10100000-00011101-10010000
// CHECK-INST: add     za.s[w8, 0, vgx2], { z12.s, z13.s }
// CHECK-ENCODING: [0x90,0x1d,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a01d90 <unknown>

add     za.s[w10, 1, vgx2], {z0.s, z1.s}  // 11000001-10100000-01011100-00010001
// CHECK-INST: add     za.s[w10, 1, vgx2], { z0.s, z1.s }
// CHECK-ENCODING: [0x11,0x5c,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a05c11 <unknown>

add     za.s[w10, 1], {z0.s, z1.s}  // 11000001-10100000-01011100-00010001
// CHECK-INST: add     za.s[w10, 1, vgx2], { z0.s, z1.s }
// CHECK-ENCODING: [0x11,0x5c,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a05c11 <unknown>

add     za.s[w8, 5, vgx2], {z22.s, z23.s}  // 11000001-10100000-00011110-11010101
// CHECK-INST: add     za.s[w8, 5, vgx2], { z22.s, z23.s }
// CHECK-ENCODING: [0xd5,0x1e,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a01ed5 <unknown>

add     za.s[w8, 5], {z22.s, z23.s}  // 11000001-10100000-00011110-11010101
// CHECK-INST: add     za.s[w8, 5, vgx2], { z22.s, z23.s }
// CHECK-ENCODING: [0xd5,0x1e,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a01ed5 <unknown>

add     za.s[w11, 2, vgx2], {z8.s, z9.s}  // 11000001-10100000-01111101-00010010
// CHECK-INST: add     za.s[w11, 2, vgx2], { z8.s, z9.s }
// CHECK-ENCODING: [0x12,0x7d,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a07d12 <unknown>

add     za.s[w11, 2], {z8.s, z9.s}  // 11000001-10100000-01111101-00010010
// CHECK-INST: add     za.s[w11, 2, vgx2], { z8.s, z9.s }
// CHECK-ENCODING: [0x12,0x7d,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a07d12 <unknown>

add     za.s[w9, 7, vgx2], {z12.s, z13.s}  // 11000001-10100000-00111101-10010111
// CHECK-INST: add     za.s[w9, 7, vgx2], { z12.s, z13.s }
// CHECK-ENCODING: [0x97,0x3d,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a03d97 <unknown>

add     za.s[w9, 7], {z12.s, z13.s}  // 11000001-10100000-00111101-10010111
// CHECK-INST: add     za.s[w9, 7, vgx2], { z12.s, z13.s }
// CHECK-ENCODING: [0x97,0x3d,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a03d97 <unknown>
add     za.s[w8, 0, vgx2], {z0.s, z1.s}, z0.s  // 11000001-00100000-00011000-00010000
// CHECK-INST: add     za.s[w8, 0, vgx2], { z0.s, z1.s }, z0.s
// CHECK-ENCODING: [0x10,0x18,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1201810 <unknown>

add     za.s[w8, 0], {z0.s - z1.s}, z0.s  // 11000001-00100000-00011000-00010000
// CHECK-INST: add     za.s[w8, 0, vgx2], { z0.s, z1.s }, z0.s
// CHECK-ENCODING: [0x10,0x18,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1201810 <unknown>

add     za.s[w10, 5, vgx2], {z10.s, z11.s}, z5.s  // 11000001-00100101-01011001-01010101
// CHECK-INST: add     za.s[w10, 5, vgx2], { z10.s, z11.s }, z5.s
// CHECK-ENCODING: [0x55,0x59,0x25,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1255955 <unknown>

add     za.s[w10, 5], {z10.s - z11.s}, z5.s  // 11000001-00100101-01011001-01010101
// CHECK-INST: add     za.s[w10, 5, vgx2], { z10.s, z11.s }, z5.s
// CHECK-ENCODING: [0x55,0x59,0x25,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1255955 <unknown>

add     za.s[w11, 7, vgx2], {z13.s, z14.s}, z8.s  // 11000001-00101000-01111001-10110111
// CHECK-INST: add     za.s[w11, 7, vgx2], { z13.s, z14.s }, z8.s
// CHECK-ENCODING: [0xb7,0x79,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12879b7 <unknown>

add     za.s[w11, 7], {z13.s - z14.s}, z8.s  // 11000001-00101000-01111001-10110111
// CHECK-INST: add     za.s[w11, 7, vgx2], { z13.s, z14.s }, z8.s
// CHECK-ENCODING: [0xb7,0x79,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12879b7 <unknown>

add     za.s[w11, 7, vgx2], {z31.s, z0.s}, z15.s  // 11000001-00101111-01g111011-11110111
// CHECK-INST: add     za.s[w11, 7, vgx2], { z31.s, z0.s }, z15.s
// CHECK-ENCODING: [0xf7,0x7b,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12f7bf7 <unknown>

add     za.s[w11, 7], {z31.s - z0.s}, z15.s  // 11000001-00101111-01111011-11110111
// CHECK-INST: add     za.s[w11, 7, vgx2], { z31.s, z0.s }, z15.s
// CHECK-ENCODING: [0xf7,0x7b,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12f7bf7 <unknown>

add     za.s[w8, 5, vgx2], {z17.s, z18.s}, z0.s  // 11000001-00100000-00011010-00110101
// CHECK-INST: add     za.s[w8, 5, vgx2], { z17.s, z18.s }, z0.s
// CHECK-ENCODING: [0x35,0x1a,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1201a35 <unknown>

add     za.s[w8, 5], {z17.s - z18.s}, z0.s  // 11000001-00100000-00011010-00110101
// CHECK-INST: add     za.s[w8, 5, vgx2], { z17.s, z18.s }, z0.s
// CHECK-ENCODING: [0x35,0x1a,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1201a35 <unknown>

add     za.s[w8, 1, vgx2], {z1.s, z2.s}, z14.s  // 11000001-00101110-00011000-00110001
// CHECK-INST: add     za.s[w8, 1, vgx2], { z1.s, z2.s }, z14.s
// CHECK-ENCODING: [0x31,0x18,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e1831 <unknown>

add     za.s[w8, 1], {z1.s - z2.s}, z14.s  // 11000001-00101110-00011000-00110001
// CHECK-INST: add     za.s[w8, 1, vgx2], { z1.s, z2.s }, z14.s
// CHECK-ENCODING: [0x31,0x18,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e1831 <unknown>

add     za.s[w10, 0, vgx2], {z19.s, z20.s}, z4.s  // 11000001-00100100-01011010-01110000
// CHECK-INST: add     za.s[w10, 0, vgx2], { z19.s, z20.s }, z4.s
// CHECK-ENCODING: [0x70,0x5a,0x24,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1245a70 <unknown>

add     za.s[w10, 0], {z19.s - z20.s}, z4.s  // 11000001-00100100-01011010-01110000
// CHECK-INST: add     za.s[w10, 0, vgx2], { z19.s, z20.s }, z4.s
// CHECK-ENCODING: [0x70,0x5a,0x24,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1245a70 <unknown>

add     za.s[w8, 0, vgx2], {z12.s, z13.s}, z2.s  // 11000001-00100010-00011001-10010000
// CHECK-INST: add     za.s[w8, 0, vgx2], { z12.s, z13.s }, z2.s
// CHECK-ENCODING: [0x90,0x19,0x22,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1221990 <unknown>

add     za.s[w8, 0], {z12.s - z13.s}, z2.s  // 11000001-00100010-00011001-10010000
// CHECK-INST: add     za.s[w8, 0, vgx2], { z12.s, z13.s }, z2.s
// CHECK-ENCODING: [0x90,0x19,0x22,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1221990 <unknown>

add     za.s[w10, 1, vgx2], {z1.s, z2.s}, z10.s  // 11000001-00101010-01011000-00110001
// CHECK-INST: add     za.s[w10, 1, vgx2], { z1.s, z2.s }, z10.s
// CHECK-ENCODING: [0x31,0x58,0x2a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12a5831 <unknown>

add     za.s[w10, 1], {z1.s - z2.s}, z10.s  // 11000001-00101010-01011000-00110001
// CHECK-INST: add     za.s[w10, 1, vgx2], { z1.s, z2.s }, z10.s
// CHECK-ENCODING: [0x31,0x58,0x2a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12a5831 <unknown>

add     za.s[w8, 5, vgx2], {z22.s, z23.s}, z14.s  // 11000001-00101110-00011010-11010101
// CHECK-INST: add     za.s[w8, 5, vgx2], { z22.s, z23.s }, z14.s
// CHECK-ENCODING: [0xd5,0x1a,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e1ad5 <unknown>

add     za.s[w8, 5], {z22.s - z23.s}, z14.s  // 11000001-00101110-00011010-11010101
// CHECK-INST: add     za.s[w8, 5, vgx2], { z22.s, z23.s }, z14.s
// CHECK-ENCODING: [0xd5,0x1a,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e1ad5 <unknown>

add     za.s[w11, 2, vgx2], {z9.s, z10.s}, z1.s  // 11000001-00100001-01111001-00110010
// CHECK-INST: add     za.s[w11, 2, vgx2], { z9.s, z10.s }, z1.s
// CHECK-ENCODING: [0x32,0x79,0x21,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1217932 <unknown>

add     za.s[w11, 2], {z9.s - z10.s}, z1.s  // 11000001-00100001-01111001-00110010
// CHECK-INST: add     za.s[w11, 2, vgx2], { z9.s, z10.s }, z1.s
// CHECK-ENCODING: [0x32,0x79,0x21,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1217932 <unknown>

add     za.s[w9, 7, vgx2], {z12.s, z13.s}, z11.s  // 11000001-00101011-00111001-10010111
// CHECK-INST: add     za.s[w9, 7, vgx2], { z12.s, z13.s }, z11.s
// CHECK-ENCODING: [0x97,0x39,0x2b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12b3997 <unknown>

add     za.s[w9, 7], {z12.s - z13.s}, z11.s  // 11000001-00101011-00111001-10010111
// CHECK-INST: add     za.s[w9, 7, vgx2], { z12.s, z13.s }, z11.s
// CHECK-ENCODING: [0x97,0x39,0x2b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12b3997 <unknown>


add     {z0.s-z1.s}, {z0.s-z1.s}, z0.s  // 11000001-10100000-10100011-00000000
// CHECK-INST: add     { z0.s, z1.s }, { z0.s, z1.s }, z0.s
// CHECK-ENCODING: [0x00,0xa3,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a0a300 <unknown>

add     {z20.s-z21.s}, {z20.s-z21.s}, z5.s  // 11000001-10100101-10100011-00010100
// CHECK-INST: add     { z20.s, z21.s }, { z20.s, z21.s }, z5.s
// CHECK-ENCODING: [0x14,0xa3,0xa5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a5a314 <unknown>

add     {z22.s-z23.s}, {z22.s-z23.s}, z8.s  // 11000001-10101000-10100011-00010110
// CHECK-INST: add     { z22.s, z23.s }, { z22.s, z23.s }, z8.s
// CHECK-ENCODING: [0x16,0xa3,0xa8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a8a316 <unknown>

add     {z30.s-z31.s}, {z30.s-z31.s}, z15.s  // 11000001-10101111-10100011-00011110
// CHECK-INST: add     { z30.s, z31.s }, { z30.s, z31.s }, z15.s
// CHECK-ENCODING: [0x1e,0xa3,0xaf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1afa31e <unknown>


add     za.s[w8, 0, vgx2], {z0.s, z1.s}, {z0.s, z1.s}  // 11000001-10100000-00011000-00010000
// CHECK-INST: add     za.s[w8, 0, vgx2], { z0.s, z1.s }, { z0.s, z1.s }
// CHECK-ENCODING: [0x10,0x18,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a01810 <unknown>

add     za.s[w8, 0], {z0.s - z1.s}, {z0.s - z1.s}  // 11000001-10100000-00011000-00010000
// CHECK-INST: add     za.s[w8, 0, vgx2], { z0.s, z1.s }, { z0.s, z1.s }
// CHECK-ENCODING: [0x10,0x18,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a01810 <unknown>

add     za.s[w10, 5, vgx2], {z10.s, z11.s}, {z20.s, z21.s}  // 11000001-10110100-01011001-01010101
// CHECK-INST: add     za.s[w10, 5, vgx2], { z10.s, z11.s }, { z20.s, z21.s }
// CHECK-ENCODING: [0x55,0x59,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b45955 <unknown>

add     za.s[w10, 5], {z10.s - z11.s}, {z20.s - z21.s}  // 11000001-10110100-01011001-01010101
// CHECK-INST: add     za.s[w10, 5, vgx2], { z10.s, z11.s }, { z20.s, z21.s }
// CHECK-ENCODING: [0x55,0x59,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b45955 <unknown>

add     za.s[w11, 7, vgx2], {z12.s, z13.s}, {z8.s, z9.s}  // 11000001-10101000-01111001-10010111
// CHECK-INST: add     za.s[w11, 7, vgx2], { z12.s, z13.s }, { z8.s, z9.s }
// CHECK-ENCODING: [0x97,0x79,0xa8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a87997 <unknown>

add     za.s[w11, 7], {z12.s - z13.s}, {z8.s - z9.s}  // 11000001-10101000-01111001-10010111
// CHECK-INST: add     za.s[w11, 7, vgx2], { z12.s, z13.s }, { z8.s, z9.s }
// CHECK-ENCODING: [0x97,0x79,0xa8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a87997 <unknown>

add     za.s[w11, 7, vgx2], {z30.s, z31.s}, {z30.s, z31.s}  // 11000001-10111110-01111011-11010111
// CHECK-INST: add     za.s[w11, 7, vgx2], { z30.s, z31.s }, { z30.s, z31.s }
// CHECK-ENCODING: [0xd7,0x7b,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be7bd7 <unknown>

add     za.s[w11, 7], {z30.s - z31.s}, {z30.s - z31.s}  // 11000001-10111110-01111011-11010111
// CHECK-INST: add     za.s[w11, 7, vgx2], { z30.s, z31.s }, { z30.s, z31.s }
// CHECK-ENCODING: [0xd7,0x7b,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be7bd7 <unknown>

add     za.s[w8, 5, vgx2], {z16.s, z17.s}, {z16.s, z17.s}  // 11000001-10110000-00011010-00010101
// CHECK-INST: add     za.s[w8, 5, vgx2], { z16.s, z17.s }, { z16.s, z17.s }
// CHECK-ENCODING: [0x15,0x1a,0xb0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b01a15 <unknown>

add     za.s[w8, 5], {z16.s - z17.s}, {z16.s - z17.s}  // 11000001-10110000-00011010-00010101
// CHECK-INST: add     za.s[w8, 5, vgx2], { z16.s, z17.s }, { z16.s, z17.s }
// CHECK-ENCODING: [0x15,0x1a,0xb0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b01a15 <unknown>

add     za.s[w8, 1, vgx2], {z0.s, z1.s}, {z30.s, z31.s}  // 11000001-10111110-00011000-00010001
// CHECK-INST: add     za.s[w8, 1, vgx2], { z0.s, z1.s }, { z30.s, z31.s }
// CHECK-ENCODING: [0x11,0x18,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be1811 <unknown>

add     za.s[w8, 1], {z0.s - z1.s}, {z30.s - z31.s}  // 11000001-10111110-00011000-00010001
// CHECK-INST: add     za.s[w8, 1, vgx2], { z0.s, z1.s }, { z30.s, z31.s }
// CHECK-ENCODING: [0x11,0x18,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be1811 <unknown>

add     za.s[w10, 0, vgx2], {z18.s, z19.s}, {z20.s, z21.s}  // 11000001-10110100-01011010-01010000
// CHECK-INST: add     za.s[w10, 0, vgx2], { z18.s, z19.s }, { z20.s, z21.s }
// CHECK-ENCODING: [0x50,0x5a,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b45a50 <unknown>

add     za.s[w10, 0], {z18.s - z19.s}, {z20.s - z21.s}  // 11000001-10110100-01011010-01010000
// CHECK-INST: add     za.s[w10, 0, vgx2], { z18.s, z19.s }, { z20.s, z21.s }
// CHECK-ENCODING: [0x50,0x5a,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b45a50 <unknown>

add     za.s[w8, 0, vgx2], {z12.s, z13.s}, {z2.s, z3.s}  // 11000001-10100010-00011001-10010000
// CHECK-INST: add     za.s[w8, 0, vgx2], { z12.s, z13.s }, { z2.s, z3.s }
// CHECK-ENCODING: [0x90,0x19,0xa2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a21990 <unknown>

add     za.s[w8, 0], {z12.s - z13.s}, {z2.s - z3.s}  // 11000001-10100010-00011001-10010000
// CHECK-INST: add     za.s[w8, 0, vgx2], { z12.s, z13.s }, { z2.s, z3.s }
// CHECK-ENCODING: [0x90,0x19,0xa2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a21990 <unknown>

add     za.s[w10, 1, vgx2], {z0.s, z1.s}, {z26.s, z27.s}  // 11000001-10111010-01011000-00010001
// CHECK-INST: add     za.s[w10, 1, vgx2], { z0.s, z1.s }, { z26.s, z27.s }
// CHECK-ENCODING: [0x11,0x58,0xba,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ba5811 <unknown>

add     za.s[w10, 1], {z0.s - z1.s}, {z26.s - z27.s}  // 11000001-10111010-01011000-00010001
// CHECK-INST: add     za.s[w10, 1, vgx2], { z0.s, z1.s }, { z26.s, z27.s }
// CHECK-ENCODING: [0x11,0x58,0xba,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ba5811 <unknown>

add     za.s[w8, 5, vgx2], {z22.s, z23.s}, {z30.s, z31.s}  // 11000001-10111110-00011010-11010101
// CHECK-INST: add     za.s[w8, 5, vgx2], { z22.s, z23.s }, { z30.s, z31.s }
// CHECK-ENCODING: [0xd5,0x1a,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be1ad5 <unknown>

add     za.s[w8, 5], {z22.s - z23.s}, {z30.s - z31.s}  // 11000001-10111110-00011010-11010101
// CHECK-INST: add     za.s[w8, 5, vgx2], { z22.s, z23.s }, { z30.s, z31.s }
// CHECK-ENCODING: [0xd5,0x1a,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be1ad5 <unknown>

add     za.s[w11, 2, vgx2], {z8.s, z9.s}, {z0.s, z1.s}  // 11000001-10100000-01111001-00010010
// CHECK-INST: add     za.s[w11, 2, vgx2], { z8.s, z9.s }, { z0.s, z1.s }
// CHECK-ENCODING: [0x12,0x79,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a07912 <unknown>

add     za.s[w11, 2], {z8.s - z9.s}, {z0.s - z1.s}  // 11000001-10100000-01111001-00010010
// CHECK-INST: add     za.s[w11, 2, vgx2], { z8.s, z9.s }, { z0.s, z1.s }
// CHECK-ENCODING: [0x12,0x79,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a07912 <unknown>

add     za.s[w9, 7, vgx2], {z12.s, z13.s}, {z10.s, z11.s}  // 11000001-10101010-00111001-10010111
// CHECK-INST: add     za.s[w9, 7, vgx2], { z12.s, z13.s }, { z10.s, z11.s }
// CHECK-ENCODING: [0x97,0x39,0xaa,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1aa3997 <unknown>

add     za.s[w9, 7], {z12.s - z13.s}, {z10.s - z11.s}  // 11000001-10101010-00111001-10010111
// CHECK-INST: add     za.s[w9, 7, vgx2], { z12.s, z13.s }, { z10.s, z11.s }
// CHECK-ENCODING: [0x97,0x39,0xaa,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1aa3997 <unknown>


add     za.d[w8, 0, vgx2], {z0.d, z1.d}  // 11000001-11100000-00011100-00010000
// CHECK-INST: add     za.d[w8, 0, vgx2], { z0.d, z1.d }
// CHECK-ENCODING: [0x10,0x1c,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e01c10 <unknown>

add     za.d[w8, 0], {z0.d, z1.d}  // 11000001-11100000-00011100-00010000
// CHECK-INST: add     za.d[w8, 0, vgx2], { z0.d, z1.d }
// CHECK-ENCODING: [0x10,0x1c,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e01c10 <unknown>

add     za.d[w10, 5, vgx2], {z10.d, z11.d}  // 11000001-11100000-01011101-01010101
// CHECK-INST: add     za.d[w10, 5, vgx2], { z10.d, z11.d }
// CHECK-ENCODING: [0x55,0x5d,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e05d55 <unknown>

add     za.d[w10, 5], {z10.d, z11.d}  // 11000001-11100000-01011101-01010101
// CHECK-INST: add     za.d[w10, 5, vgx2], { z10.d, z11.d }
// CHECK-ENCODING: [0x55,0x5d,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e05d55 <unknown>

add     za.d[w11, 7, vgx2], {z12.d, z13.d}  // 11000001-11100000-01111101-10010111
// CHECK-INST: add     za.d[w11, 7, vgx2], { z12.d, z13.d }
// CHECK-ENCODING: [0x97,0x7d,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e07d97 <unknown>

add     za.d[w11, 7], {z12.d, z13.d}  // 11000001-11100000-01111101-10010111
// CHECK-INST: add     za.d[w11, 7, vgx2], { z12.d, z13.d }
// CHECK-ENCODING: [0x97,0x7d,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e07d97 <unknown>

add     za.d[w11, 7, vgx2], {z30.d, z31.d}  // 11000001-11100000-01111111-11010111
// CHECK-INST: add     za.d[w11, 7, vgx2], { z30.d, z31.d }
// CHECK-ENCODING: [0xd7,0x7f,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e07fd7 <unknown>

add     za.d[w11, 7], {z30.d, z31.d}  // 11000001-11100000-01111111-11010111
// CHECK-INST: add     za.d[w11, 7, vgx2], { z30.d, z31.d }
// CHECK-ENCODING: [0xd7,0x7f,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e07fd7 <unknown>

add     za.d[w8, 5, vgx2], {z16.d, z17.d}  // 11000001-11100000-00011110-00010101
// CHECK-INST: add     za.d[w8, 5, vgx2], { z16.d, z17.d }
// CHECK-ENCODING: [0x15,0x1e,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e01e15 <unknown>

add     za.d[w8, 5], {z16.d, z17.d}  // 11000001-11100000-00011110-00010101
// CHECK-INST: add     za.d[w8, 5, vgx2], { z16.d, z17.d }
// CHECK-ENCODING: [0x15,0x1e,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e01e15 <unknown>

add     za.d[w8, 1, vgx2], {z0.d, z1.d}  // 11000001-11100000-00011100-00010001
// CHECK-INST: add     za.d[w8, 1, vgx2], { z0.d, z1.d }
// CHECK-ENCODING: [0x11,0x1c,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e01c11 <unknown>

add     za.d[w8, 1], {z0.d, z1.d}  // 11000001-11100000-00011100-00010001
// CHECK-INST: add     za.d[w8, 1, vgx2], { z0.d, z1.d }
// CHECK-ENCODING: [0x11,0x1c,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e01c11 <unknown>

add     za.d[w10, 0, vgx2], {z18.d, z19.d}  // 11000001-11100000-01011110-01010000
// CHECK-INST: add     za.d[w10, 0, vgx2], { z18.d, z19.d }
// CHECK-ENCODING: [0x50,0x5e,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e05e50 <unknown>

add     za.d[w10, 0], {z18.d, z19.d}  // 11000001-11100000-01011110-01010000
// CHECK-INST: add     za.d[w10, 0, vgx2], { z18.d, z19.d }
// CHECK-ENCODING: [0x50,0x5e,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e05e50 <unknown>

add     za.d[w8, 0, vgx2], {z12.d, z13.d}  // 11000001-11100000-00011101-10010000
// CHECK-INST: add     za.d[w8, 0, vgx2], { z12.d, z13.d }
// CHECK-ENCODING: [0x90,0x1d,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e01d90 <unknown>

add     za.d[w8, 0], {z12.d, z13.d}  // 11000001-11100000-00011101-10010000
// CHECK-INST: add     za.d[w8, 0, vgx2], { z12.d, z13.d }
// CHECK-ENCODING: [0x90,0x1d,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e01d90 <unknown>

add     za.d[w10, 1, vgx2], {z0.d, z1.d}  // 11000001-11100000-01011100-00010001
// CHECK-INST: add     za.d[w10, 1, vgx2], { z0.d, z1.d }
// CHECK-ENCODING: [0x11,0x5c,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e05c11 <unknown>

add     za.d[w10, 1], {z0.d, z1.d}  // 11000001-11100000-01011100-00010001
// CHECK-INST: add     za.d[w10, 1, vgx2], { z0.d, z1.d }
// CHECK-ENCODING: [0x11,0x5c,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e05c11 <unknown>

add     za.d[w8, 5, vgx2], {z22.d, z23.d}  // 11000001-11100000-00011110-11010101
// CHECK-INST: add     za.d[w8, 5, vgx2], { z22.d, z23.d }
// CHECK-ENCODING: [0xd5,0x1e,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e01ed5 <unknown>

add     za.d[w8, 5], {z22.d, z23.d}  // 11000001-11100000-00011110-11010101
// CHECK-INST: add     za.d[w8, 5, vgx2], { z22.d, z23.d }
// CHECK-ENCODING: [0xd5,0x1e,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e01ed5 <unknown>

add     za.d[w11, 2, vgx2], {z8.d, z9.d}  // 11000001-11100000-01111101-00010010
// CHECK-INST: add     za.d[w11, 2, vgx2], { z8.d, z9.d }
// CHECK-ENCODING: [0x12,0x7d,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e07d12 <unknown>

add     za.d[w11, 2], {z8.d, z9.d}  // 11000001-11100000-01111101-00010010
// CHECK-INST: add     za.d[w11, 2, vgx2], { z8.d, z9.d }
// CHECK-ENCODING: [0x12,0x7d,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e07d12 <unknown>

add     za.d[w9, 7, vgx2], {z12.d, z13.d}  // 11000001-11100000-00111101-10010111
// CHECK-INST: add     za.d[w9, 7, vgx2], { z12.d, z13.d }
// CHECK-ENCODING: [0x97,0x3d,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e03d97 <unknown>

add     za.d[w9, 7], {z12.d, z13.d}  // 11000001-11100000-00111101-10010111
// CHECK-INST: add     za.d[w9, 7, vgx2], { z12.d, z13.d }
// CHECK-ENCODING: [0x97,0x3d,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e03d97 <unknown>


add     za.d[w8, 0, vgx2], {z0.d, z1.d}, z0.d  // 11000001-01100000-00011000-00010000
// CHECK-INST: add     za.d[w8, 0, vgx2], { z0.d, z1.d }, z0.d
// CHECK-ENCODING: [0x10,0x18,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1601810 <unknown>

add     za.d[w8, 0], {z0.d - z1.d}, z0.d  // 11000001-01100000-00011000-00010000
// CHECK-INST: add     za.d[w8, 0, vgx2], { z0.d, z1.d }, z0.d
// CHECK-ENCODING: [0x10,0x18,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1601810 <unknown>

add     za.d[w10, 5, vgx2], {z10.d, z11.d}, z5.d  // 11000001-01100101-01011001-01010101
// CHECK-INST: add     za.d[w10, 5, vgx2], { z10.d, z11.d }, z5.d
// CHECK-ENCODING: [0x55,0x59,0x65,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1655955 <unknown>

add     za.d[w10, 5], {z10.d - z11.d}, z5.d  // 11000001-01100101-01011001-01010101
// CHECK-INST: add     za.d[w10, 5, vgx2], { z10.d, z11.d }, z5.d
// CHECK-ENCODING: [0x55,0x59,0x65,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1655955 <unknown>

add     za.d[w11, 7, vgx2], {z13.d, z14.d}, z8.d  // 11000001-01101000-01111001-10110111
// CHECK-INST: add     za.d[w11, 7, vgx2], { z13.d, z14.d }, z8.d
// CHECK-ENCODING: [0xb7,0x79,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16879b7 <unknown>

add     za.d[w11, 7], {z13.d - z14.d}, z8.d  // 11000001-01101000-01111001-10110111
// CHECK-INST: add     za.d[w11, 7, vgx2], { z13.d, z14.d }, z8.d
// CHECK-ENCODING: [0xb7,0x79,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16879b7 <unknown>

add     za.d[w11, 7, vgx2], {z31.d, z0.d}, z15.d  // 11000001-01101111-01g111011-11110111
// CHECK-INST: add     za.d[w11, 7, vgx2], { z31.d, z0.d }, z15.d
// CHECK-ENCODING: [0xf7,0x7b,0x6f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16f7bf7 <unknown>

add     za.d[w11, 7], {z31.d - z0.d}, z15.d  // 11000001-01101111-01111011-11110111
// CHECK-INST: add     za.d[w11, 7, vgx2], { z31.d, z0.d }, z15.d
// CHECK-ENCODING: [0xf7,0x7b,0x6f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16f7bf7 <unknown>

add     za.d[w8, 5, vgx2], {z17.d, z18.d}, z0.d  // 11000001-01100000-00011010-00110101
// CHECK-INST: add     za.d[w8, 5, vgx2], { z17.d, z18.d }, z0.d
// CHECK-ENCODING: [0x35,0x1a,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1601a35 <unknown>

add     za.d[w8, 5], {z17.d - z18.d}, z0.d  // 11000001-01100000-00011010-00110101
// CHECK-INST: add     za.d[w8, 5, vgx2], { z17.d, z18.d }, z0.d
// CHECK-ENCODING: [0x35,0x1a,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1601a35 <unknown>

add     za.d[w8, 1, vgx2], {z1.d, z2.d}, z14.d  // 11000001-01101110-00011000-00110001
// CHECK-INST: add     za.d[w8, 1, vgx2], { z1.d, z2.d }, z14.d
// CHECK-ENCODING: [0x31,0x18,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e1831 <unknown>

add     za.d[w8, 1], {z1.d - z2.d}, z14.d  // 11000001-01101110-00011000-00110001
// CHECK-INST: add     za.d[w8, 1, vgx2], { z1.d, z2.d }, z14.d
// CHECK-ENCODING: [0x31,0x18,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e1831 <unknown>

add     za.d[w10, 0, vgx2], {z19.d, z20.d}, z4.d  // 11000001-01100100-01011010-01110000
// CHECK-INST: add     za.d[w10, 0, vgx2], { z19.d, z20.d }, z4.d
// CHECK-ENCODING: [0x70,0x5a,0x64,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1645a70 <unknown>

add     za.d[w10, 0], {z19.d - z20.d}, z4.d  // 11000001-01100100-01011010-01110000
// CHECK-INST: add     za.d[w10, 0, vgx2], { z19.d, z20.d }, z4.d
// CHECK-ENCODING: [0x70,0x5a,0x64,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1645a70 <unknown>

add     za.d[w8, 0, vgx2], {z12.d, z13.d}, z2.d  // 11000001-01100010-00011001-10010000
// CHECK-INST: add     za.d[w8, 0, vgx2], { z12.d, z13.d }, z2.d
// CHECK-ENCODING: [0x90,0x19,0x62,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1621990 <unknown>

add     za.d[w8, 0], {z12.d - z13.d}, z2.d  // 11000001-01100010-00011001-10010000
// CHECK-INST: add     za.d[w8, 0, vgx2], { z12.d, z13.d }, z2.d
// CHECK-ENCODING: [0x90,0x19,0x62,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1621990 <unknown>

add     za.d[w10, 1, vgx2], {z1.d, z2.d}, z10.d  // 11000001-01101010-01011000-00110001
// CHECK-INST: add     za.d[w10, 1, vgx2], { z1.d, z2.d }, z10.d
// CHECK-ENCODING: [0x31,0x58,0x6a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16a5831 <unknown>

add     za.d[w10, 1], {z1.d - z2.d}, z10.d  // 11000001-01101010-01011000-00110001
// CHECK-INST: add     za.d[w10, 1, vgx2], { z1.d, z2.d }, z10.d
// CHECK-ENCODING: [0x31,0x58,0x6a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16a5831 <unknown>

add     za.d[w8, 5, vgx2], {z22.d, z23.d}, z14.d  // 11000001-01101110-00011010-11010101
// CHECK-INST: add     za.d[w8, 5, vgx2], { z22.d, z23.d }, z14.d
// CHECK-ENCODING: [0xd5,0x1a,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e1ad5 <unknown>

add     za.d[w8, 5], {z22.d - z23.d}, z14.d  // 11000001-01101110-00011010-11010101
// CHECK-INST: add     za.d[w8, 5, vgx2], { z22.d, z23.d }, z14.d
// CHECK-ENCODING: [0xd5,0x1a,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e1ad5 <unknown>

add     za.d[w11, 2, vgx2], {z9.d, z10.d}, z1.d  // 11000001-01100001-01111001-00110010
// CHECK-INST: add     za.d[w11, 2, vgx2], { z9.d, z10.d }, z1.d
// CHECK-ENCODING: [0x32,0x79,0x61,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1617932 <unknown>

add     za.d[w11, 2], {z9.d - z10.d}, z1.d  // 11000001-01100001-01111001-00110010
// CHECK-INST: add     za.d[w11, 2, vgx2], { z9.d, z10.d }, z1.d
// CHECK-ENCODING: [0x32,0x79,0x61,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1617932 <unknown>

add     za.d[w9, 7, vgx2], {z12.d, z13.d}, z11.d  // 11000001-01101011-00111001-10010111
// CHECK-INST: add     za.d[w9, 7, vgx2], { z12.d, z13.d }, z11.d
// CHECK-ENCODING: [0x97,0x39,0x6b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16b3997 <unknown>

add     za.d[w9, 7], {z12.d - z13.d}, z11.d  // 11000001-01101011-00111001-10010111
// CHECK-INST: add     za.d[w9, 7, vgx2], { z12.d, z13.d }, z11.d
// CHECK-ENCODING: [0x97,0x39,0x6b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16b3997 <unknown>


add     {z0.d-z1.d}, {z0.d-z1.d}, z0.d  // 11000001-11100000-10100011-00000000
// CHECK-INST: add     { z0.d, z1.d }, { z0.d, z1.d }, z0.d
// CHECK-ENCODING: [0x00,0xa3,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e0a300 <unknown>

add     {z20.d-z21.d}, {z20.d-z21.d}, z5.d  // 11000001-11100101-10100011-00010100
// CHECK-INST: add     { z20.d, z21.d }, { z20.d, z21.d }, z5.d
// CHECK-ENCODING: [0x14,0xa3,0xe5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e5a314 <unknown>

add     {z22.d-z23.d}, {z22.d-z23.d}, z8.d  // 11000001-11101000-10100011-00010110
// CHECK-INST: add     { z22.d, z23.d }, { z22.d, z23.d }, z8.d
// CHECK-ENCODING: [0x16,0xa3,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e8a316 <unknown>

add     {z30.d-z31.d}, {z30.d-z31.d}, z15.d  // 11000001-11101111-10100011-00011110
// CHECK-INST: add     { z30.d, z31.d }, { z30.d, z31.d }, z15.d
// CHECK-ENCODING: [0x1e,0xa3,0xef,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1efa31e <unknown>


add     za.d[w8, 0, vgx2], {z0.d, z1.d}, {z0.d, z1.d}  // 11000001-11100000-00011000-00010000
// CHECK-INST: add     za.d[w8, 0, vgx2], { z0.d, z1.d }, { z0.d, z1.d }
// CHECK-ENCODING: [0x10,0x18,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e01810 <unknown>

add     za.d[w8, 0], {z0.d - z1.d}, {z0.d - z1.d}  // 11000001-11100000-00011000-00010000
// CHECK-INST: add     za.d[w8, 0, vgx2], { z0.d, z1.d }, { z0.d, z1.d }
// CHECK-ENCODING: [0x10,0x18,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e01810 <unknown>

add     za.d[w10, 5, vgx2], {z10.d, z11.d}, {z20.d, z21.d}  // 11000001-11110100-01011001-01010101
// CHECK-INST: add     za.d[w10, 5, vgx2], { z10.d, z11.d }, { z20.d, z21.d }
// CHECK-ENCODING: [0x55,0x59,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f45955 <unknown>

add     za.d[w10, 5], {z10.d - z11.d}, {z20.d - z21.d}  // 11000001-11110100-01011001-01010101
// CHECK-INST: add     za.d[w10, 5, vgx2], { z10.d, z11.d }, { z20.d, z21.d }
// CHECK-ENCODING: [0x55,0x59,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f45955 <unknown>

add     za.d[w11, 7, vgx2], {z12.d, z13.d}, {z8.d, z9.d}  // 11000001-11101000-01111001-10010111
// CHECK-INST: add     za.d[w11, 7, vgx2], { z12.d, z13.d }, { z8.d, z9.d }
// CHECK-ENCODING: [0x97,0x79,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e87997 <unknown>

add     za.d[w11, 7], {z12.d - z13.d}, {z8.d - z9.d}  // 11000001-11101000-01111001-10010111
// CHECK-INST: add     za.d[w11, 7, vgx2], { z12.d, z13.d }, { z8.d, z9.d }
// CHECK-ENCODING: [0x97,0x79,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e87997 <unknown>

add     za.d[w11, 7, vgx2], {z30.d, z31.d}, {z30.d, z31.d}  // 11000001-11111110-01111011-11010111
// CHECK-INST: add     za.d[w11, 7, vgx2], { z30.d, z31.d }, { z30.d, z31.d }
// CHECK-ENCODING: [0xd7,0x7b,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe7bd7 <unknown>

add     za.d[w11, 7], {z30.d - z31.d}, {z30.d - z31.d}  // 11000001-11111110-01111011-11010111
// CHECK-INST: add     za.d[w11, 7, vgx2], { z30.d, z31.d }, { z30.d, z31.d }
// CHECK-ENCODING: [0xd7,0x7b,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe7bd7 <unknown>

add     za.d[w8, 5, vgx2], {z16.d, z17.d}, {z16.d, z17.d}  // 11000001-11110000-00011010-00010101
// CHECK-INST: add     za.d[w8, 5, vgx2], { z16.d, z17.d }, { z16.d, z17.d }
// CHECK-ENCODING: [0x15,0x1a,0xf0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f01a15 <unknown>

add     za.d[w8, 5], {z16.d - z17.d}, {z16.d - z17.d}  // 11000001-11110000-00011010-00010101
// CHECK-INST: add     za.d[w8, 5, vgx2], { z16.d, z17.d }, { z16.d, z17.d }
// CHECK-ENCODING: [0x15,0x1a,0xf0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f01a15 <unknown>

add     za.d[w8, 1, vgx2], {z0.d, z1.d}, {z30.d, z31.d}  // 11000001-11111110-00011000-00010001
// CHECK-INST: add     za.d[w8, 1, vgx2], { z0.d, z1.d }, { z30.d, z31.d }
// CHECK-ENCODING: [0x11,0x18,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe1811 <unknown>

add     za.d[w8, 1], {z0.d - z1.d}, {z30.d - z31.d}  // 11000001-11111110-00011000-00010001
// CHECK-INST: add     za.d[w8, 1, vgx2], { z0.d, z1.d }, { z30.d, z31.d }
// CHECK-ENCODING: [0x11,0x18,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe1811 <unknown>


add     za.d[w10, 0, vgx2], {z18.d, z19.d}, {z20.d, z21.d}  // 11000001-11110100-01011010-01010000
// CHECK-INST: add     za.d[w10, 0, vgx2], { z18.d, z19.d }, { z20.d, z21.d }
// CHECK-ENCODING: [0x50,0x5a,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f45a50 <unknown>

add     za.d[w10, 0], {z18.d - z19.d}, {z20.d - z21.d}  // 11000001-11110100-01011010-01010000
// CHECK-INST: add     za.d[w10, 0, vgx2], { z18.d, z19.d }, { z20.d, z21.d }
// CHECK-ENCODING: [0x50,0x5a,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f45a50 <unknown>

add     za.d[w8, 0, vgx2], {z12.d, z13.d}, {z2.d, z3.d}  // 11000001-11100010-00011001-10010000
// CHECK-INST: add     za.d[w8, 0, vgx2], { z12.d, z13.d }, { z2.d, z3.d }
// CHECK-ENCODING: [0x90,0x19,0xe2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e21990 <unknown>

add     za.d[w8, 0], {z12.d - z13.d}, {z2.d - z3.d}  // 11000001-11100010-00011001-10010000
// CHECK-INST: add     za.d[w8, 0, vgx2], { z12.d, z13.d }, { z2.d, z3.d }
// CHECK-ENCODING: [0x90,0x19,0xe2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e21990 <unknown>

add     za.d[w10, 1, vgx2], {z0.d, z1.d}, {z26.d, z27.d}  // 11000001-11111010-01011000-00010001
// CHECK-INST: add     za.d[w10, 1, vgx2], { z0.d, z1.d }, { z26.d, z27.d }
// CHECK-ENCODING: [0x11,0x58,0xfa,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fa5811 <unknown>

add     za.d[w10, 1], {z0.d - z1.d}, {z26.d - z27.d}  // 11000001-11111010-01011000-00010001
// CHECK-INST: add     za.d[w10, 1, vgx2], { z0.d, z1.d }, { z26.d, z27.d }
// CHECK-ENCODING: [0x11,0x58,0xfa,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fa5811 <unknown>

add     za.d[w8, 5, vgx2], {z22.d, z23.d}, {z30.d, z31.d}  // 11000001-11111110-00011010-11010101
// CHECK-INST: add     za.d[w8, 5, vgx2], { z22.d, z23.d }, { z30.d, z31.d }
// CHECK-ENCODING: [0xd5,0x1a,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe1ad5 <unknown>

add     za.d[w8, 5], {z22.d - z23.d}, {z30.d - z31.d}  // 11000001-11111110-00011010-11010101
// CHECK-INST: add     za.d[w8, 5, vgx2], { z22.d, z23.d }, { z30.d, z31.d }
// CHECK-ENCODING: [0xd5,0x1a,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe1ad5 <unknown>

add     za.d[w11, 2, vgx2], {z8.d, z9.d}, {z0.d, z1.d}  // 11000001-11100000-01111001-00010010
// CHECK-INST: add     za.d[w11, 2, vgx2], { z8.d, z9.d }, { z0.d, z1.d }
// CHECK-ENCODING: [0x12,0x79,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e07912 <unknown>

add     za.d[w11, 2], {z8.d - z9.d}, {z0.d - z1.d}  // 11000001-11100000-01111001-00010010
// CHECK-INST: add     za.d[w11, 2, vgx2], { z8.d, z9.d }, { z0.d, z1.d }
// CHECK-ENCODING: [0x12,0x79,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e07912 <unknown>

add     za.d[w9, 7, vgx2], {z12.d, z13.d}, {z10.d, z11.d}  // 11000001-11101010-00111001-10010111
// CHECK-INST: add     za.d[w9, 7, vgx2], { z12.d, z13.d }, { z10.d, z11.d }
// CHECK-ENCODING: [0x97,0x39,0xea,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ea3997 <unknown>

add     za.d[w9, 7], {z12.d - z13.d}, {z10.d - z11.d}  // 11000001-11101010-00111001-10010111
// CHECK-INST: add     za.d[w9, 7, vgx2], { z12.d, z13.d }, { z10.d, z11.d }
// CHECK-ENCODING: [0x97,0x39,0xea,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ea3997 <unknown>


add     {z0.b-z1.b}, {z0.b-z1.b}, z0.b  // 11000001-00100000-10100011-00000000
// CHECK-INST: add     { z0.b, z1.b }, { z0.b, z1.b }, z0.b
// CHECK-ENCODING: [0x00,0xa3,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c120a300 <unknown>

add     {z20.b-z21.b}, {z20.b-z21.b}, z5.b  // 11000001-00100101-10100011-00010100
// CHECK-INST: add     { z20.b, z21.b }, { z20.b, z21.b }, z5.b
// CHECK-ENCODING: [0x14,0xa3,0x25,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c125a314 <unknown>

add     {z22.b-z23.b}, {z22.b-z23.b}, z8.b  // 11000001-00101000-10100011-00010110
// CHECK-INST: add     { z22.b, z23.b }, { z22.b, z23.b }, z8.b
// CHECK-ENCODING: [0x16,0xa3,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c128a316 <unknown>

add     {z30.b-z31.b}, {z30.b-z31.b}, z15.b  // 11000001-00101111-10100011-00011110
// CHECK-INST: add     { z30.b, z31.b }, { z30.b, z31.b }, z15.b
// CHECK-ENCODING: [0x1e,0xa3,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12fa31e <unknown>


add     {z0.h - z3.h}, {z0.h - z3.h}, z0.h  // 11000001-01100000-10101011-00000000
// CHECK-INST: add     { z0.h - z3.h }, { z0.h - z3.h }, z0.h
// CHECK-ENCODING: [0x00,0xab,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c160ab00 <unknown>

add     {z20.h - z23.h}, {z20.h - z23.h}, z5.h  // 11000001-01100101-10101011-00010100
// CHECK-INST: add     { z20.h - z23.h }, { z20.h - z23.h }, z5.h
// CHECK-ENCODING: [0x14,0xab,0x65,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c165ab14 <unknown>

add     {z20.h - z23.h}, {z20.h - z23.h}, z8.h  // 11000001-01101000-10101011-00010100
// CHECK-INST: add     { z20.h - z23.h }, { z20.h - z23.h }, z8.h
// CHECK-ENCODING: [0x14,0xab,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c168ab14 <unknown>

add     {z28.h - z31.h}, {z28.h - z31.h}, z15.h  // 11000001-01101111-10101011-00011100
// CHECK-INST: add     { z28.h - z31.h }, { z28.h - z31.h }, z15.h
// CHECK-ENCODING: [0x1c,0xab,0x6f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16fab1c <unknown>


add     za.s[w8, 0, vgx4], {z0.s - z3.s}  // 11000001-10100001-00011100-00010000
// CHECK-INST: add     za.s[w8, 0, vgx4], { z0.s - z3.s }
// CHECK-ENCODING: [0x10,0x1c,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11c10 <unknown>

add     za.s[w8, 0], {z0.s - z3.s}  // 11000001-10100001-00011100-00010000
// CHECK-INST: add     za.s[w8, 0, vgx4], { z0.s - z3.s }
// CHECK-ENCODING: [0x10,0x1c,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11c10 <unknown>

add     za.s[w10, 5, vgx4], {z8.s - z11.s}  // 11000001-10100001-01011101-00010101
// CHECK-INST: add     za.s[w10, 5, vgx4], { z8.s - z11.s }
// CHECK-ENCODING: [0x15,0x5d,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a15d15 <unknown>

add     za.s[w10, 5], {z8.s - z11.s}  // 11000001-10100001-01011101-00010101
// CHECK-INST: add     za.s[w10, 5, vgx4], { z8.s - z11.s }
// CHECK-ENCODING: [0x15,0x5d,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a15d15 <unknown>

add     za.s[w11, 7, vgx4], {z12.s - z15.s}  // 11000001-10100001-01111101-10010111
// CHECK-INST: add     za.s[w11, 7, vgx4], { z12.s - z15.s }
// CHECK-ENCODING: [0x97,0x7d,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a17d97 <unknown>

add     za.s[w11, 7], {z12.s - z15.s}  // 11000001-10100001-01111101-10010111
// CHECK-INST: add     za.s[w11, 7, vgx4], { z12.s - z15.s }
// CHECK-ENCODING: [0x97,0x7d,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a17d97 <unknown>

add     za.s[w11, 7, vgx4], {z28.s - z31.s}  // 11000001-10100001-01111111-10010111
// CHECK-INST: add     za.s[w11, 7, vgx4], { z28.s - z31.s }
// CHECK-ENCODING: [0x97,0x7f,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a17f97 <unknown>

add     za.s[w11, 7], {z28.s - z31.s}  // 11000001-10100001-01111111-10010111
// CHECK-INST: add     za.s[w11, 7, vgx4], { z28.s - z31.s }
// CHECK-ENCODING: [0x97,0x7f,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a17f97 <unknown>

add     za.s[w8, 5, vgx4], {z16.s - z19.s}  // 11000001-10100001-00011110-00010101
// CHECK-INST: add     za.s[w8, 5, vgx4], { z16.s - z19.s }
// CHECK-ENCODING: [0x15,0x1e,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11e15 <unknown>

add     za.s[w8, 5], {z16.s - z19.s}  // 11000001-10100001-00011110-00010101
// CHECK-INST: add     za.s[w8, 5, vgx4], { z16.s - z19.s }
// CHECK-ENCODING: [0x15,0x1e,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11e15 <unknown>

add     za.s[w8, 1, vgx4], {z0.s - z3.s}  // 11000001-10100001-00011100-00010001
// CHECK-INST: add     za.s[w8, 1, vgx4], { z0.s - z3.s }
// CHECK-ENCODING: [0x11,0x1c,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11c11 <unknown>

add     za.s[w8, 1], {z0.s - z3.s}  // 11000001-10100001-00011100-00010001
// CHECK-INST: add     za.s[w8, 1, vgx4], { z0.s - z3.s }
// CHECK-ENCODING: [0x11,0x1c,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11c11 <unknown>

add     za.s[w10, 0, vgx4], {z16.s - z19.s}  // 11000001-10100001-01011110-00010000
// CHECK-INST: add     za.s[w10, 0, vgx4], { z16.s - z19.s }
// CHECK-ENCODING: [0x10,0x5e,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a15e10 <unknown>

add     za.s[w10, 0], {z16.s - z19.s}  // 11000001-10100001-01011110-00010000
// CHECK-INST: add     za.s[w10, 0, vgx4], { z16.s - z19.s }
// CHECK-ENCODING: [0x10,0x5e,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a15e10 <unknown>

add     za.s[w8, 0, vgx4], {z12.s - z15.s}  // 11000001-10100001-00011101-10010000
// CHECK-INST: add     za.s[w8, 0, vgx4], { z12.s - z15.s }
// CHECK-ENCODING: [0x90,0x1d,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11d90 <unknown>

add     za.s[w8, 0], {z12.s - z15.s}  // 11000001-10100001-00011101-10010000
// CHECK-INST: add     za.s[w8, 0, vgx4], { z12.s - z15.s }
// CHECK-ENCODING: [0x90,0x1d,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11d90 <unknown>

add     za.s[w10, 1, vgx4], {z0.s - z3.s}  // 11000001-10100001-01011100-00010001
// CHECK-INST: add     za.s[w10, 1, vgx4], { z0.s - z3.s }
// CHECK-ENCODING: [0x11,0x5c,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a15c11 <unknown>

add     za.s[w10, 1], {z0.s - z3.s}  // 11000001-10100001-01011100-00010001
// CHECK-INST: add     za.s[w10, 1, vgx4], { z0.s - z3.s }
// CHECK-ENCODING: [0x11,0x5c,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a15c11 <unknown>

add     za.s[w8, 5, vgx4], {z20.s - z23.s}  // 11000001-10100001-00011110-10010101
// CHECK-INST: add     za.s[w8, 5, vgx4], { z20.s - z23.s }
// CHECK-ENCODING: [0x95,0x1e,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11e95 <unknown>

add     za.s[w8, 5], {z20.s - z23.s}  // 11000001-10100001-00011110-10010101
// CHECK-INST: add     za.s[w8, 5, vgx4], { z20.s - z23.s }
// CHECK-ENCODING: [0x95,0x1e,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11e95 <unknown>

add     za.s[w11, 2, vgx4], {z8.s - z11.s}  // 11000001-10100001-01111101-00010010
// CHECK-INST: add     za.s[w11, 2, vgx4], { z8.s - z11.s }
// CHECK-ENCODING: [0x12,0x7d,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a17d12 <unknown>

add     za.s[w11, 2], {z8.s - z11.s}  // 11000001-10100001-01111101-00010010
// CHECK-INST: add     za.s[w11, 2, vgx4], { z8.s - z11.s }
// CHECK-ENCODING: [0x12,0x7d,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a17d12 <unknown>

add     za.s[w9, 7, vgx4], {z12.s - z15.s}  // 11000001-10100001-00111101-10010111
// CHECK-INST: add     za.s[w9, 7, vgx4], { z12.s - z15.s }
// CHECK-ENCODING: [0x97,0x3d,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a13d97 <unknown>

add     za.s[w9, 7], {z12.s - z15.s}  // 11000001-10100001-00111101-10010111
// CHECK-INST: add     za.s[w9, 7, vgx4], { z12.s - z15.s }
// CHECK-ENCODING: [0x97,0x3d,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a13d97 <unknown>


add     za.s[w8, 0, vgx4], {z0.s - z3.s}, z0.s  // 11000001-00110000-00011000-00010000
// CHECK-INST: add     za.s[w8, 0, vgx4], { z0.s - z3.s }, z0.s
// CHECK-ENCODING: [0x10,0x18,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1301810 <unknown>

add     za.s[w8, 0], {z0.s - z3.s}, z0.s  // 11000001-00110000-00011000-00010000
// CHECK-INST: add     za.s[w8, 0, vgx4], { z0.s - z3.s }, z0.s
// CHECK-ENCODING: [0x10,0x18,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1301810 <unknown>

add     za.s[w10, 5, vgx4], {z10.s - z13.s}, z5.s  // 11000001-00110101-01011001-01010101
// CHECK-INST: add     za.s[w10, 5, vgx4], { z10.s - z13.s }, z5.s
// CHECK-ENCODING: [0x55,0x59,0x35,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1355955 <unknown>

add     za.s[w10, 5], {z10.s - z13.s}, z5.s  // 11000001-00110101-01011001-01010101
// CHECK-INST: add     za.s[w10, 5, vgx4], { z10.s - z13.s }, z5.s
// CHECK-ENCODING: [0x55,0x59,0x35,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1355955 <unknown>

add     za.s[w11, 7, vgx4], {z13.s - z16.s}, z8.s  // 11000001-00111000-01111001-10110111
// CHECK-INST: add     za.s[w11, 7, vgx4], { z13.s - z16.s }, z8.s
// CHECK-ENCODING: [0xb7,0x79,0x38,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13879b7 <unknown>

add     za.s[w11, 7], {z13.s - z16.s}, z8.s  // 11000001-00111000-01111001-10110111
// CHECK-INST: add     za.s[w11, 7, vgx4], { z13.s - z16.s }, z8.s
// CHECK-ENCODING: [0xb7,0x79,0x38,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13879b7 <unknown>

add     za.s[w11, 7, vgx4], {z31.s - z2.s}, z15.s  // 11000001-00111111-01111011-11110111
// CHECK-INST: add     za.s[w11, 7, vgx4], { z31.s, z0.s, z1.s, z2.s }, z15.s
// CHECK-ENCODING: [0xf7,0x7b,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13f7bf7 <unknown>

add     za.s[w11, 7], {z31.s - z2.s}, z15.s  // 11000001-00111111-01111011-11110111
// CHECK-INST: add     za.s[w11, 7, vgx4], { z31.s, z0.s, z1.s, z2.s }, z15.s
// CHECK-ENCODING: [0xf7,0x7b,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13f7bf7 <unknown>

add     za.s[w8, 5, vgx4], {z17.s - z20.s}, z0.s  // 11000001-00110000-00011010-00110101
// CHECK-INST: add     za.s[w8, 5, vgx4], { z17.s - z20.s }, z0.s
// CHECK-ENCODING: [0x35,0x1a,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1301a35 <unknown>

add     za.s[w8, 5], {z17.s - z20.s}, z0.s  // 11000001-00110000-00011010-00110101
// CHECK-INST: add     za.s[w8, 5, vgx4], { z17.s - z20.s }, z0.s
// CHECK-ENCODING: [0x35,0x1a,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1301a35 <unknown>

add     za.s[w8, 1, vgx4], {z1.s - z4.s}, z14.s  // 11000001-00111110-00011000-00110001
// CHECK-INST: add     za.s[w8, 1, vgx4], { z1.s - z4.s }, z14.s
// CHECK-ENCODING: [0x31,0x18,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e1831 <unknown>

add     za.s[w8, 1], {z1.s - z4.s}, z14.s  // 11000001-00111110-00011000-00110001
// CHECK-INST: add     za.s[w8, 1, vgx4], { z1.s - z4.s }, z14.s
// CHECK-ENCODING: [0x31,0x18,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e1831 <unknown>

add     za.s[w10, 0, vgx4], {z19.s - z22.s}, z4.s  // 11000001-00110100-01011010-01110000
// CHECK-INST: add     za.s[w10, 0, vgx4], { z19.s - z22.s }, z4.s
// CHECK-ENCODING: [0x70,0x5a,0x34,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1345a70 <unknown>

add     za.s[w10, 0], {z19.s - z22.s}, z4.s  // 11000001-00110100-01011010-01110000
// CHECK-INST: add     za.s[w10, 0, vgx4], { z19.s - z22.s }, z4.s
// CHECK-ENCODING: [0x70,0x5a,0x34,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1345a70 <unknown>

add     za.s[w8, 0, vgx4], {z12.s - z15.s}, z2.s  // 11000001-00110010-00011001-10010000
// CHECK-INST: add     za.s[w8, 0, vgx4], { z12.s - z15.s }, z2.s
// CHECK-ENCODING: [0x90,0x19,0x32,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1321990 <unknown>

add     za.s[w8, 0], {z12.s - z15.s}, z2.s  // 11000001-00110010-00011001-10010000
// CHECK-INST: add     za.s[w8, 0, vgx4], { z12.s - z15.s }, z2.s
// CHECK-ENCODING: [0x90,0x19,0x32,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1321990 <unknown>

add     za.s[w10, 1, vgx4], {z1.s - z4.s}, z10.s  // 11000001-00111010-01011000-00110001
// CHECK-INST: add     za.s[w10, 1, vgx4], { z1.s - z4.s }, z10.s
// CHECK-ENCODING: [0x31,0x58,0x3a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13a5831 <unknown>

add     za.s[w10, 1], {z1.s - z4.s}, z10.s  // 11000001-00111010-01011000-00110001
// CHECK-INST: add     za.s[w10, 1, vgx4], { z1.s - z4.s }, z10.s
// CHECK-ENCODING: [0x31,0x58,0x3a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13a5831 <unknown>

add     za.s[w8, 5, vgx4], {z22.s - z25.s}, z14.s  // 11000001-00111110-00011010-11010101
// CHECK-INST: add     za.s[w8, 5, vgx4], { z22.s - z25.s }, z14.s
// CHECK-ENCODING: [0xd5,0x1a,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e1ad5 <unknown>

add     za.s[w8, 5], {z22.s - z25.s}, z14.s  // 11000001-00111110-00011010-11010101
// CHECK-INST: add     za.s[w8, 5, vgx4], { z22.s - z25.s }, z14.s
// CHECK-ENCODING: [0xd5,0x1a,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e1ad5 <unknown>

add     za.s[w11, 2, vgx4], {z9.s - z12.s}, z1.s  // 11000001-00110001-01111001-00110010
// CHECK-INST: add     za.s[w11, 2, vgx4], { z9.s - z12.s }, z1.s
// CHECK-ENCODING: [0x32,0x79,0x31,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1317932 <unknown>

add     za.s[w11, 2], {z9.s - z12.s}, z1.s  // 11000001-00110001-01111001-00110010
// CHECK-INST: add     za.s[w11, 2, vgx4], { z9.s - z12.s }, z1.s
// CHECK-ENCODING: [0x32,0x79,0x31,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1317932 <unknown>

add     za.s[w9, 7, vgx4], {z12.s - z15.s}, z11.s  // 11000001-00111011-00111001-10010111
// CHECK-INST: add     za.s[w9, 7, vgx4], { z12.s - z15.s }, z11.s
// CHECK-ENCODING: [0x97,0x39,0x3b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13b3997 <unknown>

add     za.s[w9, 7], {z12.s - z15.s}, z11.s  // 11000001-00111011-00111001-10010111
// CHECK-INST: add     za.s[w9, 7, vgx4], { z12.s - z15.s }, z11.s
// CHECK-ENCODING: [0x97,0x39,0x3b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13b3997 <unknown>


add     {z0.s-z3.s}, {z0.s-z3.s}, z0.s  // 11000001-10100000-10101011-00000000
// CHECK-INST: add     { z0.s - z3.s }, { z0.s - z3.s }, z0.s
// CHECK-ENCODING: [0x00,0xab,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a0ab00 <unknown>

add     {z20.s-z23.s}, {z20.s-z23.s}, z5.s  // 11000001-10100101-10101011-00010100
// CHECK-INST: add     { z20.s - z23.s }, { z20.s - z23.s }, z5.s
// CHECK-ENCODING: [0x14,0xab,0xa5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a5ab14 <unknown>

add     {z20.s-z23.s}, {z20.s-z23.s}, z8.s  // 11000001-10101000-10101011-00010100
// CHECK-INST: add     { z20.s - z23.s }, { z20.s - z23.s }, z8.s
// CHECK-ENCODING: [0x14,0xab,0xa8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a8ab14 <unknown>

add     {z28.s-z31.s}, {z28.s-z31.s}, z15.s  // 11000001-10101111-10101011-00011100
// CHECK-INST: add     { z28.s - z31.s }, { z28.s - z31.s }, z15.s
// CHECK-ENCODING: [0x1c,0xab,0xaf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1afab1c <unknown>


add     za.s[w8, 0, vgx4], {z0.s-z3.s}, {z0.s-z3.s}  // 11000001-10100001-00011000-00010000
// CHECK-INST: add     za.s[w8, 0, vgx4], { z0.s - z3.s }, { z0.s - z3.s }
// CHECK-ENCODING: [0x10,0x18,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11810 <unknown>

add     za.s[w8, 0], {z0.s-z3.s}, {z0.s-z3.s}  // 11000001-10100001-00011000-00010000
// CHECK-INST: add     za.s[w8, 0, vgx4], { z0.s - z3.s }, { z0.s - z3.s }
// CHECK-ENCODING: [0x10,0x18,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11810 <unknown>

add     za.s[w10, 5, vgx4], {z8.s - z11.s}, {z20.s - z23.s}  // 11000001-10110101-01011001-00010101
// CHECK-INST: add     za.s[w10, 5, vgx4], { z8.s - z11.s }, { z20.s - z23.s }
// CHECK-ENCODING: [0x15,0x59,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b55915 <unknown>

add     za.s[w10, 5], {z8.s - z11.s}, {z20.s - z23.s}  // 11000001-10110101-01011001-00010101
// CHECK-INST: add     za.s[w10, 5, vgx4], { z8.s - z11.s }, { z20.s - z23.s }
// CHECK-ENCODING: [0x15,0x59,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b55915 <unknown>

add     za.s[w11, 7, vgx4], {z12.s - z15.s}, {z8.s - z11.s}  // 11000001-10101001-01111001-10010111
// CHECK-INST: add     za.s[w11, 7, vgx4], { z12.s - z15.s }, { z8.s - z11.s }
// CHECK-ENCODING: [0x97,0x79,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a97997 <unknown>

add     za.s[w11, 7], {z12.s - z15.s}, {z8.s - z11.s}  // 11000001-10101001-01111001-10010111
// CHECK-INST: add     za.s[w11, 7, vgx4], { z12.s - z15.s }, { z8.s - z11.s }
// CHECK-ENCODING: [0x97,0x79,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a97997 <unknown>

add     za.s[w11, 7, vgx4], {z28.s - z31.s}, {z28.s - z31.s}  // 11000001-10111101-01111011-10010111
// CHECK-INST: add     za.s[w11, 7, vgx4], { z28.s - z31.s }, { z28.s - z31.s }
// CHECK-ENCODING: [0x97,0x7b,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd7b97 <unknown>

add     za.s[w11, 7], {z28.s - z31.s}, {z28.s - z31.s}  // 11000001-10111101-01111011-10010111
// CHECK-INST: add     za.s[w11, 7, vgx4], { z28.s - z31.s }, { z28.s - z31.s }
// CHECK-ENCODING: [0x97,0x7b,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd7b97 <unknown>

add     za.s[w8, 5, vgx4], {z16.s - z19.s}, {z16.s - z19.s}  // 11000001-10110001-00011010-00010101
// CHECK-INST: add     za.s[w8, 5, vgx4], { z16.s - z19.s }, { z16.s - z19.s }
// CHECK-ENCODING: [0x15,0x1a,0xb1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b11a15 <unknown>

add     za.s[w8, 5], {z16.s - z19.s}, {z16.s - z19.s}  // 11000001-10110001-00011010-00010101
// CHECK-INST: add     za.s[w8, 5, vgx4], { z16.s - z19.s }, { z16.s - z19.s }
// CHECK-ENCODING: [0x15,0x1a,0xb1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b11a15 <unknown>

add     za.s[w8, 1, vgx4], {z0.s - z3.s}, {z28.s - z31.s}  // 11000001-10111101-00011000-00010001
// CHECK-INST: add     za.s[w8, 1, vgx4], { z0.s - z3.s }, { z28.s - z31.s }
// CHECK-ENCODING: [0x11,0x18,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd1811 <unknown>

add     za.s[w8, 1], {z0.s - z3.s}, {z28.s - z31.s}  // 11000001-10111101-00011000-00010001
// CHECK-INST: add     za.s[w8, 1, vgx4], { z0.s - z3.s }, { z28.s - z31.s }
// CHECK-ENCODING: [0x11,0x18,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd1811 <unknown>

add     za.s[w10, 0, vgx4], {z16.s - z19.s}, {z20.s - z23.s}  // 11000001-10110101-01011010-00010000
// CHECK-INST: add     za.s[w10, 0, vgx4], { z16.s - z19.s }, { z20.s - z23.s }
// CHECK-ENCODING: [0x10,0x5a,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b55a10 <unknown>

add     za.s[w10, 0], {z16.s - z19.s}, {z20.s - z23.s}  // 11000001-10110101-01011010-00010000
// CHECK-INST: add     za.s[w10, 0, vgx4], { z16.s - z19.s }, { z20.s - z23.s }
// CHECK-ENCODING: [0x10,0x5a,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b55a10 <unknown>

add     za.s[w8, 0, vgx4], {z12.s - z15.s}, {z0.s - z3.s}  // 11000001-10100001-00011001-10010000
// CHECK-INST: add     za.s[w8, 0, vgx4], { z12.s - z15.s }, { z0.s - z3.s }
// CHECK-ENCODING: [0x90,0x19,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11990 <unknown>

add     za.s[w8, 0], {z12.s - z15.s}, {z0.s - z3.s}  // 11000001-10100001-00011001-10010000
// CHECK-INST: add     za.s[w8, 0, vgx4], { z12.s - z15.s }, { z0.s - z3.s }
// CHECK-ENCODING: [0x90,0x19,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11990 <unknown>

add     za.s[w10, 1, vgx4], {z0.s - z3.s}, {z24.s - z27.s}  // 11000001-10111001-01011000-00010001
// CHECK-INST: add     za.s[w10, 1, vgx4], { z0.s - z3.s }, { z24.s - z27.s }
// CHECK-ENCODING: [0x11,0x58,0xb9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b95811 <unknown>

add     za.s[w10, 1], {z0.s - z3.s}, {z24.s - z27.s}  // 11000001-10111001-01011000-00010001
// CHECK-INST: add     za.s[w10, 1, vgx4], { z0.s - z3.s }, { z24.s - z27.s }
// CHECK-ENCODING: [0x11,0x58,0xb9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b95811 <unknown>

add     za.s[w8, 5, vgx4], {z20.s - z23.s}, {z28.s - z31.s}  // 11000001-10111101-00011010-10010101
// CHECK-INST: add     za.s[w8, 5, vgx4], { z20.s - z23.s }, { z28.s - z31.s }
// CHECK-ENCODING: [0x95,0x1a,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd1a95 <unknown>

add     za.s[w8, 5], {z20.s - z23.s}, {z28.s - z31.s}  // 11000001-10111101-00011010-10010101
// CHECK-INST: add     za.s[w8, 5, vgx4], { z20.s - z23.s }, { z28.s - z31.s }
// CHECK-ENCODING: [0x95,0x1a,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd1a95 <unknown>

add     za.s[w11, 2, vgx4], {z8.s - z11.s}, {z0.s - z3.s}  // 11000001-10100001-01111001-00010010
// CHECK-INST: add     za.s[w11, 2, vgx4], { z8.s - z11.s }, { z0.s - z3.s }
// CHECK-ENCODING: [0x12,0x79,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a17912 <unknown>

add     za.s[w11, 2], {z8.s - z11.s}, {z0.s - z3.s}  // 11000001-10100001-01111001-00010010
// CHECK-INST: add     za.s[w11, 2, vgx4], { z8.s - z11.s }, { z0.s - z3.s }
// CHECK-ENCODING: [0x12,0x79,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a17912 <unknown>

add     za.s[w9, 7, vgx4], {z12.s - z15.s}, {z8.s - z11.s}  // 11000001-10101001-00111001-10010111
// CHECK-INST: add     za.s[w9, 7, vgx4], { z12.s - z15.s }, { z8.s - z11.s }
// CHECK-ENCODING: [0x97,0x39,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a93997 <unknown>

add     za.s[w9, 7], {z12.s - z15.s}, {z8.s - z11.s}  // 11000001-10101001-00111001-10010111
// CHECK-INST: add     za.s[w9, 7, vgx4], { z12.s - z15.s }, { z8.s - z11.s }
// CHECK-ENCODING: [0x97,0x39,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a93997 <unknown>


add     za.d[w8, 0, vgx4], {z0.d - z3.d}  // 11000001-11100001-00011100-00010000
// CHECK-INST: add     za.d[w8, 0, vgx4], { z0.d - z3.d }
// CHECK-ENCODING: [0x10,0x1c,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11c10 <unknown>

add     za.d[w8, 0], {z0.d - z3.d}  // 11000001-11100001-00011100-00010000
// CHECK-INST: add     za.d[w8, 0, vgx4], { z0.d - z3.d }
// CHECK-ENCODING: [0x10,0x1c,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11c10 <unknown>

add     za.d[w10, 5, vgx4], {z8.d - z11.d}  // 11000001-11100001-01011101-00010101
// CHECK-INST: add     za.d[w10, 5, vgx4], { z8.d - z11.d }
// CHECK-ENCODING: [0x15,0x5d,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e15d15 <unknown>

add     za.d[w10, 5], {z8.d - z11.d}  // 11000001-11100001-01011101-00010101
// CHECK-INST: add     za.d[w10, 5, vgx4], { z8.d - z11.d }
// CHECK-ENCODING: [0x15,0x5d,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e15d15 <unknown>

add     za.d[w11, 7, vgx4], {z12.d - z15.d}  // 11000001-11100001-01111101-10010111
// CHECK-INST: add     za.d[w11, 7, vgx4], { z12.d - z15.d }
// CHECK-ENCODING: [0x97,0x7d,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e17d97 <unknown>

add     za.d[w11, 7], {z12.d - z15.d}  // 11000001-11100001-01111101-10010111
// CHECK-INST: add     za.d[w11, 7, vgx4], { z12.d - z15.d }
// CHECK-ENCODING: [0x97,0x7d,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e17d97 <unknown>

add     za.d[w11, 7, vgx4], {z28.d - z31.d}  // 11000001-11100001-01111111-10010111
// CHECK-INST: add     za.d[w11, 7, vgx4], { z28.d - z31.d }
// CHECK-ENCODING: [0x97,0x7f,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e17f97 <unknown>

add     za.d[w11, 7], {z28.d - z31.d}  // 11000001-11100001-01111111-10010111
// CHECK-INST: add     za.d[w11, 7, vgx4], { z28.d - z31.d }
// CHECK-ENCODING: [0x97,0x7f,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e17f97 <unknown>

add     za.d[w8, 5, vgx4], {z16.d - z19.d}  // 11000001-11100001-00011110-00010101
// CHECK-INST: add     za.d[w8, 5, vgx4], { z16.d - z19.d }
// CHECK-ENCODING: [0x15,0x1e,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11e15 <unknown>

add     za.d[w8, 5], {z16.d - z19.d}  // 11000001-11100001-00011110-00010101
// CHECK-INST: add     za.d[w8, 5, vgx4], { z16.d - z19.d }
// CHECK-ENCODING: [0x15,0x1e,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11e15 <unknown>

add     za.d[w8, 1, vgx4], {z0.d - z3.d}  // 11000001-11100001-00011100-00010001
// CHECK-INST: add     za.d[w8, 1, vgx4], { z0.d - z3.d }
// CHECK-ENCODING: [0x11,0x1c,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11c11 <unknown>

add     za.d[w8, 1], {z0.d - z3.d}  // 11000001-11100001-00011100-00010001
// CHECK-INST: add     za.d[w8, 1, vgx4], { z0.d - z3.d }
// CHECK-ENCODING: [0x11,0x1c,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11c11 <unknown>

add     za.d[w10, 0, vgx4], {z16.d - z19.d}  // 11000001-11100001-01011110-00010000
// CHECK-INST: add     za.d[w10, 0, vgx4], { z16.d - z19.d }
// CHECK-ENCODING: [0x10,0x5e,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e15e10 <unknown>

add     za.d[w10, 0], {z16.d - z19.d}  // 11000001-11100001-01011110-00010000
// CHECK-INST: add     za.d[w10, 0, vgx4], { z16.d - z19.d }
// CHECK-ENCODING: [0x10,0x5e,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e15e10 <unknown>

add     za.d[w8, 0, vgx4], {z12.d - z15.d}  // 11000001-11100001-00011101-10010000
// CHECK-INST: add     za.d[w8, 0, vgx4], { z12.d - z15.d }
// CHECK-ENCODING: [0x90,0x1d,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11d90 <unknown>

add     za.d[w8, 0], {z12.d - z15.d}  // 11000001-11100001-00011101-10010000
// CHECK-INST: add     za.d[w8, 0, vgx4], { z12.d - z15.d }
// CHECK-ENCODING: [0x90,0x1d,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11d90 <unknown>

add     za.d[w10, 1, vgx4], {z0.d - z3.d}  // 11000001-11100001-01011100-00010001
// CHECK-INST: add     za.d[w10, 1, vgx4], { z0.d - z3.d }
// CHECK-ENCODING: [0x11,0x5c,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e15c11 <unknown>

add     za.d[w10, 1], {z0.d - z3.d}  // 11000001-11100001-01011100-00010001
// CHECK-INST: add     za.d[w10, 1, vgx4], { z0.d - z3.d }
// CHECK-ENCODING: [0x11,0x5c,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e15c11 <unknown>

add     za.d[w8, 5, vgx4], {z20.d - z23.d}  // 11000001-11100001-00011110-10010101
// CHECK-INST: add     za.d[w8, 5, vgx4], { z20.d - z23.d }
// CHECK-ENCODING: [0x95,0x1e,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11e95 <unknown>

add     za.d[w8, 5], {z20.d - z23.d}  // 11000001-11100001-00011110-10010101
// CHECK-INST: add     za.d[w8, 5, vgx4], { z20.d - z23.d }
// CHECK-ENCODING: [0x95,0x1e,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11e95 <unknown>

add     za.d[w11, 2, vgx4], {z8.d - z11.d}  // 11000001-11100001-01111101-00010010
// CHECK-INST: add     za.d[w11, 2, vgx4], { z8.d - z11.d }
// CHECK-ENCODING: [0x12,0x7d,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e17d12 <unknown>

add     za.d[w11, 2], {z8.d - z11.d}  // 11000001-11100001-01111101-00010010
// CHECK-INST: add     za.d[w11, 2, vgx4], { z8.d - z11.d }
// CHECK-ENCODING: [0x12,0x7d,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e17d12 <unknown>

add     za.d[w9, 7, vgx4], {z12.d - z15.d}  // 11000001-11100001-00111101-10010111
// CHECK-INST: add     za.d[w9, 7, vgx4], { z12.d - z15.d }
// CHECK-ENCODING: [0x97,0x3d,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e13d97 <unknown>

add     za.d[w9, 7], {z12.d - z15.d}  // 11000001-11100001-00111101-10010111
// CHECK-INST: add     za.d[w9, 7, vgx4], { z12.d - z15.d }
// CHECK-ENCODING: [0x97,0x3d,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e13d97 <unknown>


add     za.d[w8, 0, vgx4], {z0.d - z3.d}, z0.d  // 11000001-01110000-00011000-00010000
// CHECK-INST: add     za.d[w8, 0, vgx4], { z0.d - z3.d }, z0.d
// CHECK-ENCODING: [0x10,0x18,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1701810 <unknown>

add     za.d[w8, 0], {z0.d - z3.d}, z0.d  // 11000001-01110000-00011000-00010000
// CHECK-INST: add     za.d[w8, 0, vgx4], { z0.d - z3.d }, z0.d
// CHECK-ENCODING: [0x10,0x18,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1701810 <unknown>

add     za.d[w10, 5, vgx4], {z10.d - z13.d}, z5.d  // 11000001-01110101-01011001-01010101
// CHECK-INST: add     za.d[w10, 5, vgx4], { z10.d - z13.d }, z5.d
// CHECK-ENCODING: [0x55,0x59,0x75,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1755955 <unknown>

add     za.d[w10, 5], {z10.d - z13.d}, z5.d  // 11000001-01110101-01011001-01010101
// CHECK-INST: add     za.d[w10, 5, vgx4], { z10.d - z13.d }, z5.d
// CHECK-ENCODING: [0x55,0x59,0x75,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1755955 <unknown>

add     za.d[w11, 7, vgx4], {z13.d - z16.d}, z8.d  // 11000001-01111000-01111001-10110111
// CHECK-INST: add     za.d[w11, 7, vgx4], { z13.d - z16.d }, z8.d
// CHECK-ENCODING: [0xb7,0x79,0x78,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17879b7 <unknown>

add     za.d[w11, 7], {z13.d - z16.d}, z8.d  // 11000001-01111000-01111001-10110111
// CHECK-INST: add     za.d[w11, 7, vgx4], { z13.d - z16.d }, z8.d
// CHECK-ENCODING: [0xb7,0x79,0x78,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17879b7 <unknown>

add     za.d[w11, 7, vgx4], {z31.d - z2.d}, z15.d  // 11000001-01111111-01111011-11110111
// CHECK-INST: add     za.d[w11, 7, vgx4], { z31.d, z0.d, z1.d, z2.d }, z15.d
// CHECK-ENCODING: [0xf7,0x7b,0x7f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17f7bf7 <unknown>

add     za.d[w11, 7], {z31.d - z2.d}, z15.d  // 11000001-01111111-01111011-11110111
// CHECK-INST: add     za.d[w11, 7, vgx4], { z31.d, z0.d, z1.d, z2.d }, z15.d
// CHECK-ENCODING: [0xf7,0x7b,0x7f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17f7bf7 <unknown>

add     za.d[w8, 5, vgx4], {z17.d - z20.d}, z0.d  // 11000001-01110000-00011010-00110101
// CHECK-INST: add     za.d[w8, 5, vgx4], { z17.d - z20.d }, z0.d
// CHECK-ENCODING: [0x35,0x1a,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1701a35 <unknown>

add     za.d[w8, 5], {z17.d - z20.d}, z0.d  // 11000001-01110000-00011010-00110101
// CHECK-INST: add     za.d[w8, 5, vgx4], { z17.d - z20.d }, z0.d
// CHECK-ENCODING: [0x35,0x1a,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1701a35 <unknown>

add     za.d[w8, 1, vgx4], {z1.d - z4.d}, z14.d  // 11000001-01111110-00011000-00110001
// CHECK-INST: add     za.d[w8, 1, vgx4], { z1.d - z4.d }, z14.d
// CHECK-ENCODING: [0x31,0x18,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17e1831 <unknown>

add     za.d[w8, 1], {z1.d - z4.d}, z14.d  // 11000001-01111110-00011000-00110001
// CHECK-INST: add     za.d[w8, 1, vgx4], { z1.d - z4.d }, z14.d
// CHECK-ENCODING: [0x31,0x18,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17e1831 <unknown>

add     za.d[w10, 0, vgx4], {z19.d - z22.d}, z4.d  // 11000001-01110100-01011010-01110000
// CHECK-INST: add     za.d[w10, 0, vgx4], { z19.d - z22.d }, z4.d
// CHECK-ENCODING: [0x70,0x5a,0x74,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1745a70 <unknown>

add     za.d[w10, 0], {z19.d - z22.d}, z4.d  // 11000001-01110100-01011010-01110000
// CHECK-INST: add     za.d[w10, 0, vgx4], { z19.d - z22.d }, z4.d
// CHECK-ENCODING: [0x70,0x5a,0x74,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1745a70 <unknown>

add     za.d[w8, 0, vgx4], {z12.d - z15.d}, z2.d  // 11000001-01110010-00011001-10010000
// CHECK-INST: add     za.d[w8, 0, vgx4], { z12.d - z15.d }, z2.d
// CHECK-ENCODING: [0x90,0x19,0x72,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1721990 <unknown>

add     za.d[w8, 0], {z12.d - z15.d}, z2.d  // 11000001-01110010-00011001-10010000
// CHECK-INST: add     za.d[w8, 0, vgx4], { z12.d - z15.d }, z2.d
// CHECK-ENCODING: [0x90,0x19,0x72,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1721990 <unknown>

add     za.d[w10, 1, vgx4], {z1.d - z4.d}, z10.d  // 11000001-01111010-01011000-00110001
// CHECK-INST: add     za.d[w10, 1, vgx4], { z1.d - z4.d }, z10.d
// CHECK-ENCODING: [0x31,0x58,0x7a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17a5831 <unknown>

add     za.d[w10, 1], {z1.d - z4.d}, z10.d  // 11000001-01111010-01011000-00110001
// CHECK-INST: add     za.d[w10, 1, vgx4], { z1.d - z4.d }, z10.d
// CHECK-ENCODING: [0x31,0x58,0x7a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17a5831 <unknown>

add     za.d[w8, 5, vgx4], {z22.d - z25.d}, z14.d  // 11000001-01111110-00011010-11010101
// CHECK-INST: add     za.d[w8, 5, vgx4], { z22.d - z25.d }, z14.d
// CHECK-ENCODING: [0xd5,0x1a,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17e1ad5 <unknown>

add     za.d[w8, 5], {z22.d - z25.d}, z14.d  // 11000001-01111110-00011010-11010101
// CHECK-INST: add     za.d[w8, 5, vgx4], { z22.d - z25.d }, z14.d
// CHECK-ENCODING: [0xd5,0x1a,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17e1ad5 <unknown>

add     za.d[w11, 2, vgx4], {z9.d - z12.d}, z1.d  // 11000001-01110001-01111001-00110010
// CHECK-INST: add     za.d[w11, 2, vgx4], { z9.d - z12.d }, z1.d
// CHECK-ENCODING: [0x32,0x79,0x71,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1717932 <unknown>

add     za.d[w11, 2], {z9.d - z12.d}, z1.d  // 11000001-01110001-01111001-00110010
// CHECK-INST: add     za.d[w11, 2, vgx4], { z9.d - z12.d }, z1.d
// CHECK-ENCODING: [0x32,0x79,0x71,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1717932 <unknown>

add     za.d[w9, 7, vgx4], {z12.d - z15.d}, z11.d  // 11000001-01111011-00111001-10010111
// CHECK-INST: add     za.d[w9, 7, vgx4], { z12.d - z15.d }, z11.d
// CHECK-ENCODING: [0x97,0x39,0x7b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17b3997 <unknown>

add     za.d[w9, 7], {z12.d - z15.d}, z11.d  // 11000001-01111011-00111001-10010111
// CHECK-INST: add     za.d[w9, 7, vgx4], { z12.d - z15.d }, z11.d
// CHECK-ENCODING: [0x97,0x39,0x7b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17b3997 <unknown>


add     za.d[w8, 0, vgx4], {z0.d - z3.d}, {z0.d - z3.d}  // 11000001-11100001-00011000-00010000
// CHECK-INST: add     za.d[w8, 0, vgx4], { z0.d - z3.d }, { z0.d - z3.d }
// CHECK-ENCODING: [0x10,0x18,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11810 <unknown>

add     za.d[w8, 0], {z0.d - z3.d}, {z0.d - z3.d}  // 11000001-11100001-00011000-00010000
// CHECK-INST: add     za.d[w8, 0, vgx4], { z0.d - z3.d }, { z0.d - z3.d }
// CHECK-ENCODING: [0x10,0x18,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11810 <unknown>

add     za.d[w10, 5, vgx4], {z8.d - z11.d}, {z20.d - z23.d}  // 11000001-11110101-01011001-00010101
// CHECK-INST: add     za.d[w10, 5, vgx4], { z8.d - z11.d }, { z20.d - z23.d }
// CHECK-ENCODING: [0x15,0x59,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f55915 <unknown>

add     za.d[w10, 5], {z8.d - z11.d}, {z20.d - z23.d}  // 11000001-11110101-01011001-00010101
// CHECK-INST: add     za.d[w10, 5, vgx4], { z8.d - z11.d }, { z20.d - z23.d }
// CHECK-ENCODING: [0x15,0x59,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f55915 <unknown>

add     za.d[w11, 7, vgx4], {z12.d - z15.d}, {z8.d - z11.d}  // 11000001-11101001-01111001-10010111
// CHECK-INST: add     za.d[w11, 7, vgx4], { z12.d - z15.d }, { z8.d - z11.d }
// CHECK-ENCODING: [0x97,0x79,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e97997 <unknown>

add     za.d[w11, 7], {z12.d - z15.d}, {z8.d - z11.d}  // 11000001-11101001-01111001-10010111
// CHECK-INST: add     za.d[w11, 7, vgx4], { z12.d - z15.d }, { z8.d - z11.d }
// CHECK-ENCODING: [0x97,0x79,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e97997 <unknown>

add     za.d[w11, 7, vgx4], {z28.d - z31.d}, {z28.d - z31.d}  // 11000001-11111101-01111011-10010111
// CHECK-INST: add     za.d[w11, 7, vgx4], { z28.d - z31.d }, { z28.d - z31.d }
// CHECK-ENCODING: [0x97,0x7b,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd7b97 <unknown>

add     za.d[w11, 7], {z28.d - z31.d}, {z28.d - z31.d}  // 11000001-11111101-01111011-10010111
// CHECK-INST: add     za.d[w11, 7, vgx4], { z28.d - z31.d }, { z28.d - z31.d }
// CHECK-ENCODING: [0x97,0x7b,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd7b97 <unknown>

add     za.d[w8, 5, vgx4], {z16.d - z19.d}, {z16.d - z19.d}  // 11000001-11110001-00011010-00010101
// CHECK-INST: add     za.d[w8, 5, vgx4], { z16.d - z19.d }, { z16.d - z19.d }
// CHECK-ENCODING: [0x15,0x1a,0xf1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f11a15 <unknown>

add     za.d[w8, 5], {z16.d - z19.d}, {z16.d - z19.d}  // 11000001-11110001-00011010-00010101
// CHECK-INST: add     za.d[w8, 5, vgx4], { z16.d - z19.d }, { z16.d - z19.d }
// CHECK-ENCODING: [0x15,0x1a,0xf1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f11a15 <unknown>

add     za.d[w8, 1, vgx4], {z0.d - z3.d}, {z28.d - z31.d}  // 11000001-11111101-00011000-00010001
// CHECK-INST: add     za.d[w8, 1, vgx4], { z0.d - z3.d }, { z28.d - z31.d }
// CHECK-ENCODING: [0x11,0x18,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd1811 <unknown>

add     za.d[w8, 1], {z0.d - z3.d}, {z28.d - z31.d}  // 11000001-11111101-00011000-00010001
// CHECK-INST: add     za.d[w8, 1, vgx4], { z0.d - z3.d }, { z28.d - z31.d }
// CHECK-ENCODING: [0x11,0x18,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd1811 <unknown>

add     za.d[w10, 0, vgx4], {z16.d - z19.d}, {z20.d - z23.d}  // 11000001-11110101-01011010-00010000
// CHECK-INST: add     za.d[w10, 0, vgx4], { z16.d - z19.d }, { z20.d - z23.d }
// CHECK-ENCODING: [0x10,0x5a,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f55a10 <unknown>

add     za.d[w10, 0], {z16.d - z19.d}, {z20.d - z23.d}  // 11000001-11110101-01011010-00010000
// CHECK-INST: add     za.d[w10, 0, vgx4], { z16.d - z19.d }, { z20.d - z23.d }
// CHECK-ENCODING: [0x10,0x5a,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f55a10 <unknown>

add     za.d[w8, 0, vgx4], {z12.d - z15.d}, {z0.d - z3.d}  // 11000001-11100001-00011001-10010000
// CHECK-INST: add     za.d[w8, 0, vgx4], { z12.d - z15.d }, { z0.d - z3.d }
// CHECK-ENCODING: [0x90,0x19,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11990 <unknown>

add     za.d[w8, 0], {z12.d - z15.d}, {z0.d - z3.d}  // 11000001-11100001-00011001-10010000
// CHECK-INST: add     za.d[w8, 0, vgx4], { z12.d - z15.d }, { z0.d - z3.d }
// CHECK-ENCODING: [0x90,0x19,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11990 <unknown>

add     za.d[w10, 1, vgx4], {z0.d - z3.d}, {z24.d - z27.d}  // 11000001-11111001-01011000-00010001
// CHECK-INST: add     za.d[w10, 1, vgx4], { z0.d - z3.d }, { z24.d - z27.d }
// CHECK-ENCODING: [0x11,0x58,0xf9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f95811 <unknown>

add     za.d[w10, 1], {z0.d - z3.d}, {z24.d - z27.d}  // 11000001-11111001-01011000-00010001
// CHECK-INST: add     za.d[w10, 1, vgx4], { z0.d - z3.d }, { z24.d - z27.d }
// CHECK-ENCODING: [0x11,0x58,0xf9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f95811 <unknown>

add     za.d[w8, 5, vgx4], {z20.d - z23.d}, {z28.d - z31.d}  // 11000001-11111101-00011010-10010101
// CHECK-INST: add     za.d[w8, 5, vgx4], { z20.d - z23.d }, { z28.d - z31.d }
// CHECK-ENCODING: [0x95,0x1a,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd1a95 <unknown>

add     za.d[w8, 5], {z20.d - z23.d}, {z28.d - z31.d}  // 11000001-11111101-00011010-10010101
// CHECK-INST: add     za.d[w8, 5, vgx4], { z20.d - z23.d }, { z28.d - z31.d }
// CHECK-ENCODING: [0x95,0x1a,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd1a95 <unknown>

add     za.d[w11, 2, vgx4], {z8.d - z11.d}, {z0.d - z3.d}  // 11000001-11100001-01111001-00010010
// CHECK-INST: add     za.d[w11, 2, vgx4], { z8.d - z11.d }, { z0.d - z3.d }
// CHECK-ENCODING: [0x12,0x79,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e17912 <unknown>

add     za.d[w11, 2], {z8.d - z11.d}, {z0.d - z3.d}  // 11000001-11100001-01111001-00010010
// CHECK-INST: add     za.d[w11, 2, vgx4], { z8.d - z11.d }, { z0.d - z3.d }
// CHECK-ENCODING: [0x12,0x79,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e17912 <unknown>

add     za.d[w9, 7, vgx4], {z12.d - z15.d}, {z8.d - z11.d}  // 11000001-11101001-00111001-10010111
// CHECK-INST: add     za.d[w9, 7, vgx4], { z12.d - z15.d }, { z8.d - z11.d }
// CHECK-ENCODING: [0x97,0x39,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e93997 <unknown>

add     za.d[w9, 7], {z12.d - z15.d}, {z8.d - z11.d}  // 11000001-11101001-00111001-10010111
// CHECK-INST: add     za.d[w9, 7, vgx4], { z12.d - z15.d }, { z8.d - z11.d }
// CHECK-ENCODING: [0x97,0x39,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e93997 <unknown>


add     {z0.d-z3.d}, {z0.d-z3.d}, z0.d  // 11000001-11100000-10101011-00000000
// CHECK-INST: add     { z0.d - z3.d }, { z0.d - z3.d }, z0.d
// CHECK-ENCODING: [0x00,0xab,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e0ab00 <unknown>

add     {z20.d-z23.d}, {z20.d-z23.d}, z5.d  // 11000001-11100101-10101011-00010100
// CHECK-INST: add     { z20.d - z23.d }, { z20.d - z23.d }, z5.d
// CHECK-ENCODING: [0x14,0xab,0xe5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e5ab14 <unknown>

add     {z20.d-z23.d}, {z20.d-z23.d}, z8.d  // 11000001-11101000-10101011-00010100
// CHECK-INST: add     { z20.d - z23.d }, { z20.d - z23.d }, z8.d
// CHECK-ENCODING: [0x14,0xab,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e8ab14 <unknown>

add     {z28.d-z31.d}, {z28.d-z31.d}, z15.d  // 11000001-11101111-10101011-00011100
// CHECK-INST: add     { z28.d - z31.d }, { z28.d - z31.d }, z15.d
// CHECK-ENCODING: [0x1c,0xab,0xef,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1efab1c <unknown>


add     za.d[w8, 0, vgx4], {z0.d - z3.d}, {z0.d - z3.d}  // 11000001-11100001-00011000-00010000
// CHECK-INST: add     za.d[w8, 0, vgx4], { z0.d - z3.d }, { z0.d - z3.d }
// CHECK-ENCODING: [0x10,0x18,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11810 <unknown>

add     za.d[w8, 0], {z0.d - z3.d}, {z0.d - z3.d}  // 11000001-11100001-00011000-00010000
// CHECK-INST: add     za.d[w8, 0, vgx4], { z0.d - z3.d }, { z0.d - z3.d }
// CHECK-ENCODING: [0x10,0x18,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11810 <unknown>

add     za.d[w10, 5, vgx4], {z8.d - z11.d}, {z20.d - z23.d}  // 11000001-11110101-01011001-00010101
// CHECK-INST: add     za.d[w10, 5, vgx4], { z8.d - z11.d }, { z20.d - z23.d }
// CHECK-ENCODING: [0x15,0x59,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f55915 <unknown>

add     za.d[w10, 5], {z8.d - z11.d}, {z20.d - z23.d}  // 11000001-11110101-01011001-00010101
// CHECK-INST: add     za.d[w10, 5, vgx4], { z8.d - z11.d }, { z20.d - z23.d }
// CHECK-ENCODING: [0x15,0x59,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f55915 <unknown>

add     za.d[w11, 7, vgx4], {z12.d - z15.d}, {z8.d - z11.d}  // 11000001-11101001-01111001-10010111
// CHECK-INST: add     za.d[w11, 7, vgx4], { z12.d - z15.d }, { z8.d - z11.d }
// CHECK-ENCODING: [0x97,0x79,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e97997 <unknown>

add     za.d[w11, 7], {z12.d - z15.d}, {z8.d - z11.d}  // 11000001-11101001-01111001-10010111
// CHECK-INST: add     za.d[w11, 7, vgx4], { z12.d - z15.d }, { z8.d - z11.d }
// CHECK-ENCODING: [0x97,0x79,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e97997 <unknown>

add     za.d[w11, 7, vgx4], {z28.d - z31.d}, {z28.d - z31.d}  // 11000001-11111101-01111011-10010111
// CHECK-INST: add     za.d[w11, 7, vgx4], { z28.d - z31.d }, { z28.d - z31.d }
// CHECK-ENCODING: [0x97,0x7b,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd7b97 <unknown>

add     za.d[w11, 7], {z28.d - z31.d}, {z28.d - z31.d}  // 11000001-11111101-01111011-10010111
// CHECK-INST: add     za.d[w11, 7, vgx4], { z28.d - z31.d }, { z28.d - z31.d }
// CHECK-ENCODING: [0x97,0x7b,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd7b97 <unknown>

add     za.d[w8, 5, vgx4], {z16.d - z19.d}, {z16.d - z19.d}  // 11000001-11110001-00011010-00010101
// CHECK-INST: add     za.d[w8, 5, vgx4], { z16.d - z19.d }, { z16.d - z19.d }
// CHECK-ENCODING: [0x15,0x1a,0xf1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f11a15 <unknown>

add     za.d[w8, 5], {z16.d - z19.d}, {z16.d - z19.d}  // 11000001-11110001-00011010-00010101
// CHECK-INST: add     za.d[w8, 5, vgx4], { z16.d - z19.d }, { z16.d - z19.d }
// CHECK-ENCODING: [0x15,0x1a,0xf1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f11a15 <unknown>

add     za.d[w8, 1, vgx4], {z0.d - z3.d}, {z28.d - z31.d}  // 11000001-11111101-00011000-00010001
// CHECK-INST: add     za.d[w8, 1, vgx4], { z0.d - z3.d }, { z28.d - z31.d }
// CHECK-ENCODING: [0x11,0x18,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd1811 <unknown>

add     za.d[w8, 1], {z0.d - z3.d}, {z28.d - z31.d}  // 11000001-11111101-00011000-00010001
// CHECK-INST: add     za.d[w8, 1, vgx4], { z0.d - z3.d }, { z28.d - z31.d }
// CHECK-ENCODING: [0x11,0x18,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd1811 <unknown>

add     za.d[w10, 0, vgx4], {z16.d - z19.d}, {z20.d - z23.d}  // 11000001-11110101-01011010-00010000
// CHECK-INST: add     za.d[w10, 0, vgx4], { z16.d - z19.d }, { z20.d - z23.d }
// CHECK-ENCODING: [0x10,0x5a,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f55a10 <unknown>

add     za.d[w10, 0], {z16.d - z19.d}, {z20.d - z23.d}  // 11000001-11110101-01011010-00010000
// CHECK-INST: add     za.d[w10, 0, vgx4], { z16.d - z19.d }, { z20.d - z23.d }
// CHECK-ENCODING: [0x10,0x5a,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f55a10 <unknown>

add     za.d[w8, 0, vgx4], {z12.d - z15.d}, {z0.d - z3.d}  // 11000001-11100001-00011001-10010000
// CHECK-INST: add     za.d[w8, 0, vgx4], { z12.d - z15.d }, { z0.d - z3.d }
// CHECK-ENCODING: [0x90,0x19,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11990 <unknown>

add     za.d[w8, 0], {z12.d - z15.d}, {z0.d - z3.d}  // 11000001-11100001-00011001-10010000
// CHECK-INST: add     za.d[w8, 0, vgx4], { z12.d - z15.d }, { z0.d - z3.d }
// CHECK-ENCODING: [0x90,0x19,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11990 <unknown>

add     za.d[w10, 1, vgx4], {z0.d - z3.d}, {z24.d - z27.d}  // 11000001-11111001-01011000-00010001
// CHECK-INST: add     za.d[w10, 1, vgx4], { z0.d - z3.d }, { z24.d - z27.d }
// CHECK-ENCODING: [0x11,0x58,0xf9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f95811 <unknown>

add     za.d[w10, 1], {z0.d - z3.d}, {z24.d - z27.d}  // 11000001-11111001-01011000-00010001
// CHECK-INST: add     za.d[w10, 1, vgx4], { z0.d - z3.d }, { z24.d - z27.d }
// CHECK-ENCODING: [0x11,0x58,0xf9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f95811 <unknown>

add     za.d[w8, 5, vgx4], {z20.d - z23.d}, {z28.d - z31.d}  // 11000001-11111101-00011010-10010101
// CHECK-INST: add     za.d[w8, 5, vgx4], { z20.d - z23.d }, { z28.d - z31.d }
// CHECK-ENCODING: [0x95,0x1a,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd1a95 <unknown>

add     za.d[w8, 5], {z20.d - z23.d}, {z28.d - z31.d}  // 11000001-11111101-00011010-10010101
// CHECK-INST: add     za.d[w8, 5, vgx4], { z20.d - z23.d }, { z28.d - z31.d }
// CHECK-ENCODING: [0x95,0x1a,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd1a95 <unknown>

add     za.d[w11, 2, vgx4], {z8.d - z11.d}, {z0.d - z3.d}  // 11000001-11100001-01111001-00010010
// CHECK-INST: add     za.d[w11, 2, vgx4], { z8.d - z11.d }, { z0.d - z3.d }
// CHECK-ENCODING: [0x12,0x79,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e17912 <unknown>

add     za.d[w11, 2], {z8.d - z11.d}, {z0.d - z3.d}  // 11000001-11100001-01111001-00010010
// CHECK-INST: add     za.d[w11, 2, vgx4], { z8.d - z11.d }, { z0.d - z3.d }
// CHECK-ENCODING: [0x12,0x79,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e17912 <unknown>

add     za.d[w9, 7, vgx4], {z12.d - z15.d}, {z8.d - z11.d}  // 11000001-11101001-00111001-10010111
// CHECK-INST: add     za.d[w9, 7, vgx4], { z12.d - z15.d }, { z8.d - z11.d }
// CHECK-ENCODING: [0x97,0x39,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e93997 <unknown>

add     za.d[w9, 7], {z12.d - z15.d}, {z8.d - z11.d}  // 11000001-11101001-00111001-10010111
// CHECK-INST: add     za.d[w9, 7, vgx4], { z12.d - z15.d }, { z8.d - z11.d }
// CHECK-ENCODING: [0x97,0x39,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e93997 <unknown>


add     {z0.b-z3.b}, {z0.b-z3.b}, z0.b  // 11000001-00100000-10101011-00000000
// CHECK-INST: add     { z0.b - z3.b }, { z0.b - z3.b }, z0.b
// CHECK-ENCODING: [0x00,0xab,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c120ab00 <unknown>

add     {z20.b-z23.b}, {z20.b-z23.b}, z5.b  // 11000001-00100101-10101011-00010100
// CHECK-INST: add     { z20.b - z23.b }, { z20.b - z23.b }, z5.b
// CHECK-ENCODING: [0x14,0xab,0x25,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c125ab14 <unknown>

add     {z20.b-z23.b}, {z20.b-z23.b}, z8.b  // 11000001-00101000-10101011-00010100
// CHECK-INST: add     { z20.b - z23.b }, { z20.b - z23.b }, z8.b
// CHECK-ENCODING: [0x14,0xab,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c128ab14 <unknown>

add     {z28.b-z31.b}, {z28.b-z31.b}, z15.b  // 11000001-00101111-10101011-00011100
// CHECK-INST: add     { z28.b - z31.b }, { z28.b - z31.b }, z15.b
// CHECK-ENCODING: [0x1c,0xab,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12fab1c <unknown>

