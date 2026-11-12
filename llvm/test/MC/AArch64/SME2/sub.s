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


sub     za.s[w8, 0, vgx2], {z0.s, z1.s}  // 11000001-10100000-00011100-00011000
// CHECK-INST: sub     za.s[w8, 0, vgx2], { z0.s, z1.s }
// CHECK-ENCODING: [0x18,0x1c,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a01c18 <unknown>

sub     za.s[w8, 0], {z0.s, z1.s}  // 11000001-10100000-00011100-00011000
// CHECK-INST: sub     za.s[w8, 0, vgx2], { z0.s, z1.s }
// CHECK-ENCODING: [0x18,0x1c,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a01c18 <unknown>

sub     za.s[w10, 5, vgx2], {z10.s, z11.s}  // 11000001-10100000-01011101-01011101
// CHECK-INST: sub     za.s[w10, 5, vgx2], { z10.s, z11.s }
// CHECK-ENCODING: [0x5d,0x5d,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a05d5d <unknown>

sub     za.s[w10, 5], {z10.s, z11.s}  // 11000001-10100000-01011101-01011101
// CHECK-INST: sub     za.s[w10, 5, vgx2], { z10.s, z11.s }
// CHECK-ENCODING: [0x5d,0x5d,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a05d5d <unknown>

sub     za.s[w11, 7, vgx2], {z12.s, z13.s}  // 11000001-10100000-01111101-10011111
// CHECK-INST: sub     za.s[w11, 7, vgx2], { z12.s, z13.s }
// CHECK-ENCODING: [0x9f,0x7d,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a07d9f <unknown>

sub     za.s[w11, 7], {z12.s, z13.s}  // 11000001-10100000-01111101-10011111
// CHECK-INST: sub     za.s[w11, 7, vgx2], { z12.s, z13.s }
// CHECK-ENCODING: [0x9f,0x7d,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a07d9f <unknown>

sub     za.s[w11, 7, vgx2], {z30.s, z31.s}  // 11000001-10100000-01111111-11011111
// CHECK-INST: sub     za.s[w11, 7, vgx2], { z30.s, z31.s }
// CHECK-ENCODING: [0xdf,0x7f,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a07fdf <unknown>

sub     za.s[w11, 7], {z30.s, z31.s}  // 11000001-10100000-01111111-11011111
// CHECK-INST: sub     za.s[w11, 7, vgx2], { z30.s, z31.s }
// CHECK-ENCODING: [0xdf,0x7f,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a07fdf <unknown>

sub     za.s[w8, 5, vgx2], {z16.s, z17.s}  // 11000001-10100000-00011110-00011101
// CHECK-INST: sub     za.s[w8, 5, vgx2], { z16.s, z17.s }
// CHECK-ENCODING: [0x1d,0x1e,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a01e1d <unknown>

sub     za.s[w8, 5], {z16.s, z17.s}  // 11000001-10100000-00011110-00011101
// CHECK-INST: sub     za.s[w8, 5, vgx2], { z16.s, z17.s }
// CHECK-ENCODING: [0x1d,0x1e,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a01e1d <unknown>

sub     za.s[w8, 1, vgx2], {z0.s, z1.s}  // 11000001-10100000-00011100-00011001
// CHECK-INST: sub     za.s[w8, 1, vgx2], { z0.s, z1.s }
// CHECK-ENCODING: [0x19,0x1c,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a01c19 <unknown>

sub     za.s[w8, 1], {z0.s, z1.s}  // 11000001-10100000-00011100-00011001
// CHECK-INST: sub     za.s[w8, 1, vgx2], { z0.s, z1.s }
// CHECK-ENCODING: [0x19,0x1c,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a01c19 <unknown>

sub     za.s[w10, 0, vgx2], {z18.s, z19.s}  // 11000001-10100000-01011110-01011000
// CHECK-INST: sub     za.s[w10, 0, vgx2], { z18.s, z19.s }
// CHECK-ENCODING: [0x58,0x5e,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a05e58 <unknown>

sub     za.s[w10, 0], {z18.s, z19.s}  // 11000001-10100000-01011110-01011000
// CHECK-INST: sub     za.s[w10, 0, vgx2], { z18.s, z19.s }
// CHECK-ENCODING: [0x58,0x5e,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a05e58 <unknown>

sub     za.s[w8, 0, vgx2], {z12.s, z13.s}  // 11000001-10100000-00011101-10011000
// CHECK-INST: sub     za.s[w8, 0, vgx2], { z12.s, z13.s }
// CHECK-ENCODING: [0x98,0x1d,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a01d98 <unknown>

sub     za.s[w8, 0], {z12.s, z13.s}  // 11000001-10100000-00011101-10011000
// CHECK-INST: sub     za.s[w8, 0, vgx2], { z12.s, z13.s }
// CHECK-ENCODING: [0x98,0x1d,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a01d98 <unknown>

sub     za.s[w10, 1, vgx2], {z0.s, z1.s}  // 11000001-10100000-01011100-00011001
// CHECK-INST: sub     za.s[w10, 1, vgx2], { z0.s, z1.s }
// CHECK-ENCODING: [0x19,0x5c,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a05c19 <unknown>

sub     za.s[w10, 1], {z0.s, z1.s}  // 11000001-10100000-01011100-00011001
// CHECK-INST: sub     za.s[w10, 1, vgx2], { z0.s, z1.s }
// CHECK-ENCODING: [0x19,0x5c,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a05c19 <unknown>

sub     za.s[w8, 5, vgx2], {z22.s, z23.s}  // 11000001-10100000-00011110-11011101
// CHECK-INST: sub     za.s[w8, 5, vgx2], { z22.s, z23.s }
// CHECK-ENCODING: [0xdd,0x1e,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a01edd <unknown>

sub     za.s[w8, 5], {z22.s, z23.s}  // 11000001-10100000-00011110-11011101
// CHECK-INST: sub     za.s[w8, 5, vgx2], { z22.s, z23.s }
// CHECK-ENCODING: [0xdd,0x1e,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a01edd <unknown>

sub     za.s[w11, 2, vgx2], {z8.s, z9.s}  // 11000001-10100000-01111101-00011010
// CHECK-INST: sub     za.s[w11, 2, vgx2], { z8.s, z9.s }
// CHECK-ENCODING: [0x1a,0x7d,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a07d1a <unknown>

sub     za.s[w11, 2], {z8.s, z9.s}  // 11000001-10100000-01111101-00011010
// CHECK-INST: sub     za.s[w11, 2, vgx2], { z8.s, z9.s }
// CHECK-ENCODING: [0x1a,0x7d,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a07d1a <unknown>

sub     za.s[w9, 7, vgx2], {z12.s, z13.s}  // 11000001-10100000-00111101-10011111
// CHECK-INST: sub     za.s[w9, 7, vgx2], { z12.s, z13.s }
// CHECK-ENCODING: [0x9f,0x3d,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a03d9f <unknown>

sub     za.s[w9, 7], {z12.s, z13.s}  // 11000001-10100000-00111101-10011111
// CHECK-INST: sub     za.s[w9, 7, vgx2], { z12.s, z13.s }
// CHECK-ENCODING: [0x9f,0x3d,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a03d9f <unknown>


sub     za.s[w8, 0, vgx2], {z0.s, z1.s}, z0.s  // 11000001-00100000-00011000-00011000
// CHECK-INST: sub     za.s[w8, 0, vgx2], { z0.s, z1.s }, z0.s
// CHECK-ENCODING: [0x18,0x18,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1201818 <unknown>

sub     za.s[w8, 0], {z0.s - z1.s}, z0.s  // 11000001-00100000-00011000-00011000
// CHECK-INST: sub     za.s[w8, 0, vgx2], { z0.s, z1.s }, z0.s
// CHECK-ENCODING: [0x18,0x18,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1201818 <unknown>

sub     za.s[w10, 5, vgx2], {z10.s, z11.s}, z5.s  // 11000001-00100101-01011001-01011101
// CHECK-INST: sub     za.s[w10, 5, vgx2], { z10.s, z11.s }, z5.s
// CHECK-ENCODING: [0x5d,0x59,0x25,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c125595d <unknown>

sub     za.s[w10, 5], {z10.s - z11.s}, z5.s  // 11000001-00100101-01011001-01011101
// CHECK-INST: sub     za.s[w10, 5, vgx2], { z10.s, z11.s }, z5.s
// CHECK-ENCODING: [0x5d,0x59,0x25,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c125595d <unknown>

sub     za.s[w11, 7, vgx2], {z13.s, z14.s}, z8.s  // 11000001-00101000-01111001-10111111
// CHECK-INST: sub     za.s[w11, 7, vgx2], { z13.s, z14.s }, z8.s
// CHECK-ENCODING: [0xbf,0x79,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12879bf <unknown>

sub     za.s[w11, 7], {z13.s - z14.s}, z8.s  // 11000001-00101000-01111001-10111111
// CHECK-INST: sub     za.s[w11, 7, vgx2], { z13.s, z14.s }, z8.s
// CHECK-ENCODING: [0xbf,0x79,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12879bf <unknown>

sub     za.s[w11, 7, vgx2], {z31.s, z0.s}, z15.s  // 11000001-00101111-01111011-11111111
// CHECK-INST: sub     za.s[w11, 7, vgx2], { z31.s, z0.s }, z15.s
// CHECK-ENCODING: [0xff,0x7b,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12f7bff <unknown>

sub     za.s[w11, 7], {z31.s - z0.s}, z15.s  // 11000001-00101111-01111011-11111111
// CHECK-INST: sub     za.s[w11, 7, vgx2], { z31.s, z0.s }, z15.s
// CHECK-ENCODING: [0xff,0x7b,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12f7bff <unknown>

sub     za.s[w8, 5, vgx2], {z17.s, z18.s}, z0.s  // 11000001-00100000-00011010-00111101
// CHECK-INST: sub     za.s[w8, 5, vgx2], { z17.s, z18.s }, z0.s
// CHECK-ENCODING: [0x3d,0x1a,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1201a3d <unknown>

sub     za.s[w8, 5], {z17.s - z18.s}, z0.s  // 11000001-00100000-00011010-00111101
// CHECK-INST: sub     za.s[w8, 5, vgx2], { z17.s, z18.s }, z0.s
// CHECK-ENCODING: [0x3d,0x1a,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1201a3d <unknown>

sub     za.s[w8, 1, vgx2], {z1.s, z2.s}, z14.s  // 11000001-00101110-00011000-00111001
// CHECK-INST: sub     za.s[w8, 1, vgx2], { z1.s, z2.s }, z14.s
// CHECK-ENCODING: [0x39,0x18,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e1839 <unknown>

sub     za.s[w8, 1], {z1.s - z2.s}, z14.s  // 11000001-00101110-00011000-00111001
// CHECK-INST: sub     za.s[w8, 1, vgx2], { z1.s, z2.s }, z14.s
// CHECK-ENCODING: [0x39,0x18,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e1839 <unknown>

sub     za.s[w10, 0, vgx2], {z19.s, z20.s}, z4.s  // 11000001-00100100-01011010-01111000
// CHECK-INST: sub     za.s[w10, 0, vgx2], { z19.s, z20.s }, z4.s
// CHECK-ENCODING: [0x78,0x5a,0x24,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1245a78 <unknown>

sub     za.s[w10, 0], {z19.s - z20.s}, z4.s  // 11000001-00100100-01011010-01111000
// CHECK-INST: sub     za.s[w10, 0, vgx2], { z19.s, z20.s }, z4.s
// CHECK-ENCODING: [0x78,0x5a,0x24,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1245a78 <unknown>

sub     za.s[w8, 0, vgx2], {z12.s, z13.s}, z2.s  // 11000001-00100010-00011001-10011000
// CHECK-INST: sub     za.s[w8, 0, vgx2], { z12.s, z13.s }, z2.s
// CHECK-ENCODING: [0x98,0x19,0x22,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1221998 <unknown>

sub     za.s[w8, 0], {z12.s - z13.s}, z2.s  // 11000001-00100010-00011001-10011000
// CHECK-INST: sub     za.s[w8, 0, vgx2], { z12.s, z13.s }, z2.s
// CHECK-ENCODING: [0x98,0x19,0x22,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1221998 <unknown>

sub     za.s[w10, 1, vgx2], {z1.s, z2.s}, z10.s  // 11000001-00101010-01011000-00111001
// CHECK-INST: sub     za.s[w10, 1, vgx2], { z1.s, z2.s }, z10.s
// CHECK-ENCODING: [0x39,0x58,0x2a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12a5839 <unknown>

sub     za.s[w10, 1], {z1.s - z2.s}, z10.s  // 11000001-00101010-01011000-00111001
// CHECK-INST: sub     za.s[w10, 1, vgx2], { z1.s, z2.s }, z10.s
// CHECK-ENCODING: [0x39,0x58,0x2a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12a5839 <unknown>

sub     za.s[w8, 5, vgx2], {z22.s, z23.s}, z14.s  // 11000001-00101110-00011010-11011101
// CHECK-INST: sub     za.s[w8, 5, vgx2], { z22.s, z23.s }, z14.s
// CHECK-ENCODING: [0xdd,0x1a,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e1add <unknown>

sub     za.s[w8, 5], {z22.s - z23.s}, z14.s  // 11000001-00101110-00011010-11011101
// CHECK-INST: sub     za.s[w8, 5, vgx2], { z22.s, z23.s }, z14.s
// CHECK-ENCODING: [0xdd,0x1a,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e1add <unknown>

sub     za.s[w11, 2, vgx2], {z9.s, z10.s}, z1.s  // 11000001-00100001-01111001-00111010
// CHECK-INST: sub     za.s[w11, 2, vgx2], { z9.s, z10.s }, z1.s
// CHECK-ENCODING: [0x3a,0x79,0x21,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c121793a <unknown>

sub     za.s[w11, 2], {z9.s - z10.s}, z1.s  // 11000001-00100001-01111001-00111010
// CHECK-INST: sub     za.s[w11, 2, vgx2], { z9.s, z10.s }, z1.s
// CHECK-ENCODING: [0x3a,0x79,0x21,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c121793a <unknown>

sub     za.s[w9, 7, vgx2], {z12.s, z13.s}, z11.s  // 11000001-00101011-00111001-10011111
// CHECK-INST: sub     za.s[w9, 7, vgx2], { z12.s, z13.s }, z11.s
// CHECK-ENCODING: [0x9f,0x39,0x2b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12b399f <unknown>

sub     za.s[w9, 7], {z12.s - z13.s}, z11.s  // 11000001-00101011-00111001-10011111
// CHECK-INST: sub     za.s[w9, 7, vgx2], { z12.s, z13.s }, z11.s
// CHECK-ENCODING: [0x9f,0x39,0x2b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12b399f <unknown>


sub     za.s[w8, 0, vgx2], {z0.s, z1.s}, {z0.s, z1.s}  // 11000001-10100000-00011000-00011000
// CHECK-INST: sub     za.s[w8, 0, vgx2], { z0.s, z1.s }, { z0.s, z1.s }
// CHECK-ENCODING: [0x18,0x18,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a01818 <unknown>

sub     za.s[w8, 0], {z0.s - z1.s}, {z0.s - z1.s}  // 11000001-10100000-00011000-00011000
// CHECK-INST: sub     za.s[w8, 0, vgx2], { z0.s, z1.s }, { z0.s, z1.s }
// CHECK-ENCODING: [0x18,0x18,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a01818 <unknown>

sub     za.s[w10, 5, vgx2], {z10.s, z11.s}, {z20.s, z21.s}  // 11000001-10110100-01011001-01011101
// CHECK-INST: sub     za.s[w10, 5, vgx2], { z10.s, z11.s }, { z20.s, z21.s }
// CHECK-ENCODING: [0x5d,0x59,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b4595d <unknown>

sub     za.s[w10, 5], {z10.s - z11.s}, {z20.s - z21.s}  // 11000001-10110100-01011001-01011101
// CHECK-INST: sub     za.s[w10, 5, vgx2], { z10.s, z11.s }, { z20.s, z21.s }
// CHECK-ENCODING: [0x5d,0x59,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b4595d <unknown>

sub     za.s[w11, 7, vgx2], {z12.s, z13.s}, {z8.s, z9.s}  // 11000001-10101000-01111001-10011111
// CHECK-INST: sub     za.s[w11, 7, vgx2], { z12.s, z13.s }, { z8.s, z9.s }
// CHECK-ENCODING: [0x9f,0x79,0xa8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a8799f <unknown>

sub     za.s[w11, 7], {z12.s - z13.s}, {z8.s - z9.s}  // 11000001-10101000-01111001-10011111
// CHECK-INST: sub     za.s[w11, 7, vgx2], { z12.s, z13.s }, { z8.s, z9.s }
// CHECK-ENCODING: [0x9f,0x79,0xa8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a8799f <unknown>

sub     za.s[w11, 7, vgx2], {z30.s, z31.s}, {z30.s, z31.s}  // 11000001-10111110-01111011-11011111
// CHECK-INST: sub     za.s[w11, 7, vgx2], { z30.s, z31.s }, { z30.s, z31.s }
// CHECK-ENCODING: [0xdf,0x7b,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be7bdf <unknown>

sub     za.s[w11, 7], {z30.s - z31.s}, {z30.s - z31.s}  // 11000001-10111110-01111011-11011111
// CHECK-INST: sub     za.s[w11, 7, vgx2], { z30.s, z31.s }, { z30.s, z31.s }
// CHECK-ENCODING: [0xdf,0x7b,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be7bdf <unknown>

sub     za.s[w8, 5, vgx2], {z16.s, z17.s}, {z16.s, z17.s}  // 11000001-10110000-00011010-00011101
// CHECK-INST: sub     za.s[w8, 5, vgx2], { z16.s, z17.s }, { z16.s, z17.s }
// CHECK-ENCODING: [0x1d,0x1a,0xb0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b01a1d <unknown>

sub     za.s[w8, 5], {z16.s - z17.s}, {z16.s - z17.s}  // 11000001-10110000-00011010-00011101
// CHECK-INST: sub     za.s[w8, 5, vgx2], { z16.s, z17.s }, { z16.s, z17.s }
// CHECK-ENCODING: [0x1d,0x1a,0xb0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b01a1d <unknown>

sub     za.s[w8, 1, vgx2], {z0.s, z1.s}, {z30.s, z31.s}  // 11000001-10111110-00011000-00011001
// CHECK-INST: sub     za.s[w8, 1, vgx2], { z0.s, z1.s }, { z30.s, z31.s }
// CHECK-ENCODING: [0x19,0x18,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be1819 <unknown>

sub     za.s[w8, 1], {z0.s - z1.s}, {z30.s - z31.s}  // 11000001-10111110-00011000-00011001
// CHECK-INST: sub     za.s[w8, 1, vgx2], { z0.s, z1.s }, { z30.s, z31.s }
// CHECK-ENCODING: [0x19,0x18,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be1819 <unknown>

sub     za.s[w10, 0, vgx2], {z18.s, z19.s}, {z20.s, z21.s}  // 11000001-10110100-01011010-01011000
// CHECK-INST: sub     za.s[w10, 0, vgx2], { z18.s, z19.s }, { z20.s, z21.s }
// CHECK-ENCODING: [0x58,0x5a,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b45a58 <unknown>

sub     za.s[w10, 0], {z18.s - z19.s}, {z20.s - z21.s}  // 11000001-10110100-01011010-01011000
// CHECK-INST: sub     za.s[w10, 0, vgx2], { z18.s, z19.s }, { z20.s, z21.s }
// CHECK-ENCODING: [0x58,0x5a,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b45a58 <unknown>

sub     za.s[w8, 0, vgx2], {z12.s, z13.s}, {z2.s, z3.s}  // 11000001-10100010-00011001-10011000
// CHECK-INST: sub     za.s[w8, 0, vgx2], { z12.s, z13.s }, { z2.s, z3.s }
// CHECK-ENCODING: [0x98,0x19,0xa2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a21998 <unknown>

sub     za.s[w8, 0], {z12.s - z13.s}, {z2.s - z3.s}  // 11000001-10100010-00011001-10011000
// CHECK-INST: sub     za.s[w8, 0, vgx2], { z12.s, z13.s }, { z2.s, z3.s }
// CHECK-ENCODING: [0x98,0x19,0xa2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a21998 <unknown>

sub     za.s[w10, 1, vgx2], {z0.s, z1.s}, {z26.s, z27.s}  // 11000001-10111010-01011000-00011001
// CHECK-INST: sub     za.s[w10, 1, vgx2], { z0.s, z1.s }, { z26.s, z27.s }
// CHECK-ENCODING: [0x19,0x58,0xba,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ba5819 <unknown>

sub     za.s[w10, 1], {z0.s - z1.s}, {z26.s - z27.s}  // 11000001-10111010-01011000-00011001
// CHECK-INST: sub     za.s[w10, 1, vgx2], { z0.s, z1.s }, { z26.s, z27.s }
// CHECK-ENCODING: [0x19,0x58,0xba,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ba5819 <unknown>

sub     za.s[w8, 5, vgx2], {z22.s, z23.s}, {z30.s, z31.s}  // 11000001-10111110-00011010-11011101
// CHECK-INST: sub     za.s[w8, 5, vgx2], { z22.s, z23.s }, { z30.s, z31.s }
// CHECK-ENCODING: [0xdd,0x1a,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be1add <unknown>

sub     za.s[w8, 5], {z22.s - z23.s}, {z30.s - z31.s}  // 11000001-10111110-00011010-11011101
// CHECK-INST: sub     za.s[w8, 5, vgx2], { z22.s, z23.s }, { z30.s, z31.s }
// CHECK-ENCODING: [0xdd,0x1a,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be1add <unknown>

sub     za.s[w11, 2, vgx2], {z8.s, z9.s}, {z0.s, z1.s}  // 11000001-10100000-01111001-00011010
// CHECK-INST: sub     za.s[w11, 2, vgx2], { z8.s, z9.s }, { z0.s, z1.s }
// CHECK-ENCODING: [0x1a,0x79,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a0791a <unknown>

sub     za.s[w11, 2], {z8.s - z9.s}, {z0.s - z1.s}  // 11000001-10100000-01111001-00011010
// CHECK-INST: sub     za.s[w11, 2, vgx2], { z8.s, z9.s }, { z0.s, z1.s }
// CHECK-ENCODING: [0x1a,0x79,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a0791a <unknown>

sub     za.s[w9, 7, vgx2], {z12.s, z13.s}, {z10.s, z11.s}  // 11000001-10101010-00111001-10011111
// CHECK-INST: sub     za.s[w9, 7, vgx2], { z12.s, z13.s }, { z10.s, z11.s }
// CHECK-ENCODING: [0x9f,0x39,0xaa,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1aa399f <unknown>

sub     za.s[w9, 7], {z12.s - z13.s}, {z10.s - z11.s}  // 11000001-10101010-00111001-10011111
// CHECK-INST: sub     za.s[w9, 7, vgx2], { z12.s, z13.s }, { z10.s, z11.s }
// CHECK-ENCODING: [0x9f,0x39,0xaa,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1aa399f <unknown>


sub     za.d[w8, 0, vgx2], {z0.d, z1.d}  // 11000001-11100000-00011100-00011000
// CHECK-INST: sub     za.d[w8, 0, vgx2], { z0.d, z1.d }
// CHECK-ENCODING: [0x18,0x1c,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e01c18 <unknown>

sub     za.d[w8, 0], {z0.d, z1.d}  // 11000001-11100000-00011100-00011000
// CHECK-INST: sub     za.d[w8, 0, vgx2], { z0.d, z1.d }
// CHECK-ENCODING: [0x18,0x1c,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e01c18 <unknown>

sub     za.d[w10, 5, vgx2], {z10.d, z11.d}  // 11000001-11100000-01011101-01011101
// CHECK-INST: sub     za.d[w10, 5, vgx2], { z10.d, z11.d }
// CHECK-ENCODING: [0x5d,0x5d,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e05d5d <unknown>

sub     za.d[w10, 5], {z10.d, z11.d}  // 11000001-11100000-01011101-01011101
// CHECK-INST: sub     za.d[w10, 5, vgx2], { z10.d, z11.d }
// CHECK-ENCODING: [0x5d,0x5d,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e05d5d <unknown>

sub     za.d[w11, 7, vgx2], {z12.d, z13.d}  // 11000001-11100000-01111101-10011111
// CHECK-INST: sub     za.d[w11, 7, vgx2], { z12.d, z13.d }
// CHECK-ENCODING: [0x9f,0x7d,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e07d9f <unknown>

sub     za.d[w11, 7], {z12.d, z13.d}  // 11000001-11100000-01111101-10011111
// CHECK-INST: sub     za.d[w11, 7, vgx2], { z12.d, z13.d }
// CHECK-ENCODING: [0x9f,0x7d,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e07d9f <unknown>

sub     za.d[w11, 7, vgx2], {z30.d, z31.d}  // 11000001-11100000-01111111-11011111
// CHECK-INST: sub     za.d[w11, 7, vgx2], { z30.d, z31.d }
// CHECK-ENCODING: [0xdf,0x7f,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e07fdf <unknown>

sub     za.d[w11, 7], {z30.d, z31.d}  // 11000001-11100000-01111111-11011111
// CHECK-INST: sub     za.d[w11, 7, vgx2], { z30.d, z31.d }
// CHECK-ENCODING: [0xdf,0x7f,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e07fdf <unknown>

sub     za.d[w8, 5, vgx2], {z16.d, z17.d}  // 11000001-11100000-00011110-00011101
// CHECK-INST: sub     za.d[w8, 5, vgx2], { z16.d, z17.d }
// CHECK-ENCODING: [0x1d,0x1e,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e01e1d <unknown>

sub     za.d[w8, 5], {z16.d, z17.d}  // 11000001-11100000-00011110-00011101
// CHECK-INST: sub     za.d[w8, 5, vgx2], { z16.d, z17.d }
// CHECK-ENCODING: [0x1d,0x1e,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e01e1d <unknown>

sub     za.d[w8, 1, vgx2], {z0.d, z1.d}  // 11000001-11100000-00011100-00011001
// CHECK-INST: sub     za.d[w8, 1, vgx2], { z0.d, z1.d }
// CHECK-ENCODING: [0x19,0x1c,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e01c19 <unknown>

sub     za.d[w8, 1], {z0.d, z1.d}  // 11000001-11100000-00011100-00011001
// CHECK-INST: sub     za.d[w8, 1, vgx2], { z0.d, z1.d }
// CHECK-ENCODING: [0x19,0x1c,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e01c19 <unknown>

sub     za.d[w10, 0, vgx2], {z18.d, z19.d}  // 11000001-11100000-01011110-01011000
// CHECK-INST: sub     za.d[w10, 0, vgx2], { z18.d, z19.d }
// CHECK-ENCODING: [0x58,0x5e,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e05e58 <unknown>

sub     za.d[w10, 0], {z18.d, z19.d}  // 11000001-11100000-01011110-01011000
// CHECK-INST: sub     za.d[w10, 0, vgx2], { z18.d, z19.d }
// CHECK-ENCODING: [0x58,0x5e,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e05e58 <unknown>

sub     za.d[w8, 0, vgx2], {z12.d, z13.d}  // 11000001-11100000-00011101-10011000
// CHECK-INST: sub     za.d[w8, 0, vgx2], { z12.d, z13.d }
// CHECK-ENCODING: [0x98,0x1d,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e01d98 <unknown>

sub     za.d[w8, 0], {z12.d, z13.d}  // 11000001-11100000-00011101-10011000
// CHECK-INST: sub     za.d[w8, 0, vgx2], { z12.d, z13.d }
// CHECK-ENCODING: [0x98,0x1d,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e01d98 <unknown>

sub     za.d[w10, 1, vgx2], {z0.d, z1.d}  // 11000001-11100000-01011100-00011001
// CHECK-INST: sub     za.d[w10, 1, vgx2], { z0.d, z1.d }
// CHECK-ENCODING: [0x19,0x5c,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e05c19 <unknown>

sub     za.d[w10, 1], {z0.d, z1.d}  // 11000001-11100000-01011100-00011001
// CHECK-INST: sub     za.d[w10, 1, vgx2], { z0.d, z1.d }
// CHECK-ENCODING: [0x19,0x5c,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e05c19 <unknown>

sub     za.d[w8, 5, vgx2], {z22.d, z23.d}  // 11000001-11100000-00011110-11011101
// CHECK-INST: sub     za.d[w8, 5, vgx2], { z22.d, z23.d }
// CHECK-ENCODING: [0xdd,0x1e,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e01edd <unknown>

sub     za.d[w8, 5], {z22.d, z23.d}  // 11000001-11100000-00011110-11011101
// CHECK-INST: sub     za.d[w8, 5, vgx2], { z22.d, z23.d }
// CHECK-ENCODING: [0xdd,0x1e,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e01edd <unknown>

sub     za.d[w11, 2, vgx2], {z8.d, z9.d}  // 11000001-11100000-01111101-00011010
// CHECK-INST: sub     za.d[w11, 2, vgx2], { z8.d, z9.d }
// CHECK-ENCODING: [0x1a,0x7d,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e07d1a <unknown>

sub     za.d[w11, 2], {z8.d, z9.d}  // 11000001-11100000-01111101-00011010
// CHECK-INST: sub     za.d[w11, 2, vgx2], { z8.d, z9.d }
// CHECK-ENCODING: [0x1a,0x7d,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e07d1a <unknown>

sub     za.d[w9, 7, vgx2], {z12.d, z13.d}  // 11000001-11100000-00111101-10011111
// CHECK-INST: sub     za.d[w9, 7, vgx2], { z12.d, z13.d }
// CHECK-ENCODING: [0x9f,0x3d,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e03d9f <unknown>

sub     za.d[w9, 7], {z12.d, z13.d}  // 11000001-11100000-00111101-10011111
// CHECK-INST: sub     za.d[w9, 7, vgx2], { z12.d, z13.d }
// CHECK-ENCODING: [0x9f,0x3d,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e03d9f <unknown>

sub     za.d[w8, 0, vgx2], {z0.d, z1.d}, z0.d  // 11000001-01100000-00011000-00011000
// CHECK-INST: sub     za.d[w8, 0, vgx2], { z0.d, z1.d }, z0.d
// CHECK-ENCODING: [0x18,0x18,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1601818 <unknown>

sub     za.d[w8, 0], {z0.d - z1.d}, z0.d  // 11000001-01100000-00011000-00011000
// CHECK-INST: sub     za.d[w8, 0, vgx2], { z0.d, z1.d }, z0.d
// CHECK-ENCODING: [0x18,0x18,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1601818 <unknown>

sub     za.d[w10, 5, vgx2], {z10.d, z11.d}, z5.d  // 11000001-01100101-01011001-01011101
// CHECK-INST: sub     za.d[w10, 5, vgx2], { z10.d, z11.d }, z5.d
// CHECK-ENCODING: [0x5d,0x59,0x65,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c165595d <unknown>

sub     za.d[w10, 5], {z10.d - z11.d}, z5.d  // 11000001-01100101-01011001-01011101
// CHECK-INST: sub     za.d[w10, 5, vgx2], { z10.d, z11.d }, z5.d
// CHECK-ENCODING: [0x5d,0x59,0x65,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c165595d <unknown>

sub     za.d[w11, 7, vgx2], {z13.d, z14.d}, z8.d  // 11000001-01101000-01111001-10111111
// CHECK-INST: sub     za.d[w11, 7, vgx2], { z13.d, z14.d }, z8.d
// CHECK-ENCODING: [0xbf,0x79,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16879bf <unknown>

sub     za.d[w11, 7], {z13.d - z14.d}, z8.d  // 11000001-01101000-01111001-10111111
// CHECK-INST: sub     za.d[w11, 7, vgx2], { z13.d, z14.d }, z8.d
// CHECK-ENCODING: [0xbf,0x79,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16879bf <unknown>

sub     za.d[w11, 7, vgx2], {z31.d, z0.d}, z15.d  // 11000001-01101111-01111011-11111111
// CHECK-INST: sub     za.d[w11, 7, vgx2], { z31.d, z0.d }, z15.d
// CHECK-ENCODING: [0xff,0x7b,0x6f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16f7bff <unknown>

sub     za.d[w11, 7], {z31.d - z0.d}, z15.d  // 11000001-01101111-01111011-11111111
// CHECK-INST: sub     za.d[w11, 7, vgx2], { z31.d, z0.d }, z15.d
// CHECK-ENCODING: [0xff,0x7b,0x6f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16f7bff <unknown>

sub     za.d[w8, 5, vgx2], {z17.d, z18.d}, z0.d  // 11000001-01100000-00011010-00111101
// CHECK-INST: sub     za.d[w8, 5, vgx2], { z17.d, z18.d }, z0.d
// CHECK-ENCODING: [0x3d,0x1a,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1601a3d <unknown>

sub     za.d[w8, 5], {z17.d - z18.d}, z0.d  // 11000001-01100000-00011010-00111101
// CHECK-INST: sub     za.d[w8, 5, vgx2], { z17.d, z18.d }, z0.d
// CHECK-ENCODING: [0x3d,0x1a,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1601a3d <unknown>

sub     za.d[w8, 1, vgx2], {z1.d, z2.d}, z14.d  // 11000001-01101110-00011000-00111001
// CHECK-INST: sub     za.d[w8, 1, vgx2], { z1.d, z2.d }, z14.d
// CHECK-ENCODING: [0x39,0x18,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e1839 <unknown>

sub     za.d[w8, 1], {z1.d - z2.d}, z14.d  // 11000001-01101110-00011000-00111001
// CHECK-INST: sub     za.d[w8, 1, vgx2], { z1.d, z2.d }, z14.d
// CHECK-ENCODING: [0x39,0x18,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e1839 <unknown>

sub     za.d[w10, 0, vgx2], {z19.d, z20.d}, z4.d  // 11000001-01100100-01011010-01111000
// CHECK-INST: sub     za.d[w10, 0, vgx2], { z19.d, z20.d }, z4.d
// CHECK-ENCODING: [0x78,0x5a,0x64,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1645a78 <unknown>

sub     za.d[w10, 0], {z19.d - z20.d}, z4.d  // 11000001-01100100-01011010-01111000
// CHECK-INST: sub     za.d[w10, 0, vgx2], { z19.d, z20.d }, z4.d
// CHECK-ENCODING: [0x78,0x5a,0x64,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1645a78 <unknown>

sub     za.d[w8, 0, vgx2], {z12.d, z13.d}, z2.d  // 11000001-01100010-00011001-10011000
// CHECK-INST: sub     za.d[w8, 0, vgx2], { z12.d, z13.d }, z2.d
// CHECK-ENCODING: [0x98,0x19,0x62,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1621998 <unknown>

sub     za.d[w8, 0], {z12.d - z13.d}, z2.d  // 11000001-01100010-00011001-10011000
// CHECK-INST: sub     za.d[w8, 0, vgx2], { z12.d, z13.d }, z2.d
// CHECK-ENCODING: [0x98,0x19,0x62,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1621998 <unknown>

sub     za.d[w10, 1, vgx2], {z1.d, z2.d}, z10.d  // 11000001-01101010-01011000-00111001
// CHECK-INST: sub     za.d[w10, 1, vgx2], { z1.d, z2.d }, z10.d
// CHECK-ENCODING: [0x39,0x58,0x6a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16a5839 <unknown>

sub     za.d[w10, 1], {z1.d - z2.d}, z10.d  // 11000001-01101010-01011000-00111001
// CHECK-INST: sub     za.d[w10, 1, vgx2], { z1.d, z2.d }, z10.d
// CHECK-ENCODING: [0x39,0x58,0x6a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16a5839 <unknown>

sub     za.d[w8, 5, vgx2], {z22.d, z23.d}, z14.d  // 11000001-01101110-00011010-11011101
// CHECK-INST: sub     za.d[w8, 5, vgx2], { z22.d, z23.d }, z14.d
// CHECK-ENCODING: [0xdd,0x1a,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e1add <unknown>

sub     za.d[w8, 5], {z22.d - z23.d}, z14.d  // 11000001-01101110-00011010-11011101
// CHECK-INST: sub     za.d[w8, 5, vgx2], { z22.d, z23.d }, z14.d
// CHECK-ENCODING: [0xdd,0x1a,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e1add <unknown>

sub     za.d[w11, 2, vgx2], {z9.d, z10.d}, z1.d  // 11000001-01100001-01111001-00111010
// CHECK-INST: sub     za.d[w11, 2, vgx2], { z9.d, z10.d }, z1.d
// CHECK-ENCODING: [0x3a,0x79,0x61,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c161793a <unknown>

sub     za.d[w11, 2], {z9.d - z10.d}, z1.d  // 11000001-01100001-01111001-00111010
// CHECK-INST: sub     za.d[w11, 2, vgx2], { z9.d, z10.d }, z1.d
// CHECK-ENCODING: [0x3a,0x79,0x61,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c161793a <unknown>

sub     za.d[w9, 7, vgx2], {z12.d, z13.d}, z11.d  // 11000001-01101011-00111001-10011111
// CHECK-INST: sub     za.d[w9, 7, vgx2], { z12.d, z13.d }, z11.d
// CHECK-ENCODING: [0x9f,0x39,0x6b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16b399f <unknown>

sub     za.d[w9, 7], {z12.d - z13.d}, z11.d  // 11000001-01101011-00111001-10011111
// CHECK-INST: sub     za.d[w9, 7, vgx2], { z12.d, z13.d }, z11.d
// CHECK-ENCODING: [0x9f,0x39,0x6b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16b399f <unknown>


sub     za.d[w8, 0, vgx2], {z0.d, z1.d}, {z0.d, z1.d}  // 11000001-11100000-00011000-00011000
// CHECK-INST: sub     za.d[w8, 0, vgx2], { z0.d, z1.d }, { z0.d, z1.d }
// CHECK-ENCODING: [0x18,0x18,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e01818 <unknown>

sub     za.d[w8, 0], {z0.d - z1.d}, {z0.d - z1.d}  // 11000001-11100000-00011000-00011000
// CHECK-INST: sub     za.d[w8, 0, vgx2], { z0.d, z1.d }, { z0.d, z1.d }
// CHECK-ENCODING: [0x18,0x18,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e01818 <unknown>

sub     za.d[w10, 5, vgx2], {z10.d, z11.d}, {z20.d, z21.d}  // 11000001-11110100-01011001-01011101
// CHECK-INST: sub     za.d[w10, 5, vgx2], { z10.d, z11.d }, { z20.d, z21.d }
// CHECK-ENCODING: [0x5d,0x59,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f4595d <unknown>

sub     za.d[w10, 5], {z10.d - z11.d}, {z20.d - z21.d}  // 11000001-11110100-01011001-01011101
// CHECK-INST: sub     za.d[w10, 5, vgx2], { z10.d, z11.d }, { z20.d, z21.d }
// CHECK-ENCODING: [0x5d,0x59,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f4595d <unknown>

sub     za.d[w11, 7, vgx2], {z12.d, z13.d}, {z8.d, z9.d}  // 11000001-11101000-01111001-10011111
// CHECK-INST: sub     za.d[w11, 7, vgx2], { z12.d, z13.d }, { z8.d, z9.d }
// CHECK-ENCODING: [0x9f,0x79,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e8799f <unknown>

sub     za.d[w11, 7], {z12.d - z13.d}, {z8.d - z9.d}  // 11000001-11101000-01111001-10011111
// CHECK-INST: sub     za.d[w11, 7, vgx2], { z12.d, z13.d }, { z8.d, z9.d }
// CHECK-ENCODING: [0x9f,0x79,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e8799f <unknown>

sub     za.d[w11, 7, vgx2], {z30.d, z31.d}, {z30.d, z31.d}  // 11000001-11111110-01111011-11011111
// CHECK-INST: sub     za.d[w11, 7, vgx2], { z30.d, z31.d }, { z30.d, z31.d }
// CHECK-ENCODING: [0xdf,0x7b,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe7bdf <unknown>

sub     za.d[w11, 7], {z30.d - z31.d}, {z30.d - z31.d}  // 11000001-11111110-01111011-11011111
// CHECK-INST: sub     za.d[w11, 7, vgx2], { z30.d, z31.d }, { z30.d, z31.d }
// CHECK-ENCODING: [0xdf,0x7b,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe7bdf <unknown>

sub     za.d[w8, 5, vgx2], {z16.d, z17.d}, {z16.d, z17.d}  // 11000001-11110000-00011010-00011101
// CHECK-INST: sub     za.d[w8, 5, vgx2], { z16.d, z17.d }, { z16.d, z17.d }
// CHECK-ENCODING: [0x1d,0x1a,0xf0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f01a1d <unknown>

sub     za.d[w8, 5], {z16.d - z17.d}, {z16.d - z17.d}  // 11000001-11110000-00011010-00011101
// CHECK-INST: sub     za.d[w8, 5, vgx2], { z16.d, z17.d }, { z16.d, z17.d }
// CHECK-ENCODING: [0x1d,0x1a,0xf0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f01a1d <unknown>

sub     za.d[w8, 1, vgx2], {z0.d, z1.d}, {z30.d, z31.d}  // 11000001-11111110-00011000-00011001
// CHECK-INST: sub     za.d[w8, 1, vgx2], { z0.d, z1.d }, { z30.d, z31.d }
// CHECK-ENCODING: [0x19,0x18,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe1819 <unknown>

sub     za.d[w8, 1], {z0.d - z1.d}, {z30.d - z31.d}  // 11000001-11111110-00011000-00011001
// CHECK-INST: sub     za.d[w8, 1, vgx2], { z0.d, z1.d }, { z30.d, z31.d }
// CHECK-ENCODING: [0x19,0x18,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe1819 <unknown>

sub     za.d[w10, 0, vgx2], {z18.d, z19.d}, {z20.d, z21.d}  // 11000001-11110100-01011010-01011000
// CHECK-INST: sub     za.d[w10, 0, vgx2], { z18.d, z19.d }, { z20.d, z21.d }
// CHECK-ENCODING: [0x58,0x5a,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f45a58 <unknown>

sub     za.d[w10, 0], {z18.d - z19.d}, {z20.d - z21.d}  // 11000001-11110100-01011010-01011000
// CHECK-INST: sub     za.d[w10, 0, vgx2], { z18.d, z19.d }, { z20.d, z21.d }
// CHECK-ENCODING: [0x58,0x5a,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f45a58 <unknown>

sub     za.d[w8, 0, vgx2], {z12.d, z13.d}, {z2.d, z3.d}  // 11000001-11100010-00011001-10011000
// CHECK-INST: sub     za.d[w8, 0, vgx2], { z12.d, z13.d }, { z2.d, z3.d }
// CHECK-ENCODING: [0x98,0x19,0xe2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e21998 <unknown>

sub     za.d[w8, 0], {z12.d - z13.d}, {z2.d - z3.d}  // 11000001-11100010-00011001-10011000
// CHECK-INST: sub     za.d[w8, 0, vgx2], { z12.d, z13.d }, { z2.d, z3.d }
// CHECK-ENCODING: [0x98,0x19,0xe2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e21998 <unknown>

sub     za.d[w10, 1, vgx2], {z0.d, z1.d}, {z26.d, z27.d}  // 11000001-11111010-01011000-00011001
// CHECK-INST: sub     za.d[w10, 1, vgx2], { z0.d, z1.d }, { z26.d, z27.d }
// CHECK-ENCODING: [0x19,0x58,0xfa,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fa5819 <unknown>

sub     za.d[w10, 1], {z0.d - z1.d}, {z26.d - z27.d}  // 11000001-11111010-01011000-00011001
// CHECK-INST: sub     za.d[w10, 1, vgx2], { z0.d, z1.d }, { z26.d, z27.d }
// CHECK-ENCODING: [0x19,0x58,0xfa,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fa5819 <unknown>

sub     za.d[w8, 5, vgx2], {z22.d, z23.d}, {z30.d, z31.d}  // 11000001-11111110-00011010-11011101
// CHECK-INST: sub     za.d[w8, 5, vgx2], { z22.d, z23.d }, { z30.d, z31.d }
// CHECK-ENCODING: [0xdd,0x1a,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe1add <unknown>

sub     za.d[w8, 5], {z22.d - z23.d}, {z30.d - z31.d}  // 11000001-11111110-00011010-11011101
// CHECK-INST: sub     za.d[w8, 5, vgx2], { z22.d, z23.d }, { z30.d, z31.d }
// CHECK-ENCODING: [0xdd,0x1a,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe1add <unknown>

sub     za.d[w11, 2, vgx2], {z8.d, z9.d}, {z0.d, z1.d}  // 11000001-11100000-01111001-00011010
// CHECK-INST: sub     za.d[w11, 2, vgx2], { z8.d, z9.d }, { z0.d, z1.d }
// CHECK-ENCODING: [0x1a,0x79,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e0791a <unknown>

sub     za.d[w11, 2], {z8.d - z9.d}, {z0.d - z1.d}  // 11000001-11100000-01111001-00011010
// CHECK-INST: sub     za.d[w11, 2, vgx2], { z8.d, z9.d }, { z0.d, z1.d }
// CHECK-ENCODING: [0x1a,0x79,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e0791a <unknown>

sub     za.d[w9, 7, vgx2], {z12.d, z13.d}, {z10.d, z11.d}  // 11000001-11101010-00111001-10011111
// CHECK-INST: sub     za.d[w9, 7, vgx2], { z12.d, z13.d }, { z10.d, z11.d }
// CHECK-ENCODING: [0x9f,0x39,0xea,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ea399f <unknown>

sub     za.d[w9, 7], {z12.d - z13.d}, {z10.d - z11.d}  // 11000001-11101010-00111001-10011111
// CHECK-INST: sub     za.d[w9, 7, vgx2], { z12.d, z13.d }, { z10.d, z11.d }
// CHECK-ENCODING: [0x9f,0x39,0xea,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ea399f <unknown>


sub     za.s[w8, 0, vgx4], {z0.s - z3.s}  // 11000001-10100001-00011100-00011000
// CHECK-INST: sub     za.s[w8, 0, vgx4], { z0.s - z3.s }
// CHECK-ENCODING: [0x18,0x1c,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11c18 <unknown>

sub     za.s[w8, 0], {z0.s - z3.s}  // 11000001-10100001-00011100-00011000
// CHECK-INST: sub     za.s[w8, 0, vgx4], { z0.s - z3.s }
// CHECK-ENCODING: [0x18,0x1c,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11c18 <unknown>

sub     za.s[w10, 5, vgx4], {z8.s - z11.s}  // 11000001-10100001-01011101-00011101
// CHECK-INST: sub     za.s[w10, 5, vgx4], { z8.s - z11.s }
// CHECK-ENCODING: [0x1d,0x5d,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a15d1d <unknown>

sub     za.s[w10, 5], {z8.s - z11.s}  // 11000001-10100001-01011101-00011101
// CHECK-INST: sub     za.s[w10, 5, vgx4], { z8.s - z11.s }
// CHECK-ENCODING: [0x1d,0x5d,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a15d1d <unknown>

sub     za.s[w11, 7, vgx4], {z12.s - z15.s}  // 11000001-10100001-01111101-10011111
// CHECK-INST: sub     za.s[w11, 7, vgx4], { z12.s - z15.s }
// CHECK-ENCODING: [0x9f,0x7d,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a17d9f <unknown>

sub     za.s[w11, 7], {z12.s - z15.s}  // 11000001-10100001-01111101-10011111
// CHECK-INST: sub     za.s[w11, 7, vgx4], { z12.s - z15.s }
// CHECK-ENCODING: [0x9f,0x7d,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a17d9f <unknown>

sub     za.s[w11, 7, vgx4], {z28.s - z31.s}  // 11000001-10100001-01111111-10011111
// CHECK-INST: sub     za.s[w11, 7, vgx4], { z28.s - z31.s }
// CHECK-ENCODING: [0x9f,0x7f,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a17f9f <unknown>

sub     za.s[w11, 7], {z28.s - z31.s}  // 11000001-10100001-01111111-10011111
// CHECK-INST: sub     za.s[w11, 7, vgx4], { z28.s - z31.s }
// CHECK-ENCODING: [0x9f,0x7f,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a17f9f <unknown>

sub     za.s[w8, 5, vgx4], {z16.s - z19.s}  // 11000001-10100001-00011110-00011101
// CHECK-INST: sub     za.s[w8, 5, vgx4], { z16.s - z19.s }
// CHECK-ENCODING: [0x1d,0x1e,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11e1d <unknown>

sub     za.s[w8, 5], {z16.s - z19.s}  // 11000001-10100001-00011110-00011101
// CHECK-INST: sub     za.s[w8, 5, vgx4], { z16.s - z19.s }
// CHECK-ENCODING: [0x1d,0x1e,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11e1d <unknown>

sub     za.s[w8, 1, vgx4], {z0.s - z3.s}  // 11000001-10100001-00011100-00011001
// CHECK-INST: sub     za.s[w8, 1, vgx4], { z0.s - z3.s }
// CHECK-ENCODING: [0x19,0x1c,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11c19 <unknown>

sub     za.s[w8, 1], {z0.s - z3.s}  // 11000001-10100001-00011100-00011001
// CHECK-INST: sub     za.s[w8, 1, vgx4], { z0.s - z3.s }
// CHECK-ENCODING: [0x19,0x1c,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11c19 <unknown>

sub     za.s[w10, 0, vgx4], {z16.s - z19.s}  // 11000001-10100001-01011110-00011000
// CHECK-INST: sub     za.s[w10, 0, vgx4], { z16.s - z19.s }
// CHECK-ENCODING: [0x18,0x5e,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a15e18 <unknown>

sub     za.s[w10, 0], {z16.s - z19.s}  // 11000001-10100001-01011110-00011000
// CHECK-INST: sub     za.s[w10, 0, vgx4], { z16.s - z19.s }
// CHECK-ENCODING: [0x18,0x5e,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a15e18 <unknown>

sub     za.s[w8, 0, vgx4], {z12.s - z15.s}  // 11000001-10100001-00011101-10011000
// CHECK-INST: sub     za.s[w8, 0, vgx4], { z12.s - z15.s }
// CHECK-ENCODING: [0x98,0x1d,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11d98 <unknown>

sub     za.s[w8, 0], {z12.s - z15.s}  // 11000001-10100001-00011101-10011000
// CHECK-INST: sub     za.s[w8, 0, vgx4], { z12.s - z15.s }
// CHECK-ENCODING: [0x98,0x1d,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11d98 <unknown>

sub     za.s[w10, 1, vgx4], {z0.s - z3.s}  // 11000001-10100001-01011100-00011001
// CHECK-INST: sub     za.s[w10, 1, vgx4], { z0.s - z3.s }
// CHECK-ENCODING: [0x19,0x5c,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a15c19 <unknown>

sub     za.s[w10, 1], {z0.s - z3.s}  // 11000001-10100001-01011100-00011001
// CHECK-INST: sub     za.s[w10, 1, vgx4], { z0.s - z3.s }
// CHECK-ENCODING: [0x19,0x5c,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a15c19 <unknown>

sub     za.s[w8, 5, vgx4], {z20.s - z23.s}  // 11000001-10100001-00011110-10011101
// CHECK-INST: sub     za.s[w8, 5, vgx4], { z20.s - z23.s }
// CHECK-ENCODING: [0x9d,0x1e,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11e9d <unknown>

sub     za.s[w8, 5], {z20.s - z23.s}  // 11000001-10100001-00011110-10011101
// CHECK-INST: sub     za.s[w8, 5, vgx4], { z20.s - z23.s }
// CHECK-ENCODING: [0x9d,0x1e,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11e9d <unknown>

sub     za.s[w11, 2, vgx4], {z8.s - z11.s}  // 11000001-10100001-01111101-00011010
// CHECK-INST: sub     za.s[w11, 2, vgx4], { z8.s - z11.s }
// CHECK-ENCODING: [0x1a,0x7d,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a17d1a <unknown>

sub     za.s[w11, 2], {z8.s - z11.s}  // 11000001-10100001-01111101-00011010
// CHECK-INST: sub     za.s[w11, 2, vgx4], { z8.s - z11.s }
// CHECK-ENCODING: [0x1a,0x7d,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a17d1a <unknown>

sub     za.s[w9, 7, vgx4], {z12.s - z15.s}  // 11000001-10100001-00111101-10011111
// CHECK-INST: sub     za.s[w9, 7, vgx4], { z12.s - z15.s }
// CHECK-ENCODING: [0x9f,0x3d,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a13d9f <unknown>

sub     za.s[w9, 7], {z12.s - z15.s}  // 11000001-10100001-00111101-10011111
// CHECK-INST: sub     za.s[w9, 7, vgx4], { z12.s - z15.s }
// CHECK-ENCODING: [0x9f,0x3d,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a13d9f <unknown>


sub     za.s[w8, 0, vgx4], {z0.s - z3.s}, z0.s  // 11000001-00110000-00011000-00011000
// CHECK-INST: sub     za.s[w8, 0, vgx4], { z0.s - z3.s }, z0.s
// CHECK-ENCODING: [0x18,0x18,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1301818 <unknown>

sub     za.s[w8, 0], {z0.s - z3.s}, z0.s  // 11000001-00110000-00011000-00011000
// CHECK-INST: sub     za.s[w8, 0, vgx4], { z0.s - z3.s }, z0.s
// CHECK-ENCODING: [0x18,0x18,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1301818 <unknown>

sub     za.s[w10, 5, vgx4], {z10.s - z13.s}, z5.s  // 11000001-00110101-01011001-01011101
// CHECK-INST: sub     za.s[w10, 5, vgx4], { z10.s - z13.s }, z5.s
// CHECK-ENCODING: [0x5d,0x59,0x35,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c135595d <unknown>

sub     za.s[w10, 5], {z10.s - z13.s}, z5.s  // 11000001-00110101-01011001-01011101
// CHECK-INST: sub     za.s[w10, 5, vgx4], { z10.s - z13.s }, z5.s
// CHECK-ENCODING: [0x5d,0x59,0x35,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c135595d <unknown>

sub     za.s[w11, 7, vgx4], {z13.s - z16.s}, z8.s  // 11000001-00111000-01111001-10111111
// CHECK-INST: sub     za.s[w11, 7, vgx4], { z13.s - z16.s }, z8.s
// CHECK-ENCODING: [0xbf,0x79,0x38,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13879bf <unknown>

sub     za.s[w11, 7], {z13.s - z16.s}, z8.s  // 11000001-00111000-01111001-10111111
// CHECK-INST: sub     za.s[w11, 7, vgx4], { z13.s - z16.s }, z8.s
// CHECK-ENCODING: [0xbf,0x79,0x38,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13879bf <unknown>

sub     za.s[w11, 7, vgx4], {z31.s - z2.s}, z15.s  // 11000001-00111111-01111011-11111111
// CHECK-INST: sub     za.s[w11, 7, vgx4], { z31.s, z0.s, z1.s, z2.s }, z15.s
// CHECK-ENCODING: [0xff,0x7b,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13f7bff <unknown>

sub     za.s[w11, 7], {z31.s - z2.s}, z15.s  // 11000001-00111111-01111011-11111111
// CHECK-INST: sub     za.s[w11, 7, vgx4], { z31.s, z0.s, z1.s, z2.s }, z15.s
// CHECK-ENCODING: [0xff,0x7b,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13f7bff <unknown>

sub     za.s[w8, 5, vgx4], {z17.s - z20.s}, z0.s  // 11000001-00110000-00011010-00111101
// CHECK-INST: sub     za.s[w8, 5, vgx4], { z17.s - z20.s }, z0.s
// CHECK-ENCODING: [0x3d,0x1a,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1301a3d <unknown>

sub     za.s[w8, 5], {z17.s - z20.s}, z0.s  // 11000001-00110000-00011010-00111101
// CHECK-INST: sub     za.s[w8, 5, vgx4], { z17.s - z20.s }, z0.s
// CHECK-ENCODING: [0x3d,0x1a,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1301a3d <unknown>

sub     za.s[w8, 1, vgx4], {z1.s - z4.s}, z14.s  // 11000001-00111110-00011000-00111001
// CHECK-INST: sub     za.s[w8, 1, vgx4], { z1.s - z4.s }, z14.s
// CHECK-ENCODING: [0x39,0x18,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e1839 <unknown>

sub     za.s[w8, 1], {z1.s - z4.s}, z14.s  // 11000001-00111110-00011000-00111001
// CHECK-INST: sub     za.s[w8, 1, vgx4], { z1.s - z4.s }, z14.s
// CHECK-ENCODING: [0x39,0x18,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e1839 <unknown>

sub     za.s[w10, 0, vgx4], {z19.s - z22.s}, z4.s  // 11000001-00110100-01011010-01111000
// CHECK-INST: sub     za.s[w10, 0, vgx4], { z19.s - z22.s }, z4.s
// CHECK-ENCODING: [0x78,0x5a,0x34,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1345a78 <unknown>

sub     za.s[w10, 0], {z19.s - z22.s}, z4.s  // 11000001-00110100-01011010-01111000
// CHECK-INST: sub     za.s[w10, 0, vgx4], { z19.s - z22.s }, z4.s
// CHECK-ENCODING: [0x78,0x5a,0x34,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1345a78 <unknown>

sub     za.s[w8, 0, vgx4], {z12.s - z15.s}, z2.s  // 11000001-00110010-00011001-10011000
// CHECK-INST: sub     za.s[w8, 0, vgx4], { z12.s - z15.s }, z2.s
// CHECK-ENCODING: [0x98,0x19,0x32,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1321998 <unknown>

sub     za.s[w8, 0], {z12.s - z15.s}, z2.s  // 11000001-00110010-00011001-10011000
// CHECK-INST: sub     za.s[w8, 0, vgx4], { z12.s - z15.s }, z2.s
// CHECK-ENCODING: [0x98,0x19,0x32,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1321998 <unknown>

sub     za.s[w10, 1, vgx4], {z1.s - z4.s}, z10.s  // 11000001-00111010-01011000-00111001
// CHECK-INST: sub     za.s[w10, 1, vgx4], { z1.s - z4.s }, z10.s
// CHECK-ENCODING: [0x39,0x58,0x3a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13a5839 <unknown>

sub     za.s[w10, 1], {z1.s - z4.s}, z10.s  // 11000001-00111010-01011000-00111001
// CHECK-INST: sub     za.s[w10, 1, vgx4], { z1.s - z4.s }, z10.s
// CHECK-ENCODING: [0x39,0x58,0x3a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13a5839 <unknown>

sub     za.s[w8, 5, vgx4], {z22.s - z25.s}, z14.s  // 11000001-00111110-00011010-11011101
// CHECK-INST: sub     za.s[w8, 5, vgx4], { z22.s - z25.s }, z14.s
// CHECK-ENCODING: [0xdd,0x1a,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e1add <unknown>

sub     za.s[w8, 5], {z22.s - z25.s}, z14.s  // 11000001-00111110-00011010-11011101
// CHECK-INST: sub     za.s[w8, 5, vgx4], { z22.s - z25.s }, z14.s
// CHECK-ENCODING: [0xdd,0x1a,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e1add <unknown>

sub     za.s[w11, 2, vgx4], {z9.s - z12.s}, z1.s  // 11000001-00110001-01111001-00111010
// CHECK-INST: sub     za.s[w11, 2, vgx4], { z9.s - z12.s }, z1.s
// CHECK-ENCODING: [0x3a,0x79,0x31,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c131793a <unknown>

sub     za.s[w11, 2], {z9.s - z12.s}, z1.s  // 11000001-00110001-01111001-00111010
// CHECK-INST: sub     za.s[w11, 2, vgx4], { z9.s - z12.s }, z1.s
// CHECK-ENCODING: [0x3a,0x79,0x31,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c131793a <unknown>

sub     za.s[w9, 7, vgx4], {z12.s - z15.s}, z11.s  // 11000001-00111011-00111001-10011111
// CHECK-INST: sub     za.s[w9, 7, vgx4], { z12.s - z15.s }, z11.s
// CHECK-ENCODING: [0x9f,0x39,0x3b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13b399f <unknown>

sub     za.s[w9, 7], {z12.s - z15.s}, z11.s  // 11000001-00111011-00111001-10011111
// CHECK-INST: sub     za.s[w9, 7, vgx4], { z12.s - z15.s }, z11.s
// CHECK-ENCODING: [0x9f,0x39,0x3b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13b399f <unknown>


sub     za.s[w8, 0, vgx4], {z0.s - z3.s}, {z0.s - z3.s}  // 11000001-10100001-00011000-00011000
// CHECK-INST: sub     za.s[w8, 0, vgx4], { z0.s - z3.s }, { z0.s - z3.s }
// CHECK-ENCODING: [0x18,0x18,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11818 <unknown>

sub     za.s[w8, 0], {z0.s - z3.s}, {z0.s - z3.s}  // 11000001-10100001-00011000-00011000
// CHECK-INST: sub     za.s[w8, 0, vgx4], { z0.s - z3.s }, { z0.s - z3.s }
// CHECK-ENCODING: [0x18,0x18,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11818 <unknown>

sub     za.s[w10, 5, vgx4], {z8.s - z11.s}, {z20.s - z23.s}  // 11000001-10110101-01011001-00011101
// CHECK-INST: sub     za.s[w10, 5, vgx4], { z8.s - z11.s }, { z20.s - z23.s }
// CHECK-ENCODING: [0x1d,0x59,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b5591d <unknown>

sub     za.s[w10, 5], {z8.s - z11.s}, {z20.s - z23.s}  // 11000001-10110101-01011001-00011101
// CHECK-INST: sub     za.s[w10, 5, vgx4], { z8.s - z11.s }, { z20.s - z23.s }
// CHECK-ENCODING: [0x1d,0x59,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b5591d <unknown>

sub     za.s[w11, 7, vgx4], {z12.s - z15.s}, {z8.s - z11.s}  // 11000001-10101001-01111001-10011111
// CHECK-INST: sub     za.s[w11, 7, vgx4], { z12.s - z15.s }, { z8.s - z11.s }
// CHECK-ENCODING: [0x9f,0x79,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a9799f <unknown>

sub     za.s[w11, 7], {z12.s - z15.s}, {z8.s - z11.s}  // 11000001-10101001-01111001-10011111
// CHECK-INST: sub     za.s[w11, 7, vgx4], { z12.s - z15.s }, { z8.s - z11.s }
// CHECK-ENCODING: [0x9f,0x79,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a9799f <unknown>

sub     za.s[w11, 7, vgx4], {z28.s - z31.s}, {z28.s - z31.s}  // 11000001-10111101-01111011-10011111
// CHECK-INST: sub     za.s[w11, 7, vgx4], { z28.s - z31.s }, { z28.s - z31.s }
// CHECK-ENCODING: [0x9f,0x7b,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd7b9f <unknown>

sub     za.s[w11, 7], {z28.s - z31.s}, {z28.s - z31.s}  // 11000001-10111101-01111011-10011111
// CHECK-INST: sub     za.s[w11, 7, vgx4], { z28.s - z31.s }, { z28.s - z31.s }
// CHECK-ENCODING: [0x9f,0x7b,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd7b9f <unknown>

sub     za.s[w8, 5, vgx4], {z16.s - z19.s}, {z16.s - z19.s}  // 11000001-10110001-00011010-00011101
// CHECK-INST: sub     za.s[w8, 5, vgx4], { z16.s - z19.s }, { z16.s - z19.s }
// CHECK-ENCODING: [0x1d,0x1a,0xb1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b11a1d <unknown>

sub     za.s[w8, 5], {z16.s - z19.s}, {z16.s - z19.s}  // 11000001-10110001-00011010-00011101
// CHECK-INST: sub     za.s[w8, 5, vgx4], { z16.s - z19.s }, { z16.s - z19.s }
// CHECK-ENCODING: [0x1d,0x1a,0xb1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b11a1d <unknown>

sub     za.s[w8, 1, vgx4], {z0.s - z3.s}, {z28.s - z31.s}  // 11000001-10111101-00011000-00011001
// CHECK-INST: sub     za.s[w8, 1, vgx4], { z0.s - z3.s }, { z28.s - z31.s }
// CHECK-ENCODING: [0x19,0x18,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd1819 <unknown>

sub     za.s[w8, 1], {z0.s - z3.s}, {z28.s - z31.s}  // 11000001-10111101-00011000-00011001
// CHECK-INST: sub     za.s[w8, 1, vgx4], { z0.s - z3.s }, { z28.s - z31.s }
// CHECK-ENCODING: [0x19,0x18,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd1819 <unknown>

sub     za.s[w10, 0, vgx4], {z16.s - z19.s}, {z20.s - z23.s}  // 11000001-10110101-01011010-00011000
// CHECK-INST: sub     za.s[w10, 0, vgx4], { z16.s - z19.s }, { z20.s - z23.s }
// CHECK-ENCODING: [0x18,0x5a,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b55a18 <unknown>

sub     za.s[w10, 0], {z16.s - z19.s}, {z20.s - z23.s}  // 11000001-10110101-01011010-00011000
// CHECK-INST: sub     za.s[w10, 0, vgx4], { z16.s - z19.s }, { z20.s - z23.s }
// CHECK-ENCODING: [0x18,0x5a,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b55a18 <unknown>

sub     za.s[w8, 0, vgx4], {z12.s - z15.s}, {z0.s - z3.s}  // 11000001-10100001-00011001-10011000
// CHECK-INST: sub     za.s[w8, 0, vgx4], { z12.s - z15.s }, { z0.s - z3.s }
// CHECK-ENCODING: [0x98,0x19,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11998 <unknown>

sub     za.s[w8, 0], {z12.s - z15.s}, {z0.s - z3.s}  // 11000001-10100001-00011001-10011000
// CHECK-INST: sub     za.s[w8, 0, vgx4], { z12.s - z15.s }, { z0.s - z3.s }
// CHECK-ENCODING: [0x98,0x19,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11998 <unknown>

sub     za.s[w10, 1, vgx4], {z0.s - z3.s}, {z24.s - z27.s}  // 11000001-10111001-01011000-00011001
// CHECK-INST: sub     za.s[w10, 1, vgx4], { z0.s - z3.s }, { z24.s - z27.s }
// CHECK-ENCODING: [0x19,0x58,0xb9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b95819 <unknown>

sub     za.s[w10, 1], {z0.s - z3.s}, {z24.s - z27.s}  // 11000001-10111001-01011000-00011001
// CHECK-INST: sub     za.s[w10, 1, vgx4], { z0.s - z3.s }, { z24.s - z27.s }
// CHECK-ENCODING: [0x19,0x58,0xb9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b95819 <unknown>

sub     za.s[w8, 5, vgx4], {z20.s - z23.s}, {z28.s - z31.s}  // 11000001-10111101-00011010-10011101
// CHECK-INST: sub     za.s[w8, 5, vgx4], { z20.s - z23.s }, { z28.s - z31.s }
// CHECK-ENCODING: [0x9d,0x1a,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd1a9d <unknown>

sub     za.s[w8, 5], {z20.s - z23.s}, {z28.s - z31.s}  // 11000001-10111101-00011010-10011101
// CHECK-INST: sub     za.s[w8, 5, vgx4], { z20.s - z23.s }, { z28.s - z31.s }
// CHECK-ENCODING: [0x9d,0x1a,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd1a9d <unknown>

sub     za.s[w11, 2, vgx4], {z8.s - z11.s}, {z0.s - z3.s}  // 11000001-10100001-01111001-00011010
// CHECK-INST: sub     za.s[w11, 2, vgx4], { z8.s - z11.s }, { z0.s - z3.s }
// CHECK-ENCODING: [0x1a,0x79,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a1791a <unknown>

sub     za.s[w11, 2], {z8.s - z11.s}, {z0.s - z3.s}  // 11000001-10100001-01111001-00011010
// CHECK-INST: sub     za.s[w11, 2, vgx4], { z8.s - z11.s }, { z0.s - z3.s }
// CHECK-ENCODING: [0x1a,0x79,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a1791a <unknown>

sub     za.s[w9, 7, vgx4], {z12.s - z15.s}, {z8.s - z11.s}  // 11000001-10101001-00111001-10011111
// CHECK-INST: sub     za.s[w9, 7, vgx4], { z12.s - z15.s }, { z8.s - z11.s }
// CHECK-ENCODING: [0x9f,0x39,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a9399f <unknown>

sub     za.s[w9, 7], {z12.s - z15.s}, {z8.s - z11.s}  // 11000001-10101001-00111001-10011111
// CHECK-INST: sub     za.s[w9, 7, vgx4], { z12.s - z15.s }, { z8.s - z11.s }
// CHECK-ENCODING: [0x9f,0x39,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a9399f <unknown>


sub     za.d[w8, 0, vgx4], {z0.d - z3.d}  // 11000001-11100001-00011100-00011000
// CHECK-INST: sub     za.d[w8, 0, vgx4], { z0.d - z3.d }
// CHECK-ENCODING: [0x18,0x1c,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11c18 <unknown>

sub     za.d[w8, 0], {z0.d - z3.d}  // 11000001-11100001-00011100-00011000
// CHECK-INST: sub     za.d[w8, 0, vgx4], { z0.d - z3.d }
// CHECK-ENCODING: [0x18,0x1c,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11c18 <unknown>

sub     za.d[w10, 5, vgx4], {z8.d - z11.d}  // 11000001-11100001-01011101-00011101
// CHECK-INST: sub     za.d[w10, 5, vgx4], { z8.d - z11.d }
// CHECK-ENCODING: [0x1d,0x5d,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e15d1d <unknown>

sub     za.d[w10, 5], {z8.d - z11.d}  // 11000001-11100001-01011101-00011101
// CHECK-INST: sub     za.d[w10, 5, vgx4], { z8.d - z11.d }
// CHECK-ENCODING: [0x1d,0x5d,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e15d1d <unknown>

sub     za.d[w11, 7, vgx4], {z12.d - z15.d}  // 11000001-11100001-01111101-10011111
// CHECK-INST: sub     za.d[w11, 7, vgx4], { z12.d - z15.d }
// CHECK-ENCODING: [0x9f,0x7d,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e17d9f <unknown>

sub     za.d[w11, 7], {z12.d - z15.d}  // 11000001-11100001-01111101-10011111
// CHECK-INST: sub     za.d[w11, 7, vgx4], { z12.d - z15.d }
// CHECK-ENCODING: [0x9f,0x7d,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e17d9f <unknown>

sub     za.d[w11, 7, vgx4], {z28.d - z31.d}  // 11000001-11100001-01111111-10011111
// CHECK-INST: sub     za.d[w11, 7, vgx4], { z28.d - z31.d }
// CHECK-ENCODING: [0x9f,0x7f,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e17f9f <unknown>

sub     za.d[w11, 7], {z28.d - z31.d}  // 11000001-11100001-01111111-10011111
// CHECK-INST: sub     za.d[w11, 7, vgx4], { z28.d - z31.d }
// CHECK-ENCODING: [0x9f,0x7f,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e17f9f <unknown>

sub     za.d[w8, 5, vgx4], {z16.d - z19.d}  // 11000001-11100001-00011110-00011101
// CHECK-INST: sub     za.d[w8, 5, vgx4], { z16.d - z19.d }
// CHECK-ENCODING: [0x1d,0x1e,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11e1d <unknown>

sub     za.d[w8, 5], {z16.d - z19.d}  // 11000001-11100001-00011110-00011101
// CHECK-INST: sub     za.d[w8, 5, vgx4], { z16.d - z19.d }
// CHECK-ENCODING: [0x1d,0x1e,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11e1d <unknown>

sub     za.d[w8, 1, vgx4], {z0.d - z3.d}  // 11000001-11100001-00011100-00011001
// CHECK-INST: sub     za.d[w8, 1, vgx4], { z0.d - z3.d }
// CHECK-ENCODING: [0x19,0x1c,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11c19 <unknown>

sub     za.d[w8, 1], {z0.d - z3.d}  // 11000001-11100001-00011100-00011001
// CHECK-INST: sub     za.d[w8, 1, vgx4], { z0.d - z3.d }
// CHECK-ENCODING: [0x19,0x1c,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11c19 <unknown>

sub     za.d[w10, 0, vgx4], {z16.d - z19.d}  // 11000001-11100001-01011110-00011000
// CHECK-INST: sub     za.d[w10, 0, vgx4], { z16.d - z19.d }
// CHECK-ENCODING: [0x18,0x5e,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e15e18 <unknown>

sub     za.d[w10, 0], {z16.d - z19.d}  // 11000001-11100001-01011110-00011000
// CHECK-INST: sub     za.d[w10, 0, vgx4], { z16.d - z19.d }
// CHECK-ENCODING: [0x18,0x5e,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e15e18 <unknown>

sub     za.d[w8, 0, vgx4], {z12.d - z15.d}  // 11000001-11100001-00011101-10011000
// CHECK-INST: sub     za.d[w8, 0, vgx4], { z12.d - z15.d }
// CHECK-ENCODING: [0x98,0x1d,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11d98 <unknown>

sub     za.d[w8, 0], {z12.d - z15.d}  // 11000001-11100001-00011101-10011000
// CHECK-INST: sub     za.d[w8, 0, vgx4], { z12.d - z15.d }
// CHECK-ENCODING: [0x98,0x1d,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11d98 <unknown>

sub     za.d[w10, 1, vgx4], {z0.d - z3.d}  // 11000001-11100001-01011100-00011001
// CHECK-INST: sub     za.d[w10, 1, vgx4], { z0.d - z3.d }
// CHECK-ENCODING: [0x19,0x5c,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e15c19 <unknown>

sub     za.d[w10, 1], {z0.d - z3.d}  // 11000001-11100001-01011100-00011001
// CHECK-INST: sub     za.d[w10, 1, vgx4], { z0.d - z3.d }
// CHECK-ENCODING: [0x19,0x5c,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e15c19 <unknown>

sub     za.d[w8, 5, vgx4], {z20.d - z23.d}  // 11000001-11100001-00011110-10011101
// CHECK-INST: sub     za.d[w8, 5, vgx4], { z20.d - z23.d }
// CHECK-ENCODING: [0x9d,0x1e,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11e9d <unknown>

sub     za.d[w8, 5], {z20.d - z23.d}  // 11000001-11100001-00011110-10011101
// CHECK-INST: sub     za.d[w8, 5, vgx4], { z20.d - z23.d }
// CHECK-ENCODING: [0x9d,0x1e,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11e9d <unknown>

sub     za.d[w11, 2, vgx4], {z8.d - z11.d}  // 11000001-11100001-01111101-00011010
// CHECK-INST: sub     za.d[w11, 2, vgx4], { z8.d - z11.d }
// CHECK-ENCODING: [0x1a,0x7d,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e17d1a <unknown>

sub     za.d[w11, 2], {z8.d - z11.d}  // 11000001-11100001-01111101-00011010
// CHECK-INST: sub     za.d[w11, 2, vgx4], { z8.d - z11.d }
// CHECK-ENCODING: [0x1a,0x7d,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e17d1a <unknown>

sub     za.d[w9, 7, vgx4], {z12.d - z15.d}  // 11000001-11100001-00111101-10011111
// CHECK-INST: sub     za.d[w9, 7, vgx4], { z12.d - z15.d }
// CHECK-ENCODING: [0x9f,0x3d,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e13d9f <unknown>

sub     za.d[w9, 7], {z12.d - z15.d}  // 11000001-11100001-00111101-10011111
// CHECK-INST: sub     za.d[w9, 7, vgx4], { z12.d - z15.d }
// CHECK-ENCODING: [0x9f,0x3d,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e13d9f <unknown>

sub     za.d[w8, 0, vgx4], {z0.d - z3.d}, z0.d  // 11000001-01110000-00011000-00011000
// CHECK-INST: sub     za.d[w8, 0, vgx4], { z0.d - z3.d }, z0.d
// CHECK-ENCODING: [0x18,0x18,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1701818 <unknown>

sub     za.d[w8, 0], {z0.d - z3.d}, z0.d  // 11000001-01110000-00011000-00011000
// CHECK-INST: sub     za.d[w8, 0, vgx4], { z0.d - z3.d }, z0.d
// CHECK-ENCODING: [0x18,0x18,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1701818 <unknown>

sub     za.d[w10, 5, vgx4], {z10.d - z13.d}, z5.d  // 11000001-01110101-01011001-01011101
// CHECK-INST: sub     za.d[w10, 5, vgx4], { z10.d - z13.d }, z5.d
// CHECK-ENCODING: [0x5d,0x59,0x75,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c175595d <unknown>

sub     za.d[w10, 5], {z10.d - z13.d}, z5.d  // 11000001-01110101-01011001-01011101
// CHECK-INST: sub     za.d[w10, 5, vgx4], { z10.d - z13.d }, z5.d
// CHECK-ENCODING: [0x5d,0x59,0x75,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c175595d <unknown>

sub     za.d[w11, 7, vgx4], {z13.d - z16.d}, z8.d  // 11000001-01111000-01111001-10111111
// CHECK-INST: sub     za.d[w11, 7, vgx4], { z13.d - z16.d }, z8.d
// CHECK-ENCODING: [0xbf,0x79,0x78,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17879bf <unknown>

sub     za.d[w11, 7], {z13.d - z16.d}, z8.d  // 11000001-01111000-01111001-10111111
// CHECK-INST: sub     za.d[w11, 7, vgx4], { z13.d - z16.d }, z8.d
// CHECK-ENCODING: [0xbf,0x79,0x78,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17879bf <unknown>

sub     za.d[w11, 7, vgx4], {z31.d - z2.d}, z15.d  // 11000001-01111111-01111011-11111111
// CHECK-INST: sub     za.d[w11, 7, vgx4], { z31.d, z0.d, z1.d, z2.d }, z15.d
// CHECK-ENCODING: [0xff,0x7b,0x7f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17f7bff <unknown>

sub     za.d[w11, 7], {z31.d - z2.d}, z15.d  // 11000001-01111111-01111011-11111111
// CHECK-INST: sub     za.d[w11, 7, vgx4], { z31.d, z0.d, z1.d, z2.d }, z15.d
// CHECK-ENCODING: [0xff,0x7b,0x7f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17f7bff <unknown>

sub     za.d[w8, 5, vgx4], {z17.d - z20.d}, z0.d  // 11000001-01110000-00011010-00111101
// CHECK-INST: sub     za.d[w8, 5, vgx4], { z17.d - z20.d }, z0.d
// CHECK-ENCODING: [0x3d,0x1a,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1701a3d <unknown>

sub     za.d[w8, 5], {z17.d - z20.d}, z0.d  // 11000001-01110000-00011010-00111101
// CHECK-INST: sub     za.d[w8, 5, vgx4], { z17.d - z20.d }, z0.d
// CHECK-ENCODING: [0x3d,0x1a,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1701a3d <unknown>

sub     za.d[w8, 1, vgx4], {z1.d - z4.d}, z14.d  // 11000001-01111110-00011000-00111001
// CHECK-INST: sub     za.d[w8, 1, vgx4], { z1.d - z4.d }, z14.d
// CHECK-ENCODING: [0x39,0x18,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17e1839 <unknown>

sub     za.d[w8, 1], {z1.d - z4.d}, z14.d  // 11000001-01111110-00011000-00111001
// CHECK-INST: sub     za.d[w8, 1, vgx4], { z1.d - z4.d }, z14.d
// CHECK-ENCODING: [0x39,0x18,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17e1839 <unknown>

sub     za.d[w10, 0, vgx4], {z19.d - z22.d}, z4.d  // 11000001-01110100-01011010-01111000
// CHECK-INST: sub     za.d[w10, 0, vgx4], { z19.d - z22.d }, z4.d
// CHECK-ENCODING: [0x78,0x5a,0x74,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1745a78 <unknown>

sub     za.d[w10, 0], {z19.d - z22.d}, z4.d  // 11000001-01110100-01011010-01111000
// CHECK-INST: sub     za.d[w10, 0, vgx4], { z19.d - z22.d }, z4.d
// CHECK-ENCODING: [0x78,0x5a,0x74,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1745a78 <unknown>

sub     za.d[w8, 0, vgx4], {z12.d - z15.d}, z2.d  // 11000001-01110010-00011001-10011000
// CHECK-INST: sub     za.d[w8, 0, vgx4], { z12.d - z15.d }, z2.d
// CHECK-ENCODING: [0x98,0x19,0x72,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1721998 <unknown>

sub     za.d[w8, 0], {z12.d - z15.d}, z2.d  // 11000001-01110010-00011001-10011000
// CHECK-INST: sub     za.d[w8, 0, vgx4], { z12.d - z15.d }, z2.d
// CHECK-ENCODING: [0x98,0x19,0x72,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1721998 <unknown>

sub     za.d[w10, 1, vgx4], {z1.d - z4.d}, z10.d  // 11000001-01111010-01011000-00111001
// CHECK-INST: sub     za.d[w10, 1, vgx4], { z1.d - z4.d }, z10.d
// CHECK-ENCODING: [0x39,0x58,0x7a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17a5839 <unknown>

sub     za.d[w10, 1], {z1.d - z4.d}, z10.d  // 11000001-01111010-01011000-00111001
// CHECK-INST: sub     za.d[w10, 1, vgx4], { z1.d - z4.d }, z10.d
// CHECK-ENCODING: [0x39,0x58,0x7a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17a5839 <unknown>

sub     za.d[w8, 5, vgx4], {z22.d - z25.d}, z14.d  // 11000001-01111110-00011010-11011101
// CHECK-INST: sub     za.d[w8, 5, vgx4], { z22.d - z25.d }, z14.d
// CHECK-ENCODING: [0xdd,0x1a,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17e1add <unknown>

sub     za.d[w8, 5], {z22.d - z25.d}, z14.d  // 11000001-01111110-00011010-11011101
// CHECK-INST: sub     za.d[w8, 5, vgx4], { z22.d - z25.d }, z14.d
// CHECK-ENCODING: [0xdd,0x1a,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17e1add <unknown>

sub     za.d[w11, 2, vgx4], {z9.d - z12.d}, z1.d  // 11000001-01110001-01111001-00111010
// CHECK-INST: sub     za.d[w11, 2, vgx4], { z9.d - z12.d }, z1.d
// CHECK-ENCODING: [0x3a,0x79,0x71,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c171793a <unknown>

sub     za.d[w11, 2], {z9.d - z12.d}, z1.d  // 11000001-01110001-01111001-00111010
// CHECK-INST: sub     za.d[w11, 2, vgx4], { z9.d - z12.d }, z1.d
// CHECK-ENCODING: [0x3a,0x79,0x71,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c171793a <unknown>

sub     za.d[w9, 7, vgx4], {z12.d - z15.d}, z11.d  // 11000001-01111011-00111001-10011111
// CHECK-INST: sub     za.d[w9, 7, vgx4], { z12.d - z15.d }, z11.d
// CHECK-ENCODING: [0x9f,0x39,0x7b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17b399f <unknown>

sub     za.d[w9, 7], {z12.d - z15.d}, z11.d  // 11000001-01111011-00111001-10011111
// CHECK-INST: sub     za.d[w9, 7, vgx4], { z12.d - z15.d }, z11.d
// CHECK-ENCODING: [0x9f,0x39,0x7b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17b399f <unknown>


sub     za.d[w8, 0, vgx4], {z0.d - z3.d}, {z0.d - z3.d}  // 11000001-11100001-00011000-00011000
// CHECK-INST: sub     za.d[w8, 0, vgx4], { z0.d - z3.d }, { z0.d - z3.d }
// CHECK-ENCODING: [0x18,0x18,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11818 <unknown>

sub     za.d[w8, 0], {z0.d - z3.d}, {z0.d - z3.d}  // 11000001-11100001-00011000-00011000
// CHECK-INST: sub     za.d[w8, 0, vgx4], { z0.d - z3.d }, { z0.d - z3.d }
// CHECK-ENCODING: [0x18,0x18,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11818 <unknown>

sub     za.d[w10, 5, vgx4], {z8.d - z11.d}, {z20.d - z23.d}  // 11000001-11110101-01011001-00011101
// CHECK-INST: sub     za.d[w10, 5, vgx4], { z8.d - z11.d }, { z20.d - z23.d }
// CHECK-ENCODING: [0x1d,0x59,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f5591d <unknown>

sub     za.d[w10, 5], {z8.d - z11.d}, {z20.d - z23.d}  // 11000001-11110101-01011001-00011101
// CHECK-INST: sub     za.d[w10, 5, vgx4], { z8.d - z11.d }, { z20.d - z23.d }
// CHECK-ENCODING: [0x1d,0x59,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f5591d <unknown>

sub     za.d[w11, 7, vgx4], {z12.d - z15.d}, {z8.d - z11.d}  // 11000001-11101001-01111001-10011111
// CHECK-INST: sub     za.d[w11, 7, vgx4], { z12.d - z15.d }, { z8.d - z11.d }
// CHECK-ENCODING: [0x9f,0x79,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e9799f <unknown>

sub     za.d[w11, 7], {z12.d - z15.d}, {z8.d - z11.d}  // 11000001-11101001-01111001-10011111
// CHECK-INST: sub     za.d[w11, 7, vgx4], { z12.d - z15.d }, { z8.d - z11.d }
// CHECK-ENCODING: [0x9f,0x79,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e9799f <unknown>

sub     za.d[w11, 7, vgx4], {z28.d - z31.d}, {z28.d - z31.d}  // 11000001-11111101-01111011-10011111
// CHECK-INST: sub     za.d[w11, 7, vgx4], { z28.d - z31.d }, { z28.d - z31.d }
// CHECK-ENCODING: [0x9f,0x7b,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd7b9f <unknown>

sub     za.d[w11, 7], {z28.d - z31.d}, {z28.d - z31.d}  // 11000001-11111101-01111011-10011111
// CHECK-INST: sub     za.d[w11, 7, vgx4], { z28.d - z31.d }, { z28.d - z31.d }
// CHECK-ENCODING: [0x9f,0x7b,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd7b9f <unknown>

sub     za.d[w8, 5, vgx4], {z16.d - z19.d}, {z16.d - z19.d}  // 11000001-11110001-00011010-00011101
// CHECK-INST: sub     za.d[w8, 5, vgx4], { z16.d - z19.d }, { z16.d - z19.d }
// CHECK-ENCODING: [0x1d,0x1a,0xf1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f11a1d <unknown>

sub     za.d[w8, 5], {z16.d - z19.d}, {z16.d - z19.d}  // 11000001-11110001-00011010-00011101
// CHECK-INST: sub     za.d[w8, 5, vgx4], { z16.d - z19.d }, { z16.d - z19.d }
// CHECK-ENCODING: [0x1d,0x1a,0xf1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f11a1d <unknown>

sub     za.d[w8, 1, vgx4], {z0.d - z3.d}, {z28.d - z31.d}  // 11000001-11111101-00011000-00011001
// CHECK-INST: sub     za.d[w8, 1, vgx4], { z0.d - z3.d }, { z28.d - z31.d }
// CHECK-ENCODING: [0x19,0x18,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd1819 <unknown>

sub     za.d[w8, 1], {z0.d - z3.d}, {z28.d - z31.d}  // 11000001-11111101-00011000-00011001
// CHECK-INST: sub     za.d[w8, 1, vgx4], { z0.d - z3.d }, { z28.d - z31.d }
// CHECK-ENCODING: [0x19,0x18,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd1819 <unknown>

sub     za.d[w10, 0, vgx4], {z16.d - z19.d}, {z20.d - z23.d}  // 11000001-11110101-01011010-00011000
// CHECK-INST: sub     za.d[w10, 0, vgx4], { z16.d - z19.d }, { z20.d - z23.d }
// CHECK-ENCODING: [0x18,0x5a,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f55a18 <unknown>

sub     za.d[w10, 0], {z16.d - z19.d}, {z20.d - z23.d}  // 11000001-11110101-01011010-00011000
// CHECK-INST: sub     za.d[w10, 0, vgx4], { z16.d - z19.d }, { z20.d - z23.d }
// CHECK-ENCODING: [0x18,0x5a,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f55a18 <unknown>

sub     za.d[w8, 0, vgx4], {z12.d - z15.d}, {z0.d - z3.d}  // 11000001-11100001-00011001-10011000
// CHECK-INST: sub     za.d[w8, 0, vgx4], { z12.d - z15.d }, { z0.d - z3.d }
// CHECK-ENCODING: [0x98,0x19,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11998 <unknown>

sub     za.d[w8, 0], {z12.d - z15.d}, {z0.d - z3.d}  // 11000001-11100001-00011001-10011000
// CHECK-INST: sub     za.d[w8, 0, vgx4], { z12.d - z15.d }, { z0.d - z3.d }
// CHECK-ENCODING: [0x98,0x19,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11998 <unknown>

sub     za.d[w10, 1, vgx4], {z0.d - z3.d}, {z24.d - z27.d}  // 11000001-11111001-01011000-00011001
// CHECK-INST: sub     za.d[w10, 1, vgx4], { z0.d - z3.d }, { z24.d - z27.d }
// CHECK-ENCODING: [0x19,0x58,0xf9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f95819 <unknown>

sub     za.d[w10, 1], {z0.d - z3.d}, {z24.d - z27.d}  // 11000001-11111001-01011000-00011001
// CHECK-INST: sub     za.d[w10, 1, vgx4], { z0.d - z3.d }, { z24.d - z27.d }
// CHECK-ENCODING: [0x19,0x58,0xf9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f95819 <unknown>

sub     za.d[w8, 5, vgx4], {z20.d - z23.d}, {z28.d - z31.d}  // 11000001-11111101-00011010-10011101
// CHECK-INST: sub     za.d[w8, 5, vgx4], { z20.d - z23.d }, { z28.d - z31.d }
// CHECK-ENCODING: [0x9d,0x1a,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd1a9d <unknown>

sub     za.d[w8, 5], {z20.d - z23.d}, {z28.d - z31.d}  // 11000001-11111101-00011010-10011101
// CHECK-INST: sub     za.d[w8, 5, vgx4], { z20.d - z23.d }, { z28.d - z31.d }
// CHECK-ENCODING: [0x9d,0x1a,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd1a9d <unknown>

sub     za.d[w11, 2, vgx4], {z8.d - z11.d}, {z0.d - z3.d}  // 11000001-11100001-01111001-00011010
// CHECK-INST: sub     za.d[w11, 2, vgx4], { z8.d - z11.d }, { z0.d - z3.d }
// CHECK-ENCODING: [0x1a,0x79,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e1791a <unknown>

sub     za.d[w11, 2], {z8.d - z11.d}, {z0.d - z3.d}  // 11000001-11100001-01111001-00011010
// CHECK-INST: sub     za.d[w11, 2, vgx4], { z8.d - z11.d }, { z0.d - z3.d }
// CHECK-ENCODING: [0x1a,0x79,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e1791a <unknown>

sub     za.d[w9, 7, vgx4], {z12.d - z15.d}, {z8.d - z11.d}  // 11000001-11101001-00111001-10011111
// CHECK-INST: sub     za.d[w9, 7, vgx4], { z12.d - z15.d }, { z8.d - z11.d }
// CHECK-ENCODING: [0x9f,0x39,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e9399f <unknown>

sub     za.d[w9, 7], {z12.d - z15.d}, {z8.d - z11.d}  // 11000001-11101001-00111001-10011111
// CHECK-INST: sub     za.d[w9, 7, vgx4], { z12.d - z15.d }, { z8.d - z11.d }
// CHECK-ENCODING: [0x9f,0x39,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e9399f <unknown>

