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


luti4   z0.h, zt0, z0[0]  // 11000000-11001010-00010000-00000000
// CHECK-INST: luti4   z0.h, zt0, z0[0]
// CHECK-ENCODING: [0x00,0x10,0xca,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0ca1000 <unknown>

luti4   z21.h, zt0, z10[5]  // 11000000-11001011-01010001-01010101
// CHECK-INST: luti4   z21.h, zt0, z10[5]
// CHECK-ENCODING: [0x55,0x51,0xcb,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0cb5155 <unknown>

luti4   z23.h, zt0, z13[3]  // 11000000-11001010-11010001-10110111
// CHECK-INST: luti4   z23.h, zt0, z13[3]
// CHECK-ENCODING: [0xb7,0xd1,0xca,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0cad1b7 <unknown>

luti4   z31.h, zt0, z31[7]  // 11000000-11001011-11010011-11111111
// CHECK-INST: luti4   z31.h, zt0, z31[7]
// CHECK-ENCODING: [0xff,0xd3,0xcb,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0cbd3ff <unknown>


luti4   z0.s, zt0, z0[0]  // 11000000-11001010-00100000-00000000
// CHECK-INST: luti4   z0.s, zt0, z0[0]
// CHECK-ENCODING: [0x00,0x20,0xca,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0ca2000 <unknown>

luti4   z21.s, zt0, z10[5]  // 11000000-11001011-01100001-01010101
// CHECK-INST: luti4   z21.s, zt0, z10[5]
// CHECK-ENCODING: [0x55,0x61,0xcb,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0cb6155 <unknown>

luti4   z23.s, zt0, z13[3]  // 11000000-11001010-11100001-10110111
// CHECK-INST: luti4   z23.s, zt0, z13[3]
// CHECK-ENCODING: [0xb7,0xe1,0xca,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0cae1b7 <unknown>

luti4   z31.s, zt0, z31[7]  // 11000000-11001011-11100011-11111111
// CHECK-INST: luti4   z31.s, zt0, z31[7]
// CHECK-ENCODING: [0xff,0xe3,0xcb,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0cbe3ff <unknown>


luti4   z0.b, zt0, z0[0]  // 11000000-11001010-00000000-00000000
// CHECK-INST: luti4   z0.b, zt0, z0[0]
// CHECK-ENCODING: [0x00,0x00,0xca,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0ca0000 <unknown>

luti4   z21.b, zt0, z10[5]  // 11000000-11001011-01000001-01010101
// CHECK-INST: luti4   z21.b, zt0, z10[5]
// CHECK-ENCODING: [0x55,0x41,0xcb,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0cb4155 <unknown>

luti4   z23.b, zt0, z13[3]  // 11000000-11001010-11000001-10110111
// CHECK-INST: luti4   z23.b, zt0, z13[3]
// CHECK-ENCODING: [0xb7,0xc1,0xca,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0cac1b7 <unknown>

luti4   z31.b, zt0, z31[7]  // 11000000-11001011-11000011-11111111
// CHECK-INST: luti4   z31.b, zt0, z31[7]
// CHECK-ENCODING: [0xff,0xc3,0xcb,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0cbc3ff <unknown>


luti4   {z0.h - z1.h}, zt0, z0[0]  // 11000000-10001010-01010000-00000000
// CHECK-INST: luti4   { z0.h, z1.h }, zt0, z0[0]
// CHECK-ENCODING: [0x00,0x50,0x8a,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08a5000 <unknown>

luti4   {z20.h - z21.h}, zt0, z10[2]  // 11000000-10001011-01010001-01010100
// CHECK-INST: luti4   { z20.h, z21.h }, zt0, z10[2]
// CHECK-ENCODING: [0x54,0x51,0x8b,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08b5154 <unknown>

luti4   {z22.h - z23.h}, zt0, z13[1]  // 11000000-10001010-11010001-10110110
// CHECK-INST: luti4   { z22.h, z23.h }, zt0, z13[1]
// CHECK-ENCODING: [0xb6,0xd1,0x8a,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08ad1b6 <unknown>

luti4   {z30.h - z31.h}, zt0, z31[3]  // 11000000-10001011-11010011-11111110
// CHECK-INST: luti4   { z30.h, z31.h }, zt0, z31[3]
// CHECK-ENCODING: [0xfe,0xd3,0x8b,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08bd3fe <unknown>


luti4   {z0.s - z1.s}, zt0, z0[0]  // 11000000-10001010-01100000-00000000
// CHECK-INST: luti4   { z0.s, z1.s }, zt0, z0[0]
// CHECK-ENCODING: [0x00,0x60,0x8a,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08a6000 <unknown>

luti4   {z20.s - z21.s}, zt0, z10[2]  // 11000000-10001011-01100001-01010100
// CHECK-INST: luti4   { z20.s, z21.s }, zt0, z10[2]
// CHECK-ENCODING: [0x54,0x61,0x8b,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08b6154 <unknown>

luti4   {z22.s - z23.s}, zt0, z13[1]  // 11000000-10001010-11100001-10110110
// CHECK-INST: luti4   { z22.s, z23.s }, zt0, z13[1]
// CHECK-ENCODING: [0xb6,0xe1,0x8a,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08ae1b6 <unknown>

luti4   {z30.s - z31.s}, zt0, z31[3]  // 11000000-10001011-11100011-11111110
// CHECK-INST: luti4   { z30.s, z31.s }, zt0, z31[3]
// CHECK-ENCODING: [0xfe,0xe3,0x8b,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08be3fe <unknown>


luti4   {z0.b - z1.b}, zt0, z0[0]  // 11000000-10001010-01000000-00000000
// CHECK-INST: luti4   { z0.b, z1.b }, zt0, z0[0]
// CHECK-ENCODING: [0x00,0x40,0x8a,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08a4000 <unknown>

luti4   {z20.b - z21.b}, zt0, z10[2]  // 11000000-10001011-01000001-01010100
// CHECK-INST: luti4   { z20.b, z21.b }, zt0, z10[2]
// CHECK-ENCODING: [0x54,0x41,0x8b,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08b4154 <unknown>

luti4   {z22.b - z23.b}, zt0, z13[1]  // 11000000-10001010-11000001-10110110
// CHECK-INST: luti4   { z22.b, z23.b }, zt0, z13[1]
// CHECK-ENCODING: [0xb6,0xc1,0x8a,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08ac1b6 <unknown>

luti4   {z30.b - z31.b}, zt0, z31[3]  // 11000000-10001011-11000011-11111110
// CHECK-INST: luti4   { z30.b, z31.b }, zt0, z31[3]
// CHECK-ENCODING: [0xfe,0xc3,0x8b,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08bc3fe <unknown>


luti4   {z0.h - z3.h}, zt0, z0[0]  // 11000000-10001010-10010000-00000000
// CHECK-INST: luti4   { z0.h - z3.h }, zt0, z0[0]
// CHECK-ENCODING: [0x00,0x90,0x8a,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08a9000 <unknown>

luti4   {z20.h - z23.h}, zt0, z10[1]  // 11000000-10001011-10010001-01010100
// CHECK-INST: luti4   { z20.h - z23.h }, zt0, z10[1]
// CHECK-ENCODING: [0x54,0x91,0x8b,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08b9154 <unknown>

luti4   {z20.h - z23.h}, zt0, z13[0]  // 11000000-10001010-10010001-10110100
// CHECK-INST: luti4   { z20.h - z23.h }, zt0, z13[0]
// CHECK-ENCODING: [0xb4,0x91,0x8a,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08a91b4 <unknown>

luti4   {z28.h - z31.h}, zt0, z31[1]  // 11000000-10001011-10010011-11111100
// CHECK-INST: luti4   { z28.h - z31.h }, zt0, z31[1]
// CHECK-ENCODING: [0xfc,0x93,0x8b,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08b93fc <unknown>


luti4   {z0.s - z3.s}, zt0, z0[0]  // 11000000-10001010-10100000-00000000
// CHECK-INST: luti4   { z0.s - z3.s }, zt0, z0[0]
// CHECK-ENCODING: [0x00,0xa0,0x8a,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08aa000 <unknown>

luti4   {z20.s - z23.s}, zt0, z10[1]  // 11000000-10001011-10100001-01010100
// CHECK-INST: luti4   { z20.s - z23.s }, zt0, z10[1]
// CHECK-ENCODING: [0x54,0xa1,0x8b,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08ba154 <unknown>

luti4   {z20.s - z23.s}, zt0, z13[0]  // 11000000-10001010-10100001-10110100
// CHECK-INST: luti4   { z20.s - z23.s }, zt0, z13[0]
// CHECK-ENCODING: [0xb4,0xa1,0x8a,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08aa1b4 <unknown>

luti4   {z28.s - z31.s}, zt0, z31[1]  // 11000000-10001011-10100011-11111100
// CHECK-INST: luti4   { z28.s - z31.s }, zt0, z31[1]
// CHECK-ENCODING: [0xfc,0xa3,0x8b,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08ba3fc <unknown>

