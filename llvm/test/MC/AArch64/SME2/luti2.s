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


luti2   z0.h, zt0, z0[0]  // 11000000-11001100-00010000-00000000
// CHECK-INST: luti2   z0.h, zt0, z0[0]
// CHECK-ENCODING: [0x00,0x10,0xcc,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0cc1000 <unknown>

luti2   z21.h, zt0, z10[5]  // 11000000-11001101-01010001-01010101
// CHECK-INST: luti2   z21.h, zt0, z10[5]
// CHECK-ENCODING: [0x55,0x51,0xcd,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0cd5155 <unknown>

luti2   z23.h, zt0, z13[3]  // 11000000-11001100-11010001-10110111
// CHECK-INST: luti2   z23.h, zt0, z13[3]
// CHECK-ENCODING: [0xb7,0xd1,0xcc,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0ccd1b7 <unknown>

luti2   z31.h, zt0, z31[15]  // 11000000-11001111-11010011-11111111
// CHECK-INST: luti2   z31.h, zt0, z31[15]
// CHECK-ENCODING: [0xff,0xd3,0xcf,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0cfd3ff <unknown>


luti2   z0.s, zt0, z0[0]  // 11000000-11001100-00100000-00000000
// CHECK-INST: luti2   z0.s, zt0, z0[0]
// CHECK-ENCODING: [0x00,0x20,0xcc,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0cc2000 <unknown>

luti2   z21.s, zt0, z10[5]  // 11000000-11001101-01100001-01010101
// CHECK-INST: luti2   z21.s, zt0, z10[5]
// CHECK-ENCODING: [0x55,0x61,0xcd,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0cd6155 <unknown>

luti2   z23.s, zt0, z13[3]  // 11000000-11001100-11100001-10110111
// CHECK-INST: luti2   z23.s, zt0, z13[3]
// CHECK-ENCODING: [0xb7,0xe1,0xcc,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0cce1b7 <unknown>

luti2   z31.s, zt0, z31[15]  // 11000000-11001111-11100011-11111111
// CHECK-INST: luti2   z31.s, zt0, z31[15]
// CHECK-ENCODING: [0xff,0xe3,0xcf,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0cfe3ff <unknown>


luti2   z0.b, zt0, z0[0]  // 11000000-11001100-00000000-00000000
// CHECK-INST: luti2   z0.b, zt0, z0[0]
// CHECK-ENCODING: [0x00,0x00,0xcc,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0cc0000 <unknown>

luti2   z21.b, zt0, z10[5]  // 11000000-11001101-01000001-01010101
// CHECK-INST: luti2   z21.b, zt0, z10[5]
// CHECK-ENCODING: [0x55,0x41,0xcd,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0cd4155 <unknown>

luti2   z23.b, zt0, z13[3]  // 11000000-11001100-11000001-10110111
// CHECK-INST: luti2   z23.b, zt0, z13[3]
// CHECK-ENCODING: [0xb7,0xc1,0xcc,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0ccc1b7 <unknown>

luti2   z31.b, zt0, z31[15]  // 11000000-11001111-11000011-11111111
// CHECK-INST: luti2   z31.b, zt0, z31[15]
// CHECK-ENCODING: [0xff,0xc3,0xcf,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0cfc3ff <unknown>


luti2   {z0.h - z1.h}, zt0, z0[0]  // 11000000-10001100-01010000-00000000
// CHECK-INST: luti2   { z0.h, z1.h }, zt0, z0[0]
// CHECK-ENCODING: [0x00,0x50,0x8c,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08c5000 <unknown>

luti2   {z20.h - z21.h}, zt0, z10[2]  // 11000000-10001101-01010001-01010100
// CHECK-INST: luti2   { z20.h, z21.h }, zt0, z10[2]
// CHECK-ENCODING: [0x54,0x51,0x8d,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08d5154 <unknown>

luti2   {z22.h - z23.h}, zt0, z13[1]  // 11000000-10001100-11010001-10110110
// CHECK-INST: luti2   { z22.h, z23.h }, zt0, z13[1]
// CHECK-ENCODING: [0xb6,0xd1,0x8c,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08cd1b6 <unknown>

luti2   {z30.h - z31.h}, zt0, z31[7]  // 11000000-10001111-11010011-11111110
// CHECK-INST: luti2   { z30.h, z31.h }, zt0, z31[7]
// CHECK-ENCODING: [0xfe,0xd3,0x8f,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08fd3fe <unknown>


luti2   {z0.s - z1.s}, zt0, z0[0]  // 11000000-10001100-01100000-00000000
// CHECK-INST: luti2   { z0.s, z1.s }, zt0, z0[0]
// CHECK-ENCODING: [0x00,0x60,0x8c,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08c6000 <unknown>

luti2   {z20.s - z21.s}, zt0, z10[2]  // 11000000-10001101-01100001-01010100
// CHECK-INST: luti2   { z20.s, z21.s }, zt0, z10[2]
// CHECK-ENCODING: [0x54,0x61,0x8d,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08d6154 <unknown>

luti2   {z22.s - z23.s}, zt0, z13[1]  // 11000000-10001100-11100001-10110110
// CHECK-INST: luti2   { z22.s, z23.s }, zt0, z13[1]
// CHECK-ENCODING: [0xb6,0xe1,0x8c,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08ce1b6 <unknown>

luti2   {z30.s - z31.s}, zt0, z31[7]  // 11000000-10001111-11100011-11111110
// CHECK-INST: luti2   { z30.s, z31.s }, zt0, z31[7]
// CHECK-ENCODING: [0xfe,0xe3,0x8f,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08fe3fe <unknown>


luti2   {z0.b - z1.b}, zt0, z0[0]  // 11000000-10001100-01000000-00000000
// CHECK-INST: luti2   { z0.b, z1.b }, zt0, z0[0]
// CHECK-ENCODING: [0x00,0x40,0x8c,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08c4000 <unknown>

luti2   {z20.b - z21.b}, zt0, z10[2]  // 11000000-10001101-01000001-01010100
// CHECK-INST: luti2   { z20.b, z21.b }, zt0, z10[2]
// CHECK-ENCODING: [0x54,0x41,0x8d,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08d4154 <unknown>

luti2   {z22.b - z23.b}, zt0, z13[1]  // 11000000-10001100-11000001-10110110
// CHECK-INST: luti2   { z22.b, z23.b }, zt0, z13[1]
// CHECK-ENCODING: [0xb6,0xc1,0x8c,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08cc1b6 <unknown>

luti2   {z30.b - z31.b}, zt0, z31[7]  // 11000000-10001111-11000011-11111110
// CHECK-INST: luti2   { z30.b, z31.b }, zt0, z31[7]
// CHECK-ENCODING: [0xfe,0xc3,0x8f,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08fc3fe <unknown>


luti2   {z0.h - z3.h}, zt0, z0[0]  // 11000000-10001100-10010000-00000000
// CHECK-INST: luti2   { z0.h - z3.h }, zt0, z0[0]
// CHECK-ENCODING: [0x00,0x90,0x8c,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08c9000 <unknown>

luti2   {z20.h - z23.h}, zt0, z10[1]  // 11000000-10001101-10010001-01010100
// CHECK-INST: luti2   { z20.h - z23.h }, zt0, z10[1]
// CHECK-ENCODING: [0x54,0x91,0x8d,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08d9154 <unknown>

luti2   {z20.h - z23.h}, zt0, z13[0]  // 11000000-10001100-10010001-10110100
// CHECK-INST: luti2   { z20.h - z23.h }, zt0, z13[0]
// CHECK-ENCODING: [0xb4,0x91,0x8c,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08c91b4 <unknown>

luti2   {z28.h - z31.h}, zt0, z31[3]  // 11000000-10001111-10010011-11111100
// CHECK-INST: luti2   { z28.h - z31.h }, zt0, z31[3]
// CHECK-ENCODING: [0xfc,0x93,0x8f,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08f93fc <unknown>


luti2   {z0.s - z3.s}, zt0, z0[0]  // 11000000-10001100-10100000-00000000
// CHECK-INST: luti2   { z0.s - z3.s }, zt0, z0[0]
// CHECK-ENCODING: [0x00,0xa0,0x8c,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08ca000 <unknown>

luti2   {z20.s - z23.s}, zt0, z10[1]  // 11000000-10001101-10100001-01010100
// CHECK-INST: luti2   { z20.s - z23.s }, zt0, z10[1]
// CHECK-ENCODING: [0x54,0xa1,0x8d,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08da154 <unknown>

luti2   {z20.s - z23.s}, zt0, z13[0]  // 11000000-10001100-10100001-10110100
// CHECK-INST: luti2   { z20.s - z23.s }, zt0, z13[0]
// CHECK-ENCODING: [0xb4,0xa1,0x8c,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08ca1b4 <unknown>

luti2   {z28.s - z31.s}, zt0, z31[3]  // 11000000-10001111-10100011-11111100
// CHECK-INST: luti2   { z28.s - z31.s }, zt0, z31[3]
// CHECK-ENCODING: [0xfc,0xa3,0x8f,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08fa3fc <unknown>


luti2   {z0.b - z3.b}, zt0, z0[0]  // 11000000-10001100-10000000-00000000
// CHECK-INST: luti2   { z0.b - z3.b }, zt0, z0[0]
// CHECK-ENCODING: [0x00,0x80,0x8c,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08c8000 <unknown>

luti2   {z20.b - z23.b}, zt0, z10[1]  // 11000000-10001101-10000001-01010100
// CHECK-INST: luti2   { z20.b - z23.b }, zt0, z10[1]
// CHECK-ENCODING: [0x54,0x81,0x8d,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08d8154 <unknown>

luti2   {z20.b - z23.b}, zt0, z13[0]  // 11000000-10001100-10000001-10110100
// CHECK-INST: luti2   { z20.b - z23.b }, zt0, z13[0]
// CHECK-ENCODING: [0xb4,0x81,0x8c,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08c81b4 <unknown>

luti2   {z28.b - z31.b}, zt0, z31[3]  // 11000000-10001111-10000011-11111100
// CHECK-INST: luti2   { z28.b - z31.b }, zt0, z31[3]
// CHECK-ENCODING: [0xfc,0x83,0x8f,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08f83fc <unknown>

