// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2  - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


zip     {z0.q - z1.q}, z0.q, z0.q  // 11000001-00100000-11010100-00000000
// CHECK-INST: zip     { z0.q, z1.q }, z0.q, z0.q
// CHECK-ENCODING: [0x00,0xd4,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c120d400 <unknown>

zip     {z20.q - z21.q}, z10.q, z21.q  // 11000001-00110101-11010101-01010100
// CHECK-INST: zip     { z20.q, z21.q }, z10.q, z21.q
// CHECK-ENCODING: [0x54,0xd5,0x35,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c135d554 <unknown>

zip     {z22.q - z23.q}, z13.q, z8.q  // 11000001-00101000-11010101-10110110
// CHECK-INST: zip     { z22.q, z23.q }, z13.q, z8.q
// CHECK-ENCODING: [0xb6,0xd5,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c128d5b6 <unknown>

zip     {z30.q - z31.q}, z31.q, z31.q  // 11000001-00111111-11010111-11111110
// CHECK-INST: zip     { z30.q, z31.q }, z31.q, z31.q
// CHECK-ENCODING: [0xfe,0xd7,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13fd7fe <unknown>


zip     {z0.h - z1.h}, z0.h, z0.h  // 11000001-01100000-11010000-00000000
// CHECK-INST: zip     { z0.h, z1.h }, z0.h, z0.h
// CHECK-ENCODING: [0x00,0xd0,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c160d000 <unknown>

zip     {z20.h - z21.h}, z10.h, z21.h  // 11000001-01110101-11010001-01010100
// CHECK-INST: zip     { z20.h, z21.h }, z10.h, z21.h
// CHECK-ENCODING: [0x54,0xd1,0x75,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c175d154 <unknown>

zip     {z22.h - z23.h}, z13.h, z8.h  // 11000001-01101000-11010001-10110110
// CHECK-INST: zip     { z22.h, z23.h }, z13.h, z8.h
// CHECK-ENCODING: [0xb6,0xd1,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c168d1b6 <unknown>

zip     {z30.h - z31.h}, z31.h, z31.h  // 11000001-01111111-11010011-11111110
// CHECK-INST: zip     { z30.h, z31.h }, z31.h, z31.h
// CHECK-ENCODING: [0xfe,0xd3,0x7f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17fd3fe <unknown>


zip     {z0.s - z1.s}, z0.s, z0.s  // 11000001-10100000-11010000-00000000
// CHECK-INST: zip     { z0.s, z1.s }, z0.s, z0.s
// CHECK-ENCODING: [0x00,0xd0,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a0d000 <unknown>

zip     {z20.s - z21.s}, z10.s, z21.s  // 11000001-10110101-11010001-01010100
// CHECK-INST: zip     { z20.s, z21.s }, z10.s, z21.s
// CHECK-ENCODING: [0x54,0xd1,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b5d154 <unknown>

zip     {z22.s - z23.s}, z13.s, z8.s  // 11000001-10101000-11010001-10110110
// CHECK-INST: zip     { z22.s, z23.s }, z13.s, z8.s
// CHECK-ENCODING: [0xb6,0xd1,0xa8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a8d1b6 <unknown>

zip     {z30.s - z31.s}, z31.s, z31.s  // 11000001-10111111-11010011-11111110
// CHECK-INST: zip     { z30.s, z31.s }, z31.s, z31.s
// CHECK-ENCODING: [0xfe,0xd3,0xbf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bfd3fe <unknown>


zip     {z0.d - z1.d}, z0.d, z0.d  // 11000001-11100000-11010000-00000000
// CHECK-INST: zip     { z0.d, z1.d }, z0.d, z0.d
// CHECK-ENCODING: [0x00,0xd0,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e0d000 <unknown>

zip     {z20.d - z21.d}, z10.d, z21.d  // 11000001-11110101-11010001-01010100
// CHECK-INST: zip     { z20.d, z21.d }, z10.d, z21.d
// CHECK-ENCODING: [0x54,0xd1,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f5d154 <unknown>

zip     {z22.d - z23.d}, z13.d, z8.d  // 11000001-11101000-11010001-10110110
// CHECK-INST: zip     { z22.d, z23.d }, z13.d, z8.d
// CHECK-ENCODING: [0xb6,0xd1,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e8d1b6 <unknown>

zip     {z30.d - z31.d}, z31.d, z31.d  // 11000001-11111111-11010011-11111110
// CHECK-INST: zip     { z30.d, z31.d }, z31.d, z31.d
// CHECK-ENCODING: [0xfe,0xd3,0xff,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ffd3fe <unknown>


zip     {z0.b - z1.b}, z0.b, z0.b  // 11000001-00100000-11010000-00000000
// CHECK-INST: zip     { z0.b, z1.b }, z0.b, z0.b
// CHECK-ENCODING: [0x00,0xd0,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c120d000 <unknown>

zip     {z20.b, z21.b}, z10.b, z21.b  // 11000001-00110101-11010001-01010100
// CHECK-INST: zip     { z20.b, z21.b }, z10.b, z21.b
// CHECK-ENCODING: [0x54,0xd1,0x35,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c135d154 <unknown>

zip     {z22.b - z23.b}, z13.b, z8.b  // 11000001-00101000-11010001-10110110
// CHECK-INST: zip     { z22.b, z23.b }, z13.b, z8.b
// CHECK-ENCODING: [0xb6,0xd1,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c128d1b6 <unknown>

zip     {z30.b - z31.b}, z31.b, z31.b  // 11000001-00111111-11010011-11111110
// CHECK-INST: zip     { z30.b, z31.b }, z31.b, z31.b
// CHECK-ENCODING: [0xfe,0xd3,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13fd3fe <unknown>


zip     {z0.q - z3.q}, {z0.q - z3.q}  // 11000001-00110111-11100000-00000000
// CHECK-INST: zip     { z0.q - z3.q }, { z0.q - z3.q }
// CHECK-ENCODING: [0x00,0xe0,0x37,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c137e000 <unknown>

zip     {z20.q - z23.q}, {z8.q - z11.q}  // 11000001-00110111-11100001-00010100
// CHECK-INST: zip     { z20.q - z23.q }, { z8.q - z11.q }
// CHECK-ENCODING: [0x14,0xe1,0x37,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c137e114 <unknown>

zip     {z20.q - z23.q}, {z12.q - z15.q}  // 11000001-00110111-11100001-10010100
// CHECK-INST: zip     { z20.q - z23.q }, { z12.q - z15.q }
// CHECK-ENCODING: [0x94,0xe1,0x37,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c137e194 <unknown>

zip     {z28.q - z31.q}, {z28.q - z31.q}  // 11000001-00110111-11100011-10011100
// CHECK-INST: zip     { z28.q - z31.q }, { z28.q - z31.q }
// CHECK-ENCODING: [0x9c,0xe3,0x37,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c137e39c <unknown>


zip     {z0.h - z3.h}, {z0.h - z3.h}  // 11000001-01110110-11100000-00000000
// CHECK-INST: zip     { z0.h - z3.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x00,0xe0,0x76,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c176e000 <unknown>

zip     {z20.h - z23.h}, {z8.h - z11.h}  // 11000001-01110110-11100001-00010100
// CHECK-INST: zip     { z20.h - z23.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x14,0xe1,0x76,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c176e114 <unknown>

zip     {z20.h - z23.h}, {z12.h - z15.h}  // 11000001-01110110-11100001-10010100
// CHECK-INST: zip     { z20.h - z23.h }, { z12.h - z15.h }
// CHECK-ENCODING: [0x94,0xe1,0x76,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c176e194 <unknown>

zip     {z28.h - z31.h}, {z28.h - z31.h}  // 11000001-01110110-11100011-10011100
// CHECK-INST: zip     { z28.h - z31.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x9c,0xe3,0x76,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c176e39c <unknown>


zip     {z0.s - z3.s}, {z0.s - z3.s}  // 11000001-10110110-11100000-00000000
// CHECK-INST: zip     { z0.s - z3.s }, { z0.s - z3.s }
// CHECK-ENCODING: [0x00,0xe0,0xb6,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b6e000 <unknown>

zip     {z20.s - z23.s}, {z8.s - z11.s}  // 11000001-10110110-11100001-00010100
// CHECK-INST: zip     { z20.s - z23.s }, { z8.s - z11.s }
// CHECK-ENCODING: [0x14,0xe1,0xb6,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b6e114 <unknown>

zip     {z20.s - z23.s}, {z12.s - z15.s}  // 11000001-10110110-11100001-10010100
// CHECK-INST: zip     { z20.s - z23.s }, { z12.s - z15.s }
// CHECK-ENCODING: [0x94,0xe1,0xb6,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b6e194 <unknown>

zip     {z28.s - z31.s}, {z28.s - z31.s}  // 11000001-10110110-11100011-10011100
// CHECK-INST: zip     { z28.s - z31.s }, { z28.s - z31.s }
// CHECK-ENCODING: [0x9c,0xe3,0xb6,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b6e39c <unknown>


zip     {z0.d - z3.d}, {z0.d - z3.d}  // 11000001-11110110-11100000-00000000
// CHECK-INST: zip     { z0.d - z3.d }, { z0.d - z3.d }
// CHECK-ENCODING: [0x00,0xe0,0xf6,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f6e000 <unknown>

zip     {z20.d - z23.d}, {z8.d - z11.d}  // 11000001-11110110-11100001-00010100
// CHECK-INST: zip     { z20.d - z23.d }, { z8.d - z11.d }
// CHECK-ENCODING: [0x14,0xe1,0xf6,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f6e114 <unknown>

zip     {z20.d - z23.d}, {z12.d - z15.d}  // 11000001-11110110-11100001-10010100
// CHECK-INST: zip     { z20.d - z23.d }, { z12.d - z15.d }
// CHECK-ENCODING: [0x94,0xe1,0xf6,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f6e194 <unknown>

zip     {z28.d - z31.d}, {z28.d - z31.d}  // 11000001-11110110-11100011-10011100
// CHECK-INST: zip     { z28.d - z31.d }, { z28.d - z31.d }
// CHECK-ENCODING: [0x9c,0xe3,0xf6,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f6e39c <unknown>


zip     {z0.b - z3.b}, {z0.b - z3.b}  // 11000001-00110110-11100000-00000000
// CHECK-INST: zip     { z0.b - z3.b }, { z0.b - z3.b }
// CHECK-ENCODING: [0x00,0xe0,0x36,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c136e000 <unknown>

zip     {z20.b - z23.b}, {z8.b - z11.b}  // 11000001-00110110-11100001-00010100
// CHECK-INST: zip     { z20.b - z23.b }, { z8.b - z11.b }
// CHECK-ENCODING: [0x14,0xe1,0x36,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c136e114 <unknown>

zip     {z20.b - z23.b}, {z12.b - z15.b}  // 11000001-00110110-11100001-10010100
// CHECK-INST: zip     { z20.b - z23.b }, { z12.b - z15.b }
// CHECK-ENCODING: [0x94,0xe1,0x36,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c136e194 <unknown>

zip     {z28.b - z31.b}, {z28.b - z31.b}  // 11000001-00110110-11100011-10011100
// CHECK-INST: zip     { z28.b - z31.b }, { z28.b - z31.b }
// CHECK-ENCODING: [0x9c,0xe3,0x36,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c136e39c <unknown>

