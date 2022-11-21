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


uzp     {z0.q - z1.q}, z0.q, z0.q  // 11000001-00100000-11010100-00000001
// CHECK-INST: uzp     { z0.q, z1.q }, z0.q, z0.q
// CHECK-ENCODING: [0x01,0xd4,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c120d401 <unknown>

uzp     {z20.q - z21.q}, z10.q, z21.q  // 11000001-00110101-11010101-01010101
// CHECK-INST: uzp     { z20.q, z21.q }, z10.q, z21.q
// CHECK-ENCODING: [0x55,0xd5,0x35,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c135d555 <unknown>

uzp     {z22.q - z23.q}, z13.q, z8.q  // 11000001-00101000-11010101-10110111
// CHECK-INST: uzp     { z22.q, z23.q }, z13.q, z8.q
// CHECK-ENCODING: [0xb7,0xd5,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c128d5b7 <unknown>

uzp     {z30.q - z31.q}, z31.q, z31.q  // 11000001-00111111-11010111-11111111
// CHECK-INST: uzp     { z30.q, z31.q }, z31.q, z31.q
// CHECK-ENCODING: [0xff,0xd7,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13fd7ff <unknown>


uzp     {z0.h - z1.h}, z0.h, z0.h  // 11000001-01100000-11010000-00000001
// CHECK-INST: uzp     { z0.h, z1.h }, z0.h, z0.h
// CHECK-ENCODING: [0x01,0xd0,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c160d001 <unknown>

uzp     {z20.h - z21.h}, z10.h, z21.h  // 11000001-01110101-11010001-01010101
// CHECK-INST: uzp     { z20.h, z21.h }, z10.h, z21.h
// CHECK-ENCODING: [0x55,0xd1,0x75,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c175d155 <unknown>

uzp     {z22.h - z23.h}, z13.h, z8.h  // 11000001-01101000-11010001-10110111
// CHECK-INST: uzp     { z22.h, z23.h }, z13.h, z8.h
// CHECK-ENCODING: [0xb7,0xd1,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c168d1b7 <unknown>

uzp     {z30.h - z31.h}, z31.h, z31.h  // 11000001-01111111-11010011-11111111
// CHECK-INST: uzp     { z30.h, z31.h }, z31.h, z31.h
// CHECK-ENCODING: [0xff,0xd3,0x7f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17fd3ff <unknown>


uzp     {z0.s - z1.s}, z0.s, z0.s  // 11000001-10100000-11010000-00000001
// CHECK-INST: uzp     { z0.s, z1.s }, z0.s, z0.s
// CHECK-ENCODING: [0x01,0xd0,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a0d001 <unknown>

uzp     {z20.s - z21.s}, z10.s, z21.s  // 11000001-10110101-11010001-01010101
// CHECK-INST: uzp     { z20.s, z21.s }, z10.s, z21.s
// CHECK-ENCODING: [0x55,0xd1,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b5d155 <unknown>

uzp     {z22.s - z23.s}, z13.s, z8.s  // 11000001-10101000-11010001-10110111
// CHECK-INST: uzp     { z22.s, z23.s }, z13.s, z8.s
// CHECK-ENCODING: [0xb7,0xd1,0xa8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a8d1b7 <unknown>

uzp     {z30.s - z31.s}, z31.s, z31.s  // 11000001-10111111-11010011-11111111
// CHECK-INST: uzp     { z30.s, z31.s }, z31.s, z31.s
// CHECK-ENCODING: [0xff,0xd3,0xbf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bfd3ff <unknown>


uzp     {z0.d - z1.d}, z0.d, z0.d  // 11000001-11100000-11010000-00000001
// CHECK-INST: uzp     { z0.d, z1.d }, z0.d, z0.d
// CHECK-ENCODING: [0x01,0xd0,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e0d001 <unknown>

uzp     {z20.d - z21.d}, z10.d, z21.d  // 11000001-11110101-11010001-01010101
// CHECK-INST: uzp     { z20.d, z21.d }, z10.d, z21.d
// CHECK-ENCODING: [0x55,0xd1,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f5d155 <unknown>

uzp     {z22.d - z23.d}, z13.d, z8.d  // 11000001-11101000-11010001-10110111
// CHECK-INST: uzp     { z22.d, z23.d }, z13.d, z8.d
// CHECK-ENCODING: [0xb7,0xd1,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e8d1b7 <unknown>

uzp     {z30.d - z31.d}, z31.d, z31.d  // 11000001-11111111-11010011-11111111
// CHECK-INST: uzp     { z30.d, z31.d }, z31.d, z31.d
// CHECK-ENCODING: [0xff,0xd3,0xff,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ffd3ff <unknown>


uzp     {z0.b - z1.b}, z0.b, z0.b  // 11000001-00100000-11010000-00000001
// CHECK-INST: uzp     { z0.b, z1.b }, z0.b, z0.b
// CHECK-ENCODING: [0x01,0xd0,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c120d001 <unknown>

uzp     {z20.b - z21.b}, z10.b, z21.b  // 11000001-00110101-11010001-01010101
// CHECK-INST: uzp     { z20.b, z21.b }, z10.b, z21.b
// CHECK-ENCODING: [0x55,0xd1,0x35,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c135d155 <unknown>

uzp     {z22.b - z23.b}, z13.b, z8.b  // 11000001-00101000-11010001-10110111
// CHECK-INST: uzp     { z22.b, z23.b }, z13.b, z8.b
// CHECK-ENCODING: [0xb7,0xd1,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c128d1b7 <unknown>

uzp     {z30.b - z31.b}, z31.b, z31.b  // 11000001-00111111-11010011-11111111
// CHECK-INST: uzp     { z30.b, z31.b }, z31.b, z31.b
// CHECK-ENCODING: [0xff,0xd3,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13fd3ff <unknown>


uzp     {z0.q - z3.q}, {z0.q - z3.q}  // 11000001-00110111-11100000-00000010
// CHECK-INST: uzp     { z0.q - z3.q }, { z0.q - z3.q }
// CHECK-ENCODING: [0x02,0xe0,0x37,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c137e002 <unknown>

uzp     {z20.q - z23.q}, {z8.q - z11.q}  // 11000001-00110111-11100001-00010110
// CHECK-INST: uzp     { z20.q - z23.q }, { z8.q - z11.q }
// CHECK-ENCODING: [0x16,0xe1,0x37,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c137e116 <unknown>

uzp     {z20.q - z23.q}, {z12.q - z15.q}  // 11000001-00110111-11100001-10010110
// CHECK-INST: uzp     { z20.q - z23.q }, { z12.q - z15.q }
// CHECK-ENCODING: [0x96,0xe1,0x37,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c137e196 <unknown>

uzp     {z28.q - z31.q}, {z28.q - z31.q}  // 11000001-00110111-11100011-10011110
// CHECK-INST: uzp     { z28.q - z31.q }, { z28.q - z31.q }
// CHECK-ENCODING: [0x9e,0xe3,0x37,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c137e39e <unknown>


uzp     {z0.h - z3.h}, {z0.h - z3.h}  // 11000001-01110110-11100000-00000010
// CHECK-INST: uzp     { z0.h - z3.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x02,0xe0,0x76,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c176e002 <unknown>

uzp     {z20.h - z23.h}, {z8.h - z11.h}  // 11000001-01110110-11100001-00010110
// CHECK-INST: uzp     { z20.h - z23.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x16,0xe1,0x76,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c176e116 <unknown>

uzp     {z20.h - z23.h}, {z12.h - z15.h}  // 11000001-01110110-11100001-10010110
// CHECK-INST: uzp     { z20.h - z23.h }, { z12.h - z15.h }
// CHECK-ENCODING: [0x96,0xe1,0x76,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c176e196 <unknown>

uzp     {z28.h - z31.h}, {z28.h - z31.h}  // 11000001-01110110-11100011-10011110
// CHECK-INST: uzp     { z28.h - z31.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x9e,0xe3,0x76,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c176e39e <unknown>


uzp     {z0.s - z3.s}, {z0.s - z3.s}  // 11000001-10110110-11100000-00000010
// CHECK-INST: uzp     { z0.s - z3.s }, { z0.s - z3.s }
// CHECK-ENCODING: [0x02,0xe0,0xb6,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b6e002 <unknown>

uzp     {z20.s - z23.s}, {z8.s - z11.s}  // 11000001-10110110-11100001-00010110
// CHECK-INST: uzp     { z20.s - z23.s }, { z8.s - z11.s }
// CHECK-ENCODING: [0x16,0xe1,0xb6,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b6e116 <unknown>

uzp     {z20.s - z23.s}, {z12.s - z15.s}  // 11000001-10110110-11100001-10010110
// CHECK-INST: uzp     { z20.s - z23.s }, { z12.s - z15.s }
// CHECK-ENCODING: [0x96,0xe1,0xb6,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b6e196 <unknown>

uzp     {z28.s - z31.s}, {z28.s - z31.s}  // 11000001-10110110-11100011-10011110
// CHECK-INST: uzp     { z28.s - z31.s }, { z28.s - z31.s }
// CHECK-ENCODING: [0x9e,0xe3,0xb6,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b6e39e <unknown>


uzp     {z0.d - z3.d}, {z0.d - z3.d}  // 11000001-11110110-11100000-00000010
// CHECK-INST: uzp     { z0.d - z3.d }, { z0.d - z3.d }
// CHECK-ENCODING: [0x02,0xe0,0xf6,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f6e002 <unknown>

uzp     {z20.d - z23.d}, {z8.d - z11.d}  // 11000001-11110110-11100001-00010110
// CHECK-INST: uzp     { z20.d - z23.d }, { z8.d - z11.d }
// CHECK-ENCODING: [0x16,0xe1,0xf6,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f6e116 <unknown>

uzp     {z20.d - z23.d}, {z12.d - z15.d}  // 11000001-11110110-11100001-10010110
// CHECK-INST: uzp     { z20.d - z23.d }, { z12.d - z15.d }
// CHECK-ENCODING: [0x96,0xe1,0xf6,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f6e196 <unknown>

uzp     {z28.d - z31.d}, {z28.d - z31.d}  // 11000001-11110110-11100011-10011110
// CHECK-INST: uzp     { z28.d - z31.d }, { z28.d - z31.d }
// CHECK-ENCODING: [0x9e,0xe3,0xf6,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f6e39e <unknown>


uzp     {z0.b - z3.b}, {z0.b - z3.b}  // 11000001-00110110-11100000-00000010
// CHECK-INST: uzp     { z0.b - z3.b }, { z0.b - z3.b }
// CHECK-ENCODING: [0x02,0xe0,0x36,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c136e002 <unknown>

uzp     {z20.b - z23.b}, {z8.b - z11.b}  // 11000001-00110110-11100001-00010110
// CHECK-INST: uzp     { z20.b - z23.b }, { z8.b - z11.b }
// CHECK-ENCODING: [0x16,0xe1,0x36,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c136e116 <unknown>

uzp     {z20.b - z23.b}, {z12.b - z15.b}  // 11000001-00110110-11100001-10010110
// CHECK-INST: uzp     { z20.b - z23.b }, { z12.b - z15.b }
// CHECK-ENCODING: [0x96,0xe1,0x36,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c136e196 <unknown>

uzp     {z28.b - z31.b}, {z28.b - z31.b}  // 11000001-00110110-11100011-10011110
// CHECK-INST: uzp     { z28.b - z31.b }, { z28.b - z31.b }
// CHECK-ENCODING: [0x9e,0xe3,0x36,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c136e39e <unknown>

