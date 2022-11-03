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


fclamp  {z0.d, z1.d}, z0.d, z0.d  // 11000001-11100000-11000000-00000000
// CHECK-INST: fclamp  { z0.d, z1.d }, z0.d, z0.d
// CHECK-ENCODING: [0x00,0xc0,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e0c000 <unknown>

fclamp  {z20.d, z21.d}, z10.d, z21.d  // 11000001-11110101-11000001-01010100
// CHECK-INST: fclamp  { z20.d, z21.d }, z10.d, z21.d
// CHECK-ENCODING: [0x54,0xc1,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f5c154 <unknown>

fclamp  {z22.d, z23.d}, z13.d, z8.d  // 11000001-11101000-11000001-10110110
// CHECK-INST: fclamp  { z22.d, z23.d }, z13.d, z8.d
// CHECK-ENCODING: [0xb6,0xc1,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e8c1b6 <unknown>

fclamp  {z30.d, z31.d}, z31.d, z31.d  // 11000001-11111111-11000011-11111110
// CHECK-INST: fclamp  { z30.d, z31.d }, z31.d, z31.d
// CHECK-ENCODING: [0xfe,0xc3,0xff,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ffc3fe <unknown>


fclamp  {z0.h, z1.h}, z0.h, z0.h  // 11000001-01100000-11000000-00000000
// CHECK-INST: fclamp  { z0.h, z1.h }, z0.h, z0.h
// CHECK-ENCODING: [0x00,0xc0,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c160c000 <unknown>

fclamp  {z20.h, z21.h}, z10.h, z21.h  // 11000001-01110101-11000001-01010100
// CHECK-INST: fclamp  { z20.h, z21.h }, z10.h, z21.h
// CHECK-ENCODING: [0x54,0xc1,0x75,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c175c154 <unknown>

fclamp  {z22.h, z23.h}, z13.h, z8.h  // 11000001-01101000-11000001-10110110
// CHECK-INST: fclamp  { z22.h, z23.h }, z13.h, z8.h
// CHECK-ENCODING: [0xb6,0xc1,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c168c1b6 <unknown>

fclamp  {z30.h, z31.h}, z31.h, z31.h  // 11000001-01111111-11000011-11111110
// CHECK-INST: fclamp  { z30.h, z31.h }, z31.h, z31.h
// CHECK-ENCODING: [0xfe,0xc3,0x7f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17fc3fe <unknown>


fclamp  {z0.s, z1.s}, z0.s, z0.s  // 11000001-10100000-11000000-00000000
// CHECK-INST: fclamp  { z0.s, z1.s }, z0.s, z0.s
// CHECK-ENCODING: [0x00,0xc0,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a0c000 <unknown>

fclamp  {z20.s, z21.s}, z10.s, z21.s  // 11000001-10110101-11000001-01010100
// CHECK-INST: fclamp  { z20.s, z21.s }, z10.s, z21.s
// CHECK-ENCODING: [0x54,0xc1,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b5c154 <unknown>

fclamp  {z22.s, z23.s}, z13.s, z8.s  // 11000001-10101000-11000001-10110110
// CHECK-INST: fclamp  { z22.s, z23.s }, z13.s, z8.s
// CHECK-ENCODING: [0xb6,0xc1,0xa8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a8c1b6 <unknown>

fclamp  {z30.s, z31.s}, z31.s, z31.s  // 11000001-10111111-11000011-11111110
// CHECK-INST: fclamp  { z30.s, z31.s }, z31.s, z31.s
// CHECK-ENCODING: [0xfe,0xc3,0xbf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bfc3fe <unknown>


fclamp  {z0.d - z3.d}, z0.d, z0.d  // 11000001-11100000-11001000-00000000
// CHECK-INST: fclamp  { z0.d - z3.d }, z0.d, z0.d
// CHECK-ENCODING: [0x00,0xc8,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e0c800 <unknown>

fclamp  {z20.d - z23.d}, z10.d, z21.d  // 11000001-11110101-11001001-01010100
// CHECK-INST: fclamp  { z20.d - z23.d }, z10.d, z21.d
// CHECK-ENCODING: [0x54,0xc9,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f5c954 <unknown>

fclamp  {z20.d - z23.d}, z13.d, z8.d  // 11000001-11101000-11001001-10110100
// CHECK-INST: fclamp  { z20.d - z23.d }, z13.d, z8.d
// CHECK-ENCODING: [0xb4,0xc9,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e8c9b4 <unknown>

fclamp  {z28.d - z31.d}, z31.d, z31.d  // 11000001-11111111-11001011-11111100
// CHECK-INST: fclamp  { z28.d - z31.d }, z31.d, z31.d
// CHECK-ENCODING: [0xfc,0xcb,0xff,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ffcbfc <unknown>


fclamp  {z0.h - z3.h}, z0.h, z0.h  // 11000001-01100000-11001000-00000000
// CHECK-INST: fclamp  { z0.h - z3.h }, z0.h, z0.h
// CHECK-ENCODING: [0x00,0xc8,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c160c800 <unknown>

fclamp  {z20.h - z23.h}, z10.h, z21.h  // 11000001-01110101-11001001-01010100
// CHECK-INST: fclamp  { z20.h - z23.h }, z10.h, z21.h
// CHECK-ENCODING: [0x54,0xc9,0x75,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c175c954 <unknown>

fclamp  {z20.h - z23.h}, z13.h, z8.h  // 11000001-01101000-11001001-10110100
// CHECK-INST: fclamp  { z20.h - z23.h }, z13.h, z8.h
// CHECK-ENCODING: [0xb4,0xc9,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c168c9b4 <unknown>

fclamp  {z28.h - z31.h}, z31.h, z31.h  // 11000001-01111111-11001011-11111100
// CHECK-INST: fclamp  { z28.h - z31.h }, z31.h, z31.h
// CHECK-ENCODING: [0xfc,0xcb,0x7f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17fcbfc <unknown>


fclamp  {z0.s - z3.s}, z0.s, z0.s  // 11000001-10100000-11001000-00000000
// CHECK-INST: fclamp  { z0.s - z3.s }, z0.s, z0.s
// CHECK-ENCODING: [0x00,0xc8,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a0c800 <unknown>

fclamp  {z20.s - z23.s}, z10.s, z21.s  // 11000001-10110101-11001001-01010100
// CHECK-INST: fclamp  { z20.s - z23.s }, z10.s, z21.s
// CHECK-ENCODING: [0x54,0xc9,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b5c954 <unknown>

fclamp  {z20.s - z23.s}, z13.s, z8.s  // 11000001-10101000-11001001-10110100
// CHECK-INST: fclamp  { z20.s - z23.s }, z13.s, z8.s
// CHECK-ENCODING: [0xb4,0xc9,0xa8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a8c9b4 <unknown>

fclamp  {z28.s - z31.s}, z31.s, z31.s  // 11000001-10111111-11001011-11111100
// CHECK-INST: fclamp  { z28.s - z31.s }, z31.s, z31.s
// CHECK-ENCODING: [0xfc,0xcb,0xbf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bfcbfc <unknown>

