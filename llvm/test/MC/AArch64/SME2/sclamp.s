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


sclamp  {z0.h, z1.h}, z0.h, z0.h  // 11000001-01100000-11000100-00000000
// CHECK-INST: sclamp  { z0.h, z1.h }, z0.h, z0.h
// CHECK-ENCODING: [0x00,0xc4,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c160c400 <unknown>

sclamp  {z20.h, z21.h}, z10.h, z21.h  // 11000001-01110101-11000101-01010100
// CHECK-INST: sclamp  { z20.h, z21.h }, z10.h, z21.h
// CHECK-ENCODING: [0x54,0xc5,0x75,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c175c554 <unknown>

sclamp  {z22.h, z23.h}, z13.h, z8.h  // 11000001-01101000-11000101-10110110
// CHECK-INST: sclamp  { z22.h, z23.h }, z13.h, z8.h
// CHECK-ENCODING: [0xb6,0xc5,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c168c5b6 <unknown>

sclamp  {z30.h, z31.h}, z31.h, z31.h  // 11000001-01111111-11000111-11111110
// CHECK-INST: sclamp  { z30.h, z31.h }, z31.h, z31.h
// CHECK-ENCODING: [0xfe,0xc7,0x7f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17fc7fe <unknown>


sclamp  {z0.s, z1.s}, z0.s, z0.s  // 11000001-10100000-11000100-00000000
// CHECK-INST: sclamp  { z0.s, z1.s }, z0.s, z0.s
// CHECK-ENCODING: [0x00,0xc4,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a0c400 <unknown>

sclamp  {z20.s, z21.s}, z10.s, z21.s  // 11000001-10110101-11000101-01010100
// CHECK-INST: sclamp  { z20.s, z21.s }, z10.s, z21.s
// CHECK-ENCODING: [0x54,0xc5,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b5c554 <unknown>

sclamp  {z22.s, z23.s}, z13.s, z8.s  // 11000001-10101000-11000101-10110110
// CHECK-INST: sclamp  { z22.s, z23.s }, z13.s, z8.s
// CHECK-ENCODING: [0xb6,0xc5,0xa8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a8c5b6 <unknown>

sclamp  {z30.s, z31.s}, z31.s, z31.s  // 11000001-10111111-11000111-11111110
// CHECK-INST: sclamp  { z30.s, z31.s }, z31.s, z31.s
// CHECK-ENCODING: [0xfe,0xc7,0xbf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bfc7fe <unknown>


sclamp  {z0.d, z1.d}, z0.d, z0.d  // 11000001-11100000-11000100-00000000
// CHECK-INST: sclamp  { z0.d, z1.d }, z0.d, z0.d
// CHECK-ENCODING: [0x00,0xc4,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e0c400 <unknown>

sclamp  {z20.d, z21.d}, z10.d, z21.d  // 11000001-11110101-11000101-01010100
// CHECK-INST: sclamp  { z20.d, z21.d }, z10.d, z21.d
// CHECK-ENCODING: [0x54,0xc5,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f5c554 <unknown>

sclamp  {z22.d, z23.d}, z13.d, z8.d  // 11000001-11101000-11000101-10110110
// CHECK-INST: sclamp  { z22.d, z23.d }, z13.d, z8.d
// CHECK-ENCODING: [0xb6,0xc5,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e8c5b6 <unknown>

sclamp  {z30.d, z31.d}, z31.d, z31.d  // 11000001-11111111-11000111-11111110
// CHECK-INST: sclamp  { z30.d, z31.d }, z31.d, z31.d
// CHECK-ENCODING: [0xfe,0xc7,0xff,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ffc7fe <unknown>


sclamp  {z0.b, z1.b}, z0.b, z0.b  // 11000001-00100000-11000100-00000000
// CHECK-INST: sclamp  { z0.b, z1.b }, z0.b, z0.b
// CHECK-ENCODING: [0x00,0xc4,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c120c400 <unknown>

sclamp  {z20.b, z21.b}, z10.b, z21.b  // 11000001-00110101-11000101-01010100
// CHECK-INST: sclamp  { z20.b, z21.b }, z10.b, z21.b
// CHECK-ENCODING: [0x54,0xc5,0x35,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c135c554 <unknown>

sclamp  {z22.b, z23.b}, z13.b, z8.b  // 11000001-00101000-11000101-10110110
// CHECK-INST: sclamp  { z22.b, z23.b }, z13.b, z8.b
// CHECK-ENCODING: [0xb6,0xc5,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c128c5b6 <unknown>

sclamp  {z30.b, z31.b}, z31.b, z31.b  // 11000001-00111111-11000111-11111110
// CHECK-INST: sclamp  { z30.b, z31.b }, z31.b, z31.b
// CHECK-ENCODING: [0xfe,0xc7,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13fc7fe <unknown>


sclamp  {z0.h - z3.h}, z0.h, z0.h  // 11000001-01100000-11001100-00000000
// CHECK-INST: sclamp  { z0.h - z3.h }, z0.h, z0.h
// CHECK-ENCODING: [0x00,0xcc,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c160cc00 <unknown>

sclamp  {z20.h - z23.h}, z10.h, z21.h  // 11000001-01110101-11001101-01010100
// CHECK-INST: sclamp  { z20.h - z23.h }, z10.h, z21.h
// CHECK-ENCODING: [0x54,0xcd,0x75,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c175cd54 <unknown>

sclamp  {z20.h - z23.h}, z13.h, z8.h  // 11000001-01101000-11001101-10110100
// CHECK-INST: sclamp  { z20.h - z23.h }, z13.h, z8.h
// CHECK-ENCODING: [0xb4,0xcd,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c168cdb4 <unknown>

sclamp  {z28.h - z31.h}, z31.h, z31.h  // 11000001-01111111-11001111-11111100
// CHECK-INST: sclamp  { z28.h - z31.h }, z31.h, z31.h
// CHECK-ENCODING: [0xfc,0xcf,0x7f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17fcffc <unknown>


sclamp  {z0.s - z3.s}, z0.s, z0.s  // 11000001-10100000-11001100-00000000
// CHECK-INST: sclamp  { z0.s - z3.s }, z0.s, z0.s
// CHECK-ENCODING: [0x00,0xcc,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a0cc00 <unknown>

sclamp  {z20.s - z23.s}, z10.s, z21.s  // 11000001-10110101-11001101-01010100
// CHECK-INST: sclamp  { z20.s - z23.s }, z10.s, z21.s
// CHECK-ENCODING: [0x54,0xcd,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b5cd54 <unknown>

sclamp  {z20.s - z23.s}, z13.s, z8.s  // 11000001-10101000-11001101-10110100
// CHECK-INST: sclamp  { z20.s - z23.s }, z13.s, z8.s
// CHECK-ENCODING: [0xb4,0xcd,0xa8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a8cdb4 <unknown>

sclamp  {z28.s - z31.s}, z31.s, z31.s  // 11000001-10111111-11001111-11111100
// CHECK-INST: sclamp  { z28.s - z31.s }, z31.s, z31.s
// CHECK-ENCODING: [0xfc,0xcf,0xbf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bfcffc <unknown>


sclamp  {z0.d - z3.d}, z0.d, z0.d  // 11000001-11100000-11001100-00000000
// CHECK-INST: sclamp  { z0.d - z3.d }, z0.d, z0.d
// CHECK-ENCODING: [0x00,0xcc,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e0cc00 <unknown>

sclamp  {z20.d - z23.d}, z10.d, z21.d  // 11000001-11110101-11001101-01010100
// CHECK-INST: sclamp  { z20.d - z23.d }, z10.d, z21.d
// CHECK-ENCODING: [0x54,0xcd,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f5cd54 <unknown>

sclamp  {z20.d - z23.d}, z13.d, z8.d  // 11000001-11101000-11001101-10110100
// CHECK-INST: sclamp  { z20.d - z23.d }, z13.d, z8.d
// CHECK-ENCODING: [0xb4,0xcd,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e8cdb4 <unknown>

sclamp  {z28.d - z31.d}, z31.d, z31.d  // 11000001-11111111-11001111-11111100
// CHECK-INST: sclamp  { z28.d - z31.d }, z31.d, z31.d
// CHECK-ENCODING: [0xfc,0xcf,0xff,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ffcffc <unknown>


sclamp  {z0.b - z3.b}, z0.b, z0.b  // 11000001-00100000-11001100-00000000
// CHECK-INST: sclamp  { z0.b - z3.b }, z0.b, z0.b
// CHECK-ENCODING: [0x00,0xcc,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c120cc00 <unknown>

sclamp  {z20.b - z23.b}, z10.b, z21.b  // 11000001-00110101-11001101-01010100
// CHECK-INST: sclamp  { z20.b - z23.b }, z10.b, z21.b
// CHECK-ENCODING: [0x54,0xcd,0x35,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c135cd54 <unknown>

sclamp  {z20.b - z23.b}, z13.b, z8.b  // 11000001-00101000-11001101-10110100
// CHECK-INST: sclamp  { z20.b - z23.b }, z13.b, z8.b
// CHECK-ENCODING: [0xb4,0xcd,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c128cdb4 <unknown>

sclamp  {z28.b - z31.b}, z31.b, z31.b  // 11000001-00111111-11001111-11111100
// CHECK-INST: sclamp  { z28.b - z31.b }, z31.b, z31.b
// CHECK-ENCODING: [0xfc,0xcf,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13fcffc <unknown>

