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


fminnm  {z0.d, z1.d}, {z0.d, z1.d}, z0.d  // 11000001-11100000-10100001-00100001
// CHECK-INST: fminnm  { z0.d, z1.d }, { z0.d, z1.d }, z0.d
// CHECK-ENCODING: [0x21,0xa1,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e0a121 <unknown>

fminnm  {z20.d, z21.d}, {z20.d, z21.d}, z5.d  // 11000001-11100101-10100001-00110101
// CHECK-INST: fminnm  { z20.d, z21.d }, { z20.d, z21.d }, z5.d
// CHECK-ENCODING: [0x35,0xa1,0xe5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e5a135 <unknown>

fminnm  {z22.d, z23.d}, {z22.d, z23.d}, z8.d  // 11000001-11101000-10100001-00110111
// CHECK-INST: fminnm  { z22.d, z23.d }, { z22.d, z23.d }, z8.d
// CHECK-ENCODING: [0x37,0xa1,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e8a137 <unknown>

fminnm  {z30.d, z31.d}, {z30.d, z31.d}, z15.d  // 11000001-11101111-10100001-00111111
// CHECK-INST: fminnm  { z30.d, z31.d }, { z30.d, z31.d }, z15.d
// CHECK-ENCODING: [0x3f,0xa1,0xef,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1efa13f <unknown>


fminnm  {z0.d, z1.d}, {z0.d, z1.d}, {z0.d, z1.d}  // 11000001-11100000-10110001-00100001
// CHECK-INST: fminnm  { z0.d, z1.d }, { z0.d, z1.d }, { z0.d, z1.d }
// CHECK-ENCODING: [0x21,0xb1,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e0b121 <unknown>

fminnm  {z20.d, z21.d}, {z20.d, z21.d}, {z20.d, z21.d}  // 11000001-11110100-10110001-00110101
// CHECK-INST: fminnm  { z20.d, z21.d }, { z20.d, z21.d }, { z20.d, z21.d }
// CHECK-ENCODING: [0x35,0xb1,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f4b135 <unknown>

fminnm  {z22.d, z23.d}, {z22.d, z23.d}, {z8.d, z9.d}  // 11000001-11101000-10110001-00110111
// CHECK-INST: fminnm  { z22.d, z23.d }, { z22.d, z23.d }, { z8.d, z9.d }
// CHECK-ENCODING: [0x37,0xb1,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e8b137 <unknown>

fminnm  {z30.d, z31.d}, {z30.d, z31.d}, {z30.d, z31.d}  // 11000001-11111110-10110001-00111111
// CHECK-INST: fminnm  { z30.d, z31.d }, { z30.d, z31.d }, { z30.d, z31.d }
// CHECK-ENCODING: [0x3f,0xb1,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1feb13f <unknown>


fminnm  {z0.h, z1.h}, {z0.h, z1.h}, z0.h  // 11000001-01100000-10100001-00100001
// CHECK-INST: fminnm  { z0.h, z1.h }, { z0.h, z1.h }, z0.h
// CHECK-ENCODING: [0x21,0xa1,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c160a121 <unknown>

fminnm  {z20.h, z21.h}, {z20.h, z21.h}, z5.h  // 11000001-01100101-10100001-00110101
// CHECK-INST: fminnm  { z20.h, z21.h }, { z20.h, z21.h }, z5.h
// CHECK-ENCODING: [0x35,0xa1,0x65,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c165a135 <unknown>

fminnm  {z22.h, z23.h}, {z22.h, z23.h}, z8.h  // 11000001-01101000-10100001-00110111
// CHECK-INST: fminnm  { z22.h, z23.h }, { z22.h, z23.h }, z8.h
// CHECK-ENCODING: [0x37,0xa1,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c168a137 <unknown>

fminnm  {z30.h, z31.h}, {z30.h, z31.h}, z15.h  // 11000001-01101111-10100001-00111111
// CHECK-INST: fminnm  { z30.h, z31.h }, { z30.h, z31.h }, z15.h
// CHECK-ENCODING: [0x3f,0xa1,0x6f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16fa13f <unknown>


fminnm  {z0.h, z1.h}, {z0.h, z1.h}, {z0.h, z1.h}  // 11000001-01100000-10110001-00100001
// CHECK-INST: fminnm  { z0.h, z1.h }, { z0.h, z1.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x21,0xb1,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c160b121 <unknown>

fminnm  {z20.h, z21.h}, {z20.h, z21.h}, {z20.h, z21.h}  // 11000001-01110100-10110001-00110101
// CHECK-INST: fminnm  { z20.h, z21.h }, { z20.h, z21.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x35,0xb1,0x74,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c174b135 <unknown>

fminnm  {z22.h, z23.h}, {z22.h, z23.h}, {z8.h, z9.h}  // 11000001-01101000-10110001-00110111
// CHECK-INST: fminnm  { z22.h, z23.h }, { z22.h, z23.h }, { z8.h, z9.h }
// CHECK-ENCODING: [0x37,0xb1,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c168b137 <unknown>

fminnm  {z30.h, z31.h}, {z30.h, z31.h}, {z30.h, z31.h}  // 11000001-01111110-10110001-00111111
// CHECK-INST: fminnm  { z30.h, z31.h }, { z30.h, z31.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0x3f,0xb1,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17eb13f <unknown>


fminnm  {z0.s, z1.s}, {z0.s, z1.s}, z0.s  // 11000001-10100000-10100001-00100001
// CHECK-INST: fminnm  { z0.s, z1.s }, { z0.s, z1.s }, z0.s
// CHECK-ENCODING: [0x21,0xa1,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a0a121 <unknown>

fminnm  {z20.s, z21.s}, {z20.s, z21.s}, z5.s  // 11000001-10100101-10100001-00110101
// CHECK-INST: fminnm  { z20.s, z21.s }, { z20.s, z21.s }, z5.s
// CHECK-ENCODING: [0x35,0xa1,0xa5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a5a135 <unknown>

fminnm  {z22.s, z23.s}, {z22.s, z23.s}, z8.s  // 11000001-10101000-10100001-00110111
// CHECK-INST: fminnm  { z22.s, z23.s }, { z22.s, z23.s }, z8.s
// CHECK-ENCODING: [0x37,0xa1,0xa8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a8a137 <unknown>

fminnm  {z30.s, z31.s}, {z30.s, z31.s}, z15.s  // 11000001-10101111-10100001-00111111
// CHECK-INST: fminnm  { z30.s, z31.s }, { z30.s, z31.s }, z15.s
// CHECK-ENCODING: [0x3f,0xa1,0xaf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1afa13f <unknown>


fminnm  {z0.s, z1.s}, {z0.s, z1.s}, {z0.s, z1.s}  // 11000001-10100000-10110001-00100001
// CHECK-INST: fminnm  { z0.s, z1.s }, { z0.s, z1.s }, { z0.s, z1.s }
// CHECK-ENCODING: [0x21,0xb1,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a0b121 <unknown>

fminnm  {z20.s, z21.s}, {z20.s, z21.s}, {z20.s, z21.s}  // 11000001-10110100-10110001-00110101
// CHECK-INST: fminnm  { z20.s, z21.s }, { z20.s, z21.s }, { z20.s, z21.s }
// CHECK-ENCODING: [0x35,0xb1,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b4b135 <unknown>

fminnm  {z22.s, z23.s}, {z22.s, z23.s}, {z8.s, z9.s}  // 11000001-10101000-10110001-00110111
// CHECK-INST: fminnm  { z22.s, z23.s }, { z22.s, z23.s }, { z8.s, z9.s }
// CHECK-ENCODING: [0x37,0xb1,0xa8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a8b137 <unknown>

fminnm  {z30.s, z31.s}, {z30.s, z31.s}, {z30.s, z31.s}  // 11000001-10111110-10110001-00111111
// CHECK-INST: fminnm  { z30.s, z31.s }, { z30.s, z31.s }, { z30.s, z31.s }
// CHECK-ENCODING: [0x3f,0xb1,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1beb13f <unknown>


fminnm  {z0.d - z3.d}, {z0.d - z3.d}, z0.d  // 11000001-11100000-10101001-00100001
// CHECK-INST: fminnm  { z0.d - z3.d }, { z0.d - z3.d }, z0.d
// CHECK-ENCODING: [0x21,0xa9,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e0a921 <unknown>

fminnm  {z20.d - z23.d}, {z20.d - z23.d}, z5.d  // 11000001-11100101-10101001-00110101
// CHECK-INST: fminnm  { z20.d - z23.d }, { z20.d - z23.d }, z5.d
// CHECK-ENCODING: [0x35,0xa9,0xe5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e5a935 <unknown>

fminnm  {z20.d - z23.d}, {z20.d - z23.d}, z8.d  // 11000001-11101000-10101001-00110101
// CHECK-INST: fminnm  { z20.d - z23.d }, { z20.d - z23.d }, z8.d
// CHECK-ENCODING: [0x35,0xa9,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e8a935 <unknown>

fminnm  {z28.d - z31.d}, {z28.d - z31.d}, z15.d  // 11000001-11101111-10101001-00111101
// CHECK-INST: fminnm  { z28.d - z31.d }, { z28.d - z31.d }, z15.d
// CHECK-ENCODING: [0x3d,0xa9,0xef,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1efa93d <unknown>


fminnm  {z0.d - z3.d}, {z0.d - z3.d}, {z0.d - z3.d}  // 11000001-11100000-10111001-00100001
// CHECK-INST: fminnm  { z0.d - z3.d }, { z0.d - z3.d }, { z0.d - z3.d }
// CHECK-ENCODING: [0x21,0xb9,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e0b921 <unknown>

fminnm  {z20.d - z23.d}, {z20.d - z23.d}, {z20.d - z23.d}  // 11000001-11110100-10111001-00110101
// CHECK-INST: fminnm  { z20.d - z23.d }, { z20.d - z23.d }, { z20.d - z23.d }
// CHECK-ENCODING: [0x35,0xb9,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f4b935 <unknown>

fminnm  {z20.d - z23.d}, {z20.d - z23.d}, {z8.d - z11.d}  // 11000001-11101000-10111001-00110101
// CHECK-INST: fminnm  { z20.d - z23.d }, { z20.d - z23.d }, { z8.d - z11.d }
// CHECK-ENCODING: [0x35,0xb9,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e8b935 <unknown>

fminnm  {z28.d - z31.d}, {z28.d - z31.d}, {z28.d - z31.d}  // 11000001-11111100-10111001-00111101
// CHECK-INST: fminnm  { z28.d - z31.d }, { z28.d - z31.d }, { z28.d - z31.d }
// CHECK-ENCODING: [0x3d,0xb9,0xfc,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fcb93d <unknown>


fminnm  {z0.h - z3.h}, {z0.h - z3.h}, z0.h  // 11000001-01100000-10101001-00100001
// CHECK-INST: fminnm  { z0.h - z3.h }, { z0.h - z3.h }, z0.h
// CHECK-ENCODING: [0x21,0xa9,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c160a921 <unknown>

fminnm  {z20.h - z23.h}, {z20.h - z23.h}, z5.h  // 11000001-01100101-10101001-00110101
// CHECK-INST: fminnm  { z20.h - z23.h }, { z20.h - z23.h }, z5.h
// CHECK-ENCODING: [0x35,0xa9,0x65,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c165a935 <unknown>

fminnm  {z20.h - z23.h}, {z20.h - z23.h}, z8.h  // 11000001-01101000-10101001-00110101
// CHECK-INST: fminnm  { z20.h - z23.h }, { z20.h - z23.h }, z8.h
// CHECK-ENCODING: [0x35,0xa9,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c168a935 <unknown>

fminnm  {z28.h - z31.h}, {z28.h - z31.h}, z15.h  // 11000001-01101111-10101001-00111101
// CHECK-INST: fminnm  { z28.h - z31.h }, { z28.h - z31.h }, z15.h
// CHECK-ENCODING: [0x3d,0xa9,0x6f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16fa93d <unknown>


fminnm  {z0.h - z3.h}, {z0.h - z3.h}, {z0.h - z3.h}  // 11000001-01100000-10111001-00100001
// CHECK-INST: fminnm  { z0.h - z3.h }, { z0.h - z3.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x21,0xb9,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c160b921 <unknown>

fminnm  {z20.h - z23.h}, {z20.h - z23.h}, {z20.h - z23.h}  // 11000001-01110100-10111001-00110101
// CHECK-INST: fminnm  { z20.h - z23.h }, { z20.h - z23.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x35,0xb9,0x74,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c174b935 <unknown>

fminnm  {z20.h - z23.h}, {z20.h - z23.h}, {z8.h - z11.h}  // 11000001-01101000-10111001-00110101
// CHECK-INST: fminnm  { z20.h - z23.h }, { z20.h - z23.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x35,0xb9,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c168b935 <unknown>

fminnm  {z28.h - z31.h}, {z28.h - z31.h}, {z28.h - z31.h}  // 11000001-01111100-10111001-00111101
// CHECK-INST: fminnm  { z28.h - z31.h }, { z28.h - z31.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x3d,0xb9,0x7c,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17cb93d <unknown>


fminnm  {z0.s - z3.s}, {z0.s - z3.s}, z0.s  // 11000001-10100000-10101001-00100001
// CHECK-INST: fminnm  { z0.s - z3.s }, { z0.s - z3.s }, z0.s
// CHECK-ENCODING: [0x21,0xa9,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a0a921 <unknown>

fminnm  {z20.s - z23.s}, {z20.s - z23.s}, z5.s  // 11000001-10100101-10101001-00110101
// CHECK-INST: fminnm  { z20.s - z23.s }, { z20.s - z23.s }, z5.s
// CHECK-ENCODING: [0x35,0xa9,0xa5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a5a935 <unknown>

fminnm  {z20.s - z23.s}, {z20.s - z23.s}, z8.s  // 11000001-10101000-10101001-00110101
// CHECK-INST: fminnm  { z20.s - z23.s }, { z20.s - z23.s }, z8.s
// CHECK-ENCODING: [0x35,0xa9,0xa8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a8a935 <unknown>

fminnm  {z28.s - z31.s}, {z28.s - z31.s}, z15.s  // 11000001-10101111-10101001-00111101
// CHECK-INST: fminnm  { z28.s - z31.s }, { z28.s - z31.s }, z15.s
// CHECK-ENCODING: [0x3d,0xa9,0xaf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1afa93d <unknown>


fminnm  {z0.s - z3.s}, {z0.s - z3.s}, {z0.s - z3.s}  // 11000001-10100000-10111001-00100001
// CHECK-INST: fminnm  { z0.s - z3.s }, { z0.s - z3.s }, { z0.s - z3.s }
// CHECK-ENCODING: [0x21,0xb9,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a0b921 <unknown>

fminnm  {z20.s - z23.s}, {z20.s - z23.s}, {z20.s - z23.s}  // 11000001-10110100-10111001-00110101
// CHECK-INST: fminnm  { z20.s - z23.s }, { z20.s - z23.s }, { z20.s - z23.s }
// CHECK-ENCODING: [0x35,0xb9,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b4b935 <unknown>

fminnm  {z20.s - z23.s}, {z20.s - z23.s}, {z8.s - z11.s}  // 11000001-10101000-10111001-00110101
// CHECK-INST: fminnm  { z20.s - z23.s }, { z20.s - z23.s }, { z8.s - z11.s }
// CHECK-ENCODING: [0x35,0xb9,0xa8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a8b935 <unknown>

fminnm  {z28.s - z31.s}, {z28.s - z31.s}, {z28.s - z31.s}  // 11000001-10111100-10111001-00111101
// CHECK-INST: fminnm  { z28.s - z31.s }, { z28.s - z31.s }, { z28.s - z31.s }
// CHECK-ENCODING: [0x3d,0xb9,0xbc,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bcb93d <unknown>

