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


smops   za0.s, p0/m, p0/m, z0.h, z0.h  // 10100000-10000000-00000000-00011000
// CHECK-INST: smops   za0.s, p0/m, p0/m, z0.h, z0.h
// CHECK-ENCODING: [0x18,0x00,0x80,0xa0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a0800018 <unknown>

smops   za1.s, p5/m, p2/m, z10.h, z21.h  // 10100000-10010101-01010101-01011001
// CHECK-INST: smops   za1.s, p5/m, p2/m, z10.h, z21.h
// CHECK-ENCODING: [0x59,0x55,0x95,0xa0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a0955559 <unknown>

smops   za3.s, p3/m, p7/m, z13.h, z8.h  // 10100000-10001000-11101101-10111011
// CHECK-INST: smops   za3.s, p3/m, p7/m, z13.h, z8.h
// CHECK-ENCODING: [0xbb,0xed,0x88,0xa0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a088edbb <unknown>

smops   za3.s, p7/m, p7/m, z31.h, z31.h  // 10100000-10011111-11111111-11111011
// CHECK-INST: smops   za3.s, p7/m, p7/m, z31.h, z31.h
// CHECK-ENCODING: [0xfb,0xff,0x9f,0xa0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a09ffffb <unknown>

smops   za1.s, p3/m, p0/m, z17.h, z16.h  // 10100000-10010000-00001110-00111001
// CHECK-INST: smops   za1.s, p3/m, p0/m, z17.h, z16.h
// CHECK-ENCODING: [0x39,0x0e,0x90,0xa0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a0900e39 <unknown>

smops   za1.s, p1/m, p4/m, z1.h, z30.h  // 10100000-10011110-10000100-00111001
// CHECK-INST: smops   za1.s, p1/m, p4/m, z1.h, z30.h
// CHECK-ENCODING: [0x39,0x84,0x9e,0xa0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a09e8439 <unknown>

smops   za0.s, p5/m, p2/m, z19.h, z20.h  // 10100000-10010100-01010110-01111000
// CHECK-INST: smops   za0.s, p5/m, p2/m, z19.h, z20.h
// CHECK-ENCODING: [0x78,0x56,0x94,0xa0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a0945678 <unknown>

smops   za0.s, p6/m, p0/m, z12.h, z2.h  // 10100000-10000010-00011001-10011000
// CHECK-INST: smops   za0.s, p6/m, p0/m, z12.h, z2.h
// CHECK-ENCODING: [0x98,0x19,0x82,0xa0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a0821998 <unknown>

smops   za1.s, p2/m, p6/m, z1.h, z26.h  // 10100000-10011010-11001000-00111001
// CHECK-INST: smops   za1.s, p2/m, p6/m, z1.h, z26.h
// CHECK-ENCODING: [0x39,0xc8,0x9a,0xa0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a09ac839 <unknown>

smops   za1.s, p2/m, p0/m, z22.h, z30.h  // 10100000-10011110-00001010-11011001
// CHECK-INST: smops   za1.s, p2/m, p0/m, z22.h, z30.h
// CHECK-ENCODING: [0xd9,0x0a,0x9e,0xa0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a09e0ad9 <unknown>

smops   za2.s, p5/m, p7/m, z9.h, z1.h  // 10100000-10000001-11110101-00111010
// CHECK-INST: smops   za2.s, p5/m, p7/m, z9.h, z1.h
// CHECK-ENCODING: [0x3a,0xf5,0x81,0xa0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a081f53a <unknown>

smops   za3.s, p2/m, p5/m, z12.h, z11.h  // 10100000-10001011-10101001-10011011
// CHECK-INST: smops   za3.s, p2/m, p5/m, z12.h, z11.h
// CHECK-ENCODING: [0x9b,0xa9,0x8b,0xa0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a08ba99b <unknown>

