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


umops   za0.s, p0/m, p0/m, z0.h, z0.h  // 10100001-10000000-00000000-00011000
// CHECK-INST: umops   za0.s, p0/m, p0/m, z0.h, z0.h
// CHECK-ENCODING: [0x18,0x00,0x80,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1800018 <unknown>

umops   za1.s, p5/m, p2/m, z10.h, z21.h  // 10100001-10010101-01010101-01011001
// CHECK-INST: umops   za1.s, p5/m, p2/m, z10.h, z21.h
// CHECK-ENCODING: [0x59,0x55,0x95,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1955559 <unknown>

umops   za3.s, p3/m, p7/m, z13.h, z8.h  // 10100001-10001000-11101101-10111011
// CHECK-INST: umops   za3.s, p3/m, p7/m, z13.h, z8.h
// CHECK-ENCODING: [0xbb,0xed,0x88,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a188edbb <unknown>

umops   za3.s, p7/m, p7/m, z31.h, z31.h  // 10100001-10011111-11111111-11111011
// CHECK-INST: umops   za3.s, p7/m, p7/m, z31.h, z31.h
// CHECK-ENCODING: [0xfb,0xff,0x9f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a19ffffb <unknown>

umops   za1.s, p3/m, p0/m, z17.h, z16.h  // 10100001-10010000-00001110-00111001
// CHECK-INST: umops   za1.s, p3/m, p0/m, z17.h, z16.h
// CHECK-ENCODING: [0x39,0x0e,0x90,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1900e39 <unknown>

umops   za1.s, p1/m, p4/m, z1.h, z30.h  // 10100001-10011110-10000100-00111001
// CHECK-INST: umops   za1.s, p1/m, p4/m, z1.h, z30.h
// CHECK-ENCODING: [0x39,0x84,0x9e,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a19e8439 <unknown>

umops   za0.s, p5/m, p2/m, z19.h, z20.h  // 10100001-10010100-01010110-01111000
// CHECK-INST: umops   za0.s, p5/m, p2/m, z19.h, z20.h
// CHECK-ENCODING: [0x78,0x56,0x94,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1945678 <unknown>

umops   za0.s, p6/m, p0/m, z12.h, z2.h  // 10100001-10000010-00011001-10011000
// CHECK-INST: umops   za0.s, p6/m, p0/m, z12.h, z2.h
// CHECK-ENCODING: [0x98,0x19,0x82,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1821998 <unknown>

umops   za1.s, p2/m, p6/m, z1.h, z26.h  // 10100001-10011010-11001000-00111001
// CHECK-INST: umops   za1.s, p2/m, p6/m, z1.h, z26.h
// CHECK-ENCODING: [0x39,0xc8,0x9a,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a19ac839 <unknown>

umops   za1.s, p2/m, p0/m, z22.h, z30.h  // 10100001-10011110-00001010-11011001
// CHECK-INST: umops   za1.s, p2/m, p0/m, z22.h, z30.h
// CHECK-ENCODING: [0xd9,0x0a,0x9e,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a19e0ad9 <unknown>

umops   za2.s, p5/m, p7/m, z9.h, z1.h  // 10100001-10000001-11110101-00111010
// CHECK-INST: umops   za2.s, p5/m, p7/m, z9.h, z1.h
// CHECK-ENCODING: [0x3a,0xf5,0x81,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a181f53a <unknown>

umops   za3.s, p2/m, p5/m, z12.h, z11.h  // 10100001-10001011-10101001-10011011
// CHECK-INST: umops   za3.s, p2/m, p5/m, z12.h, z11.h
// CHECK-ENCODING: [0x9b,0xa9,0x8b,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a18ba99b <unknown>

