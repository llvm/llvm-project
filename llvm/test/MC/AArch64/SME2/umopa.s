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


umopa   za0.s, p0/m, p0/m, z0.h, z0.h  // 10100001-10000000-00000000-00001000
// CHECK-INST: umopa   za0.s, p0/m, p0/m, z0.h, z0.h
// CHECK-ENCODING: [0x08,0x00,0x80,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1800008 <unknown>

umopa   za1.s, p5/m, p2/m, z10.h, z21.h  // 10100001-10010101-01010101-01001001
// CHECK-INST: umopa   za1.s, p5/m, p2/m, z10.h, z21.h
// CHECK-ENCODING: [0x49,0x55,0x95,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1955549 <unknown>

umopa   za3.s, p3/m, p7/m, z13.h, z8.h  // 10100001-10001000-11101101-10101011
// CHECK-INST: umopa   za3.s, p3/m, p7/m, z13.h, z8.h
// CHECK-ENCODING: [0xab,0xed,0x88,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a188edab <unknown>

umopa   za3.s, p7/m, p7/m, z31.h, z31.h  // 10100001-10011111-11111111-11101011
// CHECK-INST: umopa   za3.s, p7/m, p7/m, z31.h, z31.h
// CHECK-ENCODING: [0xeb,0xff,0x9f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a19fffeb <unknown>

umopa   za1.s, p3/m, p0/m, z17.h, z16.h  // 10100001-10010000-00001110-00101001
// CHECK-INST: umopa   za1.s, p3/m, p0/m, z17.h, z16.h
// CHECK-ENCODING: [0x29,0x0e,0x90,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1900e29 <unknown>

umopa   za1.s, p1/m, p4/m, z1.h, z30.h  // 10100001-10011110-10000100-00101001
// CHECK-INST: umopa   za1.s, p1/m, p4/m, z1.h, z30.h
// CHECK-ENCODING: [0x29,0x84,0x9e,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a19e8429 <unknown>

umopa   za0.s, p5/m, p2/m, z19.h, z20.h  // 10100001-10010100-01010110-01101000
// CHECK-INST: umopa   za0.s, p5/m, p2/m, z19.h, z20.h
// CHECK-ENCODING: [0x68,0x56,0x94,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1945668 <unknown>

umopa   za0.s, p6/m, p0/m, z12.h, z2.h  // 10100001-10000010-00011001-10001000
// CHECK-INST: umopa   za0.s, p6/m, p0/m, z12.h, z2.h
// CHECK-ENCODING: [0x88,0x19,0x82,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1821988 <unknown>

umopa   za1.s, p2/m, p6/m, z1.h, z26.h  // 10100001-10011010-11001000-00101001
// CHECK-INST: umopa   za1.s, p2/m, p6/m, z1.h, z26.h
// CHECK-ENCODING: [0x29,0xc8,0x9a,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a19ac829 <unknown>

umopa   za1.s, p2/m, p0/m, z22.h, z30.h  // 10100001-10011110-00001010-11001001
// CHECK-INST: umopa   za1.s, p2/m, p0/m, z22.h, z30.h
// CHECK-ENCODING: [0xc9,0x0a,0x9e,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a19e0ac9 <unknown>

umopa   za2.s, p5/m, p7/m, z9.h, z1.h  // 10100001-10000001-11110101-00101010
// CHECK-INST: umopa   za2.s, p5/m, p7/m, z9.h, z1.h
// CHECK-ENCODING: [0x2a,0xf5,0x81,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a181f52a <unknown>

umopa   za3.s, p2/m, p5/m, z12.h, z11.h  // 10100001-10001011-10101001-10001011
// CHECK-INST: umopa   za3.s, p2/m, p5/m, z12.h, z11.h
// CHECK-ENCODING: [0x8b,0xa9,0x8b,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a18ba98b <unknown>

