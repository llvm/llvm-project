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


bmopa   za0.s, p0/m, p0/m, z0.s, z0.s  // 10000000-10000000-00000000-00001000
// CHECK-INST: bmopa   za0.s, p0/m, p0/m, z0.s, z0.s
// CHECK-ENCODING: [0x08,0x00,0x80,0x80]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: 80800008 <unknown>

bmopa   za1.s, p5/m, p2/m, z10.s, z21.s  // 10000000-10010101-01010101-01001001
// CHECK-INST: bmopa   za1.s, p5/m, p2/m, z10.s, z21.s
// CHECK-ENCODING: [0x49,0x55,0x95,0x80]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: 80955549 <unknown>

bmopa   za3.s, p3/m, p7/m, z13.s, z8.s  // 10000000-10001000-11101101-10101011
// CHECK-INST: bmopa   za3.s, p3/m, p7/m, z13.s, z8.s
// CHECK-ENCODING: [0xab,0xed,0x88,0x80]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: 8088edab <unknown>

bmopa   za3.s, p7/m, p7/m, z31.s, z31.s  // 10000000-10011111-11111111-11101011
// CHECK-INST: bmopa   za3.s, p7/m, p7/m, z31.s, z31.s
// CHECK-ENCODING: [0xeb,0xff,0x9f,0x80]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: 809fffeb <unknown>

bmopa   za1.s, p3/m, p0/m, z17.s, z16.s  // 10000000-10010000-00001110-00101001
// CHECK-INST: bmopa   za1.s, p3/m, p0/m, z17.s, z16.s
// CHECK-ENCODING: [0x29,0x0e,0x90,0x80]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: 80900e29 <unknown>

bmopa   za1.s, p1/m, p4/m, z1.s, z30.s  // 10000000-10011110-10000100-00101001
// CHECK-INST: bmopa   za1.s, p1/m, p4/m, z1.s, z30.s
// CHECK-ENCODING: [0x29,0x84,0x9e,0x80]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: 809e8429 <unknown>

bmopa   za0.s, p5/m, p2/m, z19.s, z20.s  // 10000000-10010100-01010110-01101000
// CHECK-INST: bmopa   za0.s, p5/m, p2/m, z19.s, z20.s
// CHECK-ENCODING: [0x68,0x56,0x94,0x80]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: 80945668 <unknown>

bmopa   za0.s, p6/m, p0/m, z12.s, z2.s  // 10000000-10000010-00011001-10001000
// CHECK-INST: bmopa   za0.s, p6/m, p0/m, z12.s, z2.s
// CHECK-ENCODING: [0x88,0x19,0x82,0x80]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: 80821988 <unknown>

bmopa   za1.s, p2/m, p6/m, z1.s, z26.s  // 10000000-10011010-11001000-00101001
// CHECK-INST: bmopa   za1.s, p2/m, p6/m, z1.s, z26.s
// CHECK-ENCODING: [0x29,0xc8,0x9a,0x80]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: 809ac829 <unknown>

bmopa   za1.s, p2/m, p0/m, z22.s, z30.s  // 10000000-10011110-00001010-11001001
// CHECK-INST: bmopa   za1.s, p2/m, p0/m, z22.s, z30.s
// CHECK-ENCODING: [0xc9,0x0a,0x9e,0x80]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: 809e0ac9 <unknown>

bmopa   za2.s, p5/m, p7/m, z9.s, z1.s  // 10000000-10000001-11110101-00101010
// CHECK-INST: bmopa   za2.s, p5/m, p7/m, z9.s, z1.s
// CHECK-ENCODING: [0x2a,0xf5,0x81,0x80]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: 8081f52a <unknown>

bmopa   za3.s, p2/m, p5/m, z12.s, z11.s  // 10000000-10001011-10101001-10001011
// CHECK-INST: bmopa   za3.s, p2/m, p5/m, z12.s, z11.s
// CHECK-ENCODING: [0x8b,0xa9,0x8b,0x80]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: 808ba98b <unknown>

