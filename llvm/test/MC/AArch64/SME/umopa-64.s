// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme-i64 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme-i64 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme-i64 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme-i64 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme-i64 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme-i64 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

umopa   za0.d, p0/m, p0/m, z0.h, z0.h
// CHECK-INST: umopa   za0.d, p0/m, p0/m, z0.h, z0.h
// CHECK-ENCODING: [0x00,0x00,0xe0,0xa1]
// CHECK-ERROR: instruction requires: sme-i64
// CHECK-UNKNOWN: 00 00 e0 a1 <unknown>

umopa   za5.d, p5/m, p2/m, z10.h, z21.h
// CHECK-INST: umopa   za5.d, p5/m, p2/m, z10.h, z21.h
// CHECK-ENCODING: [0x45,0x55,0xf5,0xa1]
// CHECK-ERROR: instruction requires: sme-i64
// CHECK-UNKNOWN: 45 55 f5 a1 <unknown>

umopa   za7.d, p3/m, p7/m, z13.h, z8.h
// CHECK-INST: umopa   za7.d, p3/m, p7/m, z13.h, z8.h
// CHECK-ENCODING: [0xa7,0xed,0xe8,0xa1]
// CHECK-ERROR: instruction requires: sme-i64
// CHECK-UNKNOWN: a7 ed e8 a1 <unknown>

umopa   za7.d, p7/m, p7/m, z31.h, z31.h
// CHECK-INST: umopa   za7.d, p7/m, p7/m, z31.h, z31.h
// CHECK-ENCODING: [0xe7,0xff,0xff,0xa1]
// CHECK-ERROR: instruction requires: sme-i64
// CHECK-UNKNOWN: e7 ff ff a1 <unknown>

umopa   za5.d, p3/m, p0/m, z17.h, z16.h
// CHECK-INST: umopa   za5.d, p3/m, p0/m, z17.h, z16.h
// CHECK-ENCODING: [0x25,0x0e,0xf0,0xa1]
// CHECK-ERROR: instruction requires: sme-i64
// CHECK-UNKNOWN: 25 0e f0 a1 <unknown>

umopa   za1.d, p1/m, p4/m, z1.h, z30.h
// CHECK-INST: umopa   za1.d, p1/m, p4/m, z1.h, z30.h
// CHECK-ENCODING: [0x21,0x84,0xfe,0xa1]
// CHECK-ERROR: instruction requires: sme-i64
// CHECK-UNKNOWN: 21 84 fe a1 <unknown>

umopa   za0.d, p5/m, p2/m, z19.h, z20.h
// CHECK-INST: umopa   za0.d, p5/m, p2/m, z19.h, z20.h
// CHECK-ENCODING: [0x60,0x56,0xf4,0xa1]
// CHECK-ERROR: instruction requires: sme-i64
// CHECK-UNKNOWN: 60 56 f4 a1 <unknown>

umopa   za0.d, p6/m, p0/m, z12.h, z2.h
// CHECK-INST: umopa   za0.d, p6/m, p0/m, z12.h, z2.h
// CHECK-ENCODING: [0x80,0x19,0xe2,0xa1]
// CHECK-ERROR: instruction requires: sme-i64
// CHECK-UNKNOWN: 80 19 e2 a1 <unknown>

umopa   za1.d, p2/m, p6/m, z1.h, z26.h
// CHECK-INST: umopa   za1.d, p2/m, p6/m, z1.h, z26.h
// CHECK-ENCODING: [0x21,0xc8,0xfa,0xa1]
// CHECK-ERROR: instruction requires: sme-i64
// CHECK-UNKNOWN: 21 c8 fa a1 <unknown>

umopa   za5.d, p2/m, p0/m, z22.h, z30.h
// CHECK-INST: umopa   za5.d, p2/m, p0/m, z22.h, z30.h
// CHECK-ENCODING: [0xc5,0x0a,0xfe,0xa1]
// CHECK-ERROR: instruction requires: sme-i64
// CHECK-UNKNOWN: c5 0a fe a1 <unknown>

umopa   za2.d, p5/m, p7/m, z9.h, z1.h
// CHECK-INST: umopa   za2.d, p5/m, p7/m, z9.h, z1.h
// CHECK-ENCODING: [0x22,0xf5,0xe1,0xa1]
// CHECK-ERROR: instruction requires: sme-i64
// CHECK-UNKNOWN: 22 f5 e1 a1 <unknown>

umopa   za7.d, p2/m, p5/m, z12.h, z11.h
// CHECK-INST: umopa   za7.d, p2/m, p5/m, z12.h, z11.h
// CHECK-ENCODING: [0x87,0xa9,0xeb,0xa1]
// CHECK-ERROR: instruction requires: sme-i64
// CHECK-UNKNOWN: 87 a9 eb a1 <unknown>
