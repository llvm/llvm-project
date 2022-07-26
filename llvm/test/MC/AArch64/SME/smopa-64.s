// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme-i64 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme-i64 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme-i64 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme-i64 < %s \
// RUN:   | llvm-objdump -d --mattr=-sme - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme-i64 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme-i64 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

smopa   za0.d, p0/m, p0/m, z0.h, z0.h
// CHECK-INST: smopa   za0.d, p0/m, p0/m, z0.h, z0.h
// CHECK-ENCODING: [0x00,0x00,0xc0,0xa0]
// CHECK-ERROR: instruction requires: sme-i64
// CHECK-UNKNOWN: a0c00000 <unknown>

smopa   za5.d, p5/m, p2/m, z10.h, z21.h
// CHECK-INST: smopa   za5.d, p5/m, p2/m, z10.h, z21.h
// CHECK-ENCODING: [0x45,0x55,0xd5,0xa0]
// CHECK-ERROR: instruction requires: sme-i64
// CHECK-UNKNOWN: a0d55545 <unknown>

smopa   za7.d, p3/m, p7/m, z13.h, z8.h
// CHECK-INST: smopa   za7.d, p3/m, p7/m, z13.h, z8.h
// CHECK-ENCODING: [0xa7,0xed,0xc8,0xa0]
// CHECK-ERROR: instruction requires: sme-i64
// CHECK-UNKNOWN: a0c8eda7 <unknown>

smopa   za7.d, p7/m, p7/m, z31.h, z31.h
// CHECK-INST: smopa   za7.d, p7/m, p7/m, z31.h, z31.h
// CHECK-ENCODING: [0xe7,0xff,0xdf,0xa0]
// CHECK-ERROR: instruction requires: sme-i64
// CHECK-UNKNOWN: a0dfffe7 <unknown>

smopa   za5.d, p3/m, p0/m, z17.h, z16.h
// CHECK-INST: smopa   za5.d, p3/m, p0/m, z17.h, z16.h
// CHECK-ENCODING: [0x25,0x0e,0xd0,0xa0]
// CHECK-ERROR: instruction requires: sme-i64
// CHECK-UNKNOWN: a0d00e25 <unknown>

smopa   za1.d, p1/m, p4/m, z1.h, z30.h
// CHECK-INST: smopa   za1.d, p1/m, p4/m, z1.h, z30.h
// CHECK-ENCODING: [0x21,0x84,0xde,0xa0]
// CHECK-ERROR: instruction requires: sme-i64
// CHECK-UNKNOWN: a0de8421 <unknown>

smopa   za0.d, p5/m, p2/m, z19.h, z20.h
// CHECK-INST: smopa   za0.d, p5/m, p2/m, z19.h, z20.h
// CHECK-ENCODING: [0x60,0x56,0xd4,0xa0]
// CHECK-ERROR: instruction requires: sme-i64
// CHECK-UNKNOWN: a0d45660 <unknown>

smopa   za0.d, p6/m, p0/m, z12.h, z2.h
// CHECK-INST: smopa   za0.d, p6/m, p0/m, z12.h, z2.h
// CHECK-ENCODING: [0x80,0x19,0xc2,0xa0]
// CHECK-ERROR: instruction requires: sme-i64
// CHECK-UNKNOWN: a0c21980 <unknown>

smopa   za1.d, p2/m, p6/m, z1.h, z26.h
// CHECK-INST: smopa   za1.d, p2/m, p6/m, z1.h, z26.h
// CHECK-ENCODING: [0x21,0xc8,0xda,0xa0]
// CHECK-ERROR: instruction requires: sme-i64
// CHECK-UNKNOWN: a0dac821 <unknown>

smopa   za5.d, p2/m, p0/m, z22.h, z30.h
// CHECK-INST: smopa   za5.d, p2/m, p0/m, z22.h, z30.h
// CHECK-ENCODING: [0xc5,0x0a,0xde,0xa0]
// CHECK-ERROR: instruction requires: sme-i64
// CHECK-UNKNOWN: a0de0ac5 <unknown>

smopa   za2.d, p5/m, p7/m, z9.h, z1.h
// CHECK-INST: smopa   za2.d, p5/m, p7/m, z9.h, z1.h
// CHECK-ENCODING: [0x22,0xf5,0xc1,0xa0]
// CHECK-ERROR: instruction requires: sme-i64
// CHECK-UNKNOWN: a0c1f522 <unknown>

smopa   za7.d, p2/m, p5/m, z12.h, z11.h
// CHECK-INST: smopa   za7.d, p2/m, p5/m, z12.h, z11.h
// CHECK-ENCODING: [0x87,0xa9,0xcb,0xa0]
// CHECK-ERROR: instruction requires: sme-i64
// CHECK-UNKNOWN: a0cba987 <unknown>
