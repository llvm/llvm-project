// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme-i16i64 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme-i16i64 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme-i16i64 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme-i16i64 < %s \
// RUN:   | llvm-objdump -d --mattr=-sme - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme-i16i64 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme-i16i64 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

umops   za0.d, p0/m, p0/m, z0.h, z0.h
// CHECK-INST: umops   za0.d, p0/m, p0/m, z0.h, z0.h
// CHECK-ENCODING: [0x10,0x00,0xe0,0xa1]
// CHECK-ERROR: instruction requires: sme-i16i64
// CHECK-UNKNOWN: a1e00010 <unknown>

umops   za5.d, p5/m, p2/m, z10.h, z21.h
// CHECK-INST: umops   za5.d, p5/m, p2/m, z10.h, z21.h
// CHECK-ENCODING: [0x55,0x55,0xf5,0xa1]
// CHECK-ERROR: instruction requires: sme-i16i64
// CHECK-UNKNOWN: a1f55555 <unknown>

umops   za7.d, p3/m, p7/m, z13.h, z8.h
// CHECK-INST: umops   za7.d, p3/m, p7/m, z13.h, z8.h
// CHECK-ENCODING: [0xb7,0xed,0xe8,0xa1]
// CHECK-ERROR: instruction requires: sme-i16i64
// CHECK-UNKNOWN: a1e8edb7 <unknown>

umops   za7.d, p7/m, p7/m, z31.h, z31.h
// CHECK-INST: umops   za7.d, p7/m, p7/m, z31.h, z31.h
// CHECK-ENCODING: [0xf7,0xff,0xff,0xa1]
// CHECK-ERROR: instruction requires: sme-i16i64
// CHECK-UNKNOWN: a1fffff7 <unknown>

umops   za5.d, p3/m, p0/m, z17.h, z16.h
// CHECK-INST: umops   za5.d, p3/m, p0/m, z17.h, z16.h
// CHECK-ENCODING: [0x35,0x0e,0xf0,0xa1]
// CHECK-ERROR: instruction requires: sme-i16i64
// CHECK-UNKNOWN: a1f00e35 <unknown>

umops   za1.d, p1/m, p4/m, z1.h, z30.h
// CHECK-INST: umops   za1.d, p1/m, p4/m, z1.h, z30.h
// CHECK-ENCODING: [0x31,0x84,0xfe,0xa1]
// CHECK-ERROR: instruction requires: sme-i16i64
// CHECK-UNKNOWN: a1fe8431 <unknown>

umops   za0.d, p5/m, p2/m, z19.h, z20.h
// CHECK-INST: umops   za0.d, p5/m, p2/m, z19.h, z20.h
// CHECK-ENCODING: [0x70,0x56,0xf4,0xa1]
// CHECK-ERROR: instruction requires: sme-i16i64
// CHECK-UNKNOWN: a1f45670 <unknown>

umops   za0.d, p6/m, p0/m, z12.h, z2.h
// CHECK-INST: umops   za0.d, p6/m, p0/m, z12.h, z2.h
// CHECK-ENCODING: [0x90,0x19,0xe2,0xa1]
// CHECK-ERROR: instruction requires: sme-i16i64
// CHECK-UNKNOWN: a1e21990 <unknown>

umops   za1.d, p2/m, p6/m, z1.h, z26.h
// CHECK-INST: umops   za1.d, p2/m, p6/m, z1.h, z26.h
// CHECK-ENCODING: [0x31,0xc8,0xfa,0xa1]
// CHECK-ERROR: instruction requires: sme-i16i64
// CHECK-UNKNOWN: a1fac831 <unknown>

umops   za5.d, p2/m, p0/m, z22.h, z30.h
// CHECK-INST: umops   za5.d, p2/m, p0/m, z22.h, z30.h
// CHECK-ENCODING: [0xd5,0x0a,0xfe,0xa1]
// CHECK-ERROR: instruction requires: sme-i16i64
// CHECK-UNKNOWN: a1fe0ad5 <unknown>

umops   za2.d, p5/m, p7/m, z9.h, z1.h
// CHECK-INST: umops   za2.d, p5/m, p7/m, z9.h, z1.h
// CHECK-ENCODING: [0x32,0xf5,0xe1,0xa1]
// CHECK-ERROR: instruction requires: sme-i16i64
// CHECK-UNKNOWN: a1e1f532 <unknown>

umops   za7.d, p2/m, p5/m, z12.h, z11.h
// CHECK-INST: umops   za7.d, p2/m, p5/m, z12.h, z11.h
// CHECK-ENCODING: [0x97,0xa9,0xeb,0xa1]
// CHECK-ERROR: instruction requires: sme-i16i64
// CHECK-UNKNOWN: a1eba997 <unknown>
