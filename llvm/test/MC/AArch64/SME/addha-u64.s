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

addha   za0.d, p0/m, p0/m, z0.d
// CHECK-INST: addha   za0.d, p0/m, p0/m, z0.d
// CHECK-ENCODING: [0x00,0x00,0xd0,0xc0]
// CHECK-ERROR: instruction requires: sme-i16i64
// CHECK-UNKNOWN: c0d00000 <unknown>

addha   za5.d, p5/m, p2/m, z10.d
// CHECK-INST: addha   za5.d, p5/m, p2/m, z10.d
// CHECK-ENCODING: [0x45,0x55,0xd0,0xc0]
// CHECK-ERROR: instruction requires: sme-i16i64
// CHECK-UNKNOWN: c0d05545 <unknown>

addha   za7.d, p3/m, p7/m, z13.d
// CHECK-INST: addha   za7.d, p3/m, p7/m, z13.d
// CHECK-ENCODING: [0xa7,0xed,0xd0,0xc0]
// CHECK-ERROR: instruction requires: sme-i16i64
// CHECK-UNKNOWN: c0d0eda7 <unknown>

addha   za7.d, p7/m, p7/m, z31.d
// CHECK-INST: addha   za7.d, p7/m, p7/m, z31.d
// CHECK-ENCODING: [0xe7,0xff,0xd0,0xc0]
// CHECK-ERROR: instruction requires: sme-i16i64
// CHECK-UNKNOWN: c0d0ffe7 <unknown>

addha   za5.d, p3/m, p0/m, z17.d
// CHECK-INST: addha   za5.d, p3/m, p0/m, z17.d
// CHECK-ENCODING: [0x25,0x0e,0xd0,0xc0]
// CHECK-ERROR: instruction requires: sme-i16i64
// CHECK-UNKNOWN: c0d00e25 <unknown>

addha   za1.d, p1/m, p4/m, z1.d
// CHECK-INST: addha   za1.d, p1/m, p4/m, z1.d
// CHECK-ENCODING: [0x21,0x84,0xd0,0xc0]
// CHECK-ERROR: instruction requires: sme-i16i64
// CHECK-UNKNOWN: c0d08421 <unknown>

addha   za0.d, p5/m, p2/m, z19.d
// CHECK-INST: addha   za0.d, p5/m, p2/m, z19.d
// CHECK-ENCODING: [0x60,0x56,0xd0,0xc0]
// CHECK-ERROR: instruction requires: sme-i16i64
// CHECK-UNKNOWN: c0d05660 <unknown>

addha   za0.d, p6/m, p0/m, z12.d
// CHECK-INST: addha   za0.d, p6/m, p0/m, z12.d
// CHECK-ENCODING: [0x80,0x19,0xd0,0xc0]
// CHECK-ERROR: instruction requires: sme-i16i64
// CHECK-UNKNOWN: c0d01980 <unknown>

addha   za1.d, p2/m, p6/m, z1.d
// CHECK-INST: addha   za1.d, p2/m, p6/m, z1.d
// CHECK-ENCODING: [0x21,0xc8,0xd0,0xc0]
// CHECK-ERROR: instruction requires: sme-i16i64
// CHECK-UNKNOWN: c0d0c821 <unknown>

addha   za5.d, p2/m, p0/m, z22.d
// CHECK-INST: addha   za5.d, p2/m, p0/m, z22.d
// CHECK-ENCODING: [0xc5,0x0a,0xd0,0xc0]
// CHECK-ERROR: instruction requires: sme-i16i64
// CHECK-UNKNOWN: c0d00ac5 <unknown>

addha   za2.d, p5/m, p7/m, z9.d
// CHECK-INST: addha   za2.d, p5/m, p7/m, z9.d
// CHECK-ENCODING: [0x22,0xf5,0xd0,0xc0]
// CHECK-ERROR: instruction requires: sme-i16i64
// CHECK-UNKNOWN: c0d0f522 <unknown>

addha   za7.d, p2/m, p5/m, z12.d
// CHECK-INST: addha   za7.d, p2/m, p5/m, z12.d
// CHECK-ENCODING: [0x87,0xa9,0xd0,0xc0]
// CHECK-ERROR: instruction requires: sme-i16i64
// CHECK-UNKNOWN: c0d0a987 <unknown>
