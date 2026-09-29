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

addva   za0.d, p0/m, p0/m, z0.d
// CHECK-INST: addva   za0.d, p0/m, p0/m, z0.d
// CHECK-ENCODING: [0x00,0x00,0xd1,0xc0]
// CHECK-ERROR: instruction requires: sme-i16i64
// CHECK-UNKNOWN: c0d10000 <unknown>

addva   za5.d, p5/m, p2/m, z10.d
// CHECK-INST: addva   za5.d, p5/m, p2/m, z10.d
// CHECK-ENCODING: [0x45,0x55,0xd1,0xc0]
// CHECK-ERROR: instruction requires: sme-i16i64
// CHECK-UNKNOWN: c0d15545 <unknown>

addva   za7.d, p3/m, p7/m, z13.d
// CHECK-INST: addva   za7.d, p3/m, p7/m, z13.d
// CHECK-ENCODING: [0xa7,0xed,0xd1,0xc0]
// CHECK-ERROR: instruction requires: sme-i16i64
// CHECK-UNKNOWN: c0d1eda7 <unknown>

addva   za7.d, p7/m, p7/m, z31.d
// CHECK-INST: addva   za7.d, p7/m, p7/m, z31.d
// CHECK-ENCODING: [0xe7,0xff,0xd1,0xc0]
// CHECK-ERROR: instruction requires: sme-i16i64
// CHECK-UNKNOWN: c0d1ffe7 <unknown>

addva   za5.d, p3/m, p0/m, z17.d
// CHECK-INST: addva   za5.d, p3/m, p0/m, z17.d
// CHECK-ENCODING: [0x25,0x0e,0xd1,0xc0]
// CHECK-ERROR: instruction requires: sme-i16i64
// CHECK-UNKNOWN: c0d10e25 <unknown>

addva   za1.d, p1/m, p4/m, z1.d
// CHECK-INST: addva   za1.d, p1/m, p4/m, z1.d
// CHECK-ENCODING: [0x21,0x84,0xd1,0xc0]
// CHECK-ERROR: instruction requires: sme-i16i64
// CHECK-UNKNOWN: c0d18421 <unknown>

addva   za0.d, p5/m, p2/m, z19.d
// CHECK-INST: addva   za0.d, p5/m, p2/m, z19.d
// CHECK-ENCODING: [0x60,0x56,0xd1,0xc0]
// CHECK-ERROR: instruction requires: sme-i16i64
// CHECK-UNKNOWN: c0d15660 <unknown>

addva   za0.d, p6/m, p0/m, z12.d
// CHECK-INST: addva   za0.d, p6/m, p0/m, z12.d
// CHECK-ENCODING: [0x80,0x19,0xd1,0xc0]
// CHECK-ERROR: instruction requires: sme-i16i64
// CHECK-UNKNOWN: c0d11980 <unknown>

addva   za1.d, p2/m, p6/m, z1.d
// CHECK-INST: addva   za1.d, p2/m, p6/m, z1.d
// CHECK-ENCODING: [0x21,0xc8,0xd1,0xc0]
// CHECK-ERROR: instruction requires: sme-i16i64
// CHECK-UNKNOWN: c0d1c821 <unknown>

addva   za5.d, p2/m, p0/m, z22.d
// CHECK-INST: addva   za5.d, p2/m, p0/m, z22.d
// CHECK-ENCODING: [0xc5,0x0a,0xd1,0xc0]
// CHECK-ERROR: instruction requires: sme-i16i64
// CHECK-UNKNOWN: c0d10ac5 <unknown>

addva   za2.d, p5/m, p7/m, z9.d
// CHECK-INST: addva   za2.d, p5/m, p7/m, z9.d
// CHECK-ENCODING: [0x22,0xf5,0xd1,0xc0]
// CHECK-ERROR: instruction requires: sme-i16i64
// CHECK-UNKNOWN: c0d1f522 <unknown>

addva   za7.d, p2/m, p5/m, z12.d
// CHECK-INST: addva   za7.d, p2/m, p5/m, z12.d
// CHECK-ENCODING: [0x87,0xa9,0xd1,0xc0]
// CHECK-ERROR: instruction requires: sme-i16i64
// CHECK-UNKNOWN: c0d1a987 <unknown>
