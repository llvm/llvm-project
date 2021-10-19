// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme < %s \
// RUN:        | llvm-objdump -d --mattr=+sme - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

addva   za0.s, p0/m, p0/m, z0.s
// CHECK-INST: addva   za0.s, p0/m, p0/m, z0.s
// CHECK-ENCODING: [0x00,0x00,0x91,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 00 91 c0 <unknown>

addva   za1.s, p5/m, p2/m, z10.s
// CHECK-INST: addva   za1.s, p5/m, p2/m, z10.s
// CHECK-ENCODING: [0x41,0x55,0x91,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 41 55 91 c0 <unknown>

addva   za3.s, p3/m, p7/m, z13.s
// CHECK-INST: addva   za3.s, p3/m, p7/m, z13.s
// CHECK-ENCODING: [0xa3,0xed,0x91,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a3 ed 91 c0 <unknown>

addva   za3.s, p7/m, p7/m, z31.s
// CHECK-INST: addva   za3.s, p7/m, p7/m, z31.s
// CHECK-ENCODING: [0xe3,0xff,0x91,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: e3 ff 91 c0 <unknown>

addva   za1.s, p3/m, p0/m, z17.s
// CHECK-INST: addva   za1.s, p3/m, p0/m, z17.s
// CHECK-ENCODING: [0x21,0x0e,0x91,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 0e 91 c0 <unknown>

addva   za1.s, p1/m, p4/m, z1.s
// CHECK-INST: addva   za1.s, p1/m, p4/m, z1.s
// CHECK-ENCODING: [0x21,0x84,0x91,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 84 91 c0 <unknown>

addva   za0.s, p5/m, p2/m, z19.s
// CHECK-INST: addva   za0.s, p5/m, p2/m, z19.s
// CHECK-ENCODING: [0x60,0x56,0x91,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 60 56 91 c0 <unknown>

addva   za0.s, p6/m, p0/m, z12.s
// CHECK-INST: addva   za0.s, p6/m, p0/m, z12.s
// CHECK-ENCODING: [0x80,0x19,0x91,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 19 91 c0 <unknown>

addva   za1.s, p2/m, p6/m, z1.s
// CHECK-INST: addva   za1.s, p2/m, p6/m, z1.s
// CHECK-ENCODING: [0x21,0xc8,0x91,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 c8 91 c0 <unknown>

addva   za1.s, p2/m, p0/m, z22.s
// CHECK-INST: addva   za1.s, p2/m, p0/m, z22.s
// CHECK-ENCODING: [0xc1,0x0a,0x91,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: c1 0a 91 c0 <unknown>

addva   za2.s, p5/m, p7/m, z9.s
// CHECK-INST: addva   za2.s, p5/m, p7/m, z9.s
// CHECK-ENCODING: [0x22,0xf5,0x91,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 f5 91 c0 <unknown>

addva   za3.s, p2/m, p5/m, z12.s
// CHECK-INST: addva   za3.s, p2/m, p5/m, z12.s
// CHECK-ENCODING: [0x83,0xa9,0x91,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 83 a9 91 c0 <unknown>
