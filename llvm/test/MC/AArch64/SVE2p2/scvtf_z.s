// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2p2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2p2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2p2 < %s \
// RUN:        | llvm-objdump -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sve2p2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

// convert to half

scvtf   z0.h, p0/z, z0.h  // 01100100-01011100-11000000-00000000
// CHECK-INST: scvtf   z0.h, p0/z, z0.h
// CHECK-ENCODING: [0x00,0xc0,0x5c,0x64]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 645cc000 <unknown>

scvtf   z21.h, p5/z, z10.s  // 01100100-01011101-10010101-01010101
// CHECK-INST: scvtf   z21.h, p5/z, z10.s
// CHECK-ENCODING: [0x55,0x95,0x5d,0x64]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 645d9555 <unknown>

scvtf   z31.h, p7/z, z31.d  // 01100100-01011101-11011111-11111111
// CHECK-INST: scvtf   z31.h, p7/z, z31.d
// CHECK-ENCODING: [0xff,0xdf,0x5d,0x64]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 645ddfff <unknown>

// convert to single

scvtf   z0.s, p0/z, z0.s  // 01100100-10011101-10000000-00000000
// CHECK-INST: scvtf   z0.s, p0/z, z0.s
// CHECK-ENCODING: [0x00,0x80,0x9d,0x64]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 649d8000 <unknown>

scvtf   z23.s, p3/z, z13.d  // 01100100-11011101-10001101-10110111
// CHECK-INST: scvtf   z23.s, p3/z, z13.d
// CHECK-ENCODING: [0xb7,0x8d,0xdd,0x64]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 64dd8db7 <unknown>

// convert to double

scvtf   z21.d, p5/z, z10.s  // 01100100-11011100-10010101-01010101
// CHECK-INST: scvtf   z21.d, p5/z, z10.s
// CHECK-ENCODING: [0x55,0x95,0xdc,0x64]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 64dc9555 <unknown>

scvtf   z31.d, p7/z, z31.d  // 01100100-11011101-11011111-11111111
// CHECK-INST: scvtf   z31.d, p7/z, z31.d
// CHECK-ENCODING: [0xff,0xdf,0xdd,0x64]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 64dddfff <unknown>