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

ucvtf   z0.h, p0/z, z0.h  // 01100100-01011100-11100000-00000000
// CHECK-INST: ucvtf   z0.h, p0/z, z0.h
// CHECK-ENCODING: [0x00,0xe0,0x5c,0x64]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 645ce000 <unknown>

ucvtf   z21.h, p5/z, z10.s  // 01100100-01011101-10110101-01010101
// CHECK-INST: ucvtf   z21.h, p5/z, z10.s
// CHECK-ENCODING: [0x55,0xb5,0x5d,0x64]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 645db555 <unknown>

ucvtf   z31.h, p7/z, z31.d  // 01100100-01011101-11111111-11111111
// CHECK-INST: ucvtf   z31.h, p7/z, z31.d
// CHECK-ENCODING: [0xff,0xff,0x5d,0x64]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 645dffff <unknown>

// convert to single

ucvtf   z23.s, p3/z, z13.s  // 01100100-10011101-10101101-10110111
// CHECK-INST: ucvtf   z23.s, p3/z, z13.s
// CHECK-ENCODING: [0xb7,0xad,0x9d,0x64]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 649dadb7 <unknown>

ucvtf   z21.s, p5/z, z10.d  // 01100100-11011101-10110101-01010101
// CHECK-INST: ucvtf   z21.s, p5/z, z10.d
// CHECK-ENCODING: [0x55,0xb5,0xdd,0x64]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 64ddb555 <unknown>

// convert to double

ucvtf   z0.d, p0/z, z0.s  // 01100100-11011100-10100000-00000000
// CHECK-INST: ucvtf   z0.d, p0/z, z0.s
// CHECK-ENCODING: [0x00,0xa0,0xdc,0x64]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 64dca000 <unknown>

ucvtf   z31.d, p7/z, z31.d  // 01100100-11011101-11111111-11111111
// CHECK-INST: ucvtf   z31.d, p7/z, z31.d
// CHECK-ENCODING: [0xff,0xff,0xdd,0x64]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 64ddffff <unknown>