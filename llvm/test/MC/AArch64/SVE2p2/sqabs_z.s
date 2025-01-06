// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2p2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2p2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2p2 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sve2p2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

sqabs   z0.b, p0/z, z0.b  // 01000100-00001010-10100000-00000000
// CHECK-INST: sqabs   z0.b, p0/z, z0.b
// CHECK-ENCODING: [0x00,0xa0,0x0a,0x44]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 440aa000 <unknown>

sqabs   z21.h, p5/z, z10.h  // 01000100-01001010-10110101-01010101
// CHECK-INST: sqabs   z21.h, p5/z, z10.h
// CHECK-ENCODING: [0x55,0xb5,0x4a,0x44]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 444ab555 <unknown>

sqabs   z23.s, p3/z, z13.s  // 01000100-10001010-10101101-10110111
// CHECK-INST: sqabs   z23.s, p3/z, z13.s
// CHECK-ENCODING: [0xb7,0xad,0x8a,0x44]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 448aadb7 <unknown>

sqabs   z31.d, p7/z, z31.d  // 01000100-11001010-10111111-11111111
// CHECK-INST: sqabs   z31.d, p7/z, z31.d
// CHECK-ENCODING: [0xff,0xbf,0xca,0x44]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 44cabfff <unknown>