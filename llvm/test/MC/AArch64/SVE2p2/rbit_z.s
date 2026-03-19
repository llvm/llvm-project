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

rbit    z0.b, p0/z, z0.b  // 00000101-00100111-10100000-00000000
// CHECK-INST: rbit    z0.b, p0/z, z0.b
// CHECK-ENCODING: [0x00,0xa0,0x27,0x05]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 0527a000 <unknown>

rbit    z21.b, p5/z, z10.b  // 00000101-00100111-10110101-01010101
// CHECK-INST: rbit    z21.b, p5/z, z10.b
// CHECK-ENCODING: [0x55,0xb5,0x27,0x05]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 0527b555 <unknown>

rbit    z23.h, p3/z, z13.h  // 00000101-01100111-10101101-10110111
// CHECK-INST: rbit    z23.h, p3/z, z13.h
// CHECK-ENCODING: [0xb7,0xad,0x67,0x05]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 0567adb7 <unknown>

rbit    z23.s, p3/z, z13.s  // 00000101-10100111-10101101-10110111
// CHECK-INST: rbit    z23.s, p3/z, z13.s
// CHECK-ENCODING: [0xb7,0xad,0xa7,0x05]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 05a7adb7 <unknown>

rbit    z31.d, p7/z, z31.d  // 00000101-11100111-10111111-11111111
// CHECK-INST: rbit    z31.d, p7/z, z31.d
// CHECK-ENCODING: [0xff,0xbf,0xe7,0x05]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 05e7bfff <unknown>