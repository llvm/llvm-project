// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2p2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2p2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2p2 < %s \
// RUN:        | llvm-objdump -d --mattr=-sve2p2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sve2p2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

expand  z0.b, p0, z0.b  // 00000101-00110001-10000000-00000000
// CHECK-INST: expand  z0.b, p0, z0.b
// CHECK-ENCODING: [0x00,0x80,0x31,0x05]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 05318000 <unknown>

expand  z21.h, p5, z10.h  // 00000101-01110001-10010101-01010101
// CHECK-INST: expand  z21.h, p5, z10.h
// CHECK-ENCODING: [0x55,0x95,0x71,0x05]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 05719555 <unknown>

expand  z23.s, p3, z13.s  // 00000101-10110001-10001101-10110111
// CHECK-INST: expand  z23.s, p3, z13.s
// CHECK-ENCODING: [0xb7,0x8d,0xb1,0x05]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 05b18db7 <unknown>

expand  z31.d, p7, z31.d  // 00000101-11110001-10011111-11111111
// CHECK-INST: expand  z31.d, p7, z31.d
// CHECK-ENCODING: [0xff,0x9f,0xf1,0x05]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 05f19fff <unknown>