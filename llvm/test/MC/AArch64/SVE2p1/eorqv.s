// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p1 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p1 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2p1 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p1 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2p1,-sve2p1 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p1 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2p1 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

eorqv   v0.8h, p0, z0.h  // 00000100-01011101-00100000-00000000
// CHECK-INST: eorqv   v0.8h, p0, z0.h
// CHECK-ENCODING: [0x00,0x20,0x5d,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 045d2000 <unknown>

eorqv   v21.8h, p5, z10.h  // 00000100-01011101-00110101-01010101
// CHECK-INST: eorqv   v21.8h, p5, z10.h
// CHECK-ENCODING: [0x55,0x35,0x5d,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 045d3555 <unknown>

eorqv   v23.8h, p3, z13.h  // 00000100-01011101-00101101-10110111
// CHECK-INST: eorqv   v23.8h, p3, z13.h
// CHECK-ENCODING: [0xb7,0x2d,0x5d,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 045d2db7 <unknown>

eorqv   v31.8h, p7, z31.h  // 00000100-01011101-00111111-11111111
// CHECK-INST: eorqv   v31.8h, p7, z31.h
// CHECK-ENCODING: [0xff,0x3f,0x5d,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 045d3fff <unknown>

eorqv   v0.4s, p0, z0.s  // 00000100-10011101-00100000-00000000
// CHECK-INST: eorqv   v0.4s, p0, z0.s
// CHECK-ENCODING: [0x00,0x20,0x9d,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 049d2000 <unknown>

eorqv   v21.4s, p5, z10.s  // 00000100-10011101-00110101-01010101
// CHECK-INST: eorqv   v21.4s, p5, z10.s
// CHECK-ENCODING: [0x55,0x35,0x9d,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 049d3555 <unknown>

eorqv   v23.4s, p3, z13.s  // 00000100-10011101-00101101-10110111
// CHECK-INST: eorqv   v23.4s, p3, z13.s
// CHECK-ENCODING: [0xb7,0x2d,0x9d,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 049d2db7 <unknown>

eorqv   v31.4s, p7, z31.s  // 00000100-10011101-00111111-11111111
// CHECK-INST: eorqv   v31.4s, p7, z31.s
// CHECK-ENCODING: [0xff,0x3f,0x9d,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 049d3fff <unknown>

eorqv   v0.2d, p0, z0.d  // 00000100-11011101-00100000-00000000
// CHECK-INST: eorqv   v0.2d, p0, z0.d
// CHECK-ENCODING: [0x00,0x20,0xdd,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 04dd2000 <unknown>

eorqv   v21.2d, p5, z10.d  // 00000100-11011101-00110101-01010101
// CHECK-INST: eorqv   v21.2d, p5, z10.d
// CHECK-ENCODING: [0x55,0x35,0xdd,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 04dd3555 <unknown>

eorqv   v23.2d, p3, z13.d  // 00000100-11011101-00101101-10110111
// CHECK-INST: eorqv   v23.2d, p3, z13.d
// CHECK-ENCODING: [0xb7,0x2d,0xdd,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 04dd2db7 <unknown>

eorqv   v31.2d, p7, z31.d  // 00000100-11011101-00111111-11111111
// CHECK-INST: eorqv   v31.2d, p7, z31.d
// CHECK-ENCODING: [0xff,0x3f,0xdd,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 04dd3fff <unknown>

eorqv   v0.16b, p0, z0.b  // 00000100-00011101-00100000-00000000
// CHECK-INST: eorqv   v0.16b, p0, z0.b
// CHECK-ENCODING: [0x00,0x20,0x1d,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 041d2000 <unknown>

eorqv   v21.16b, p5, z10.b  // 00000100-00011101-00110101-01010101
// CHECK-INST: eorqv   v21.16b, p5, z10.b
// CHECK-ENCODING: [0x55,0x35,0x1d,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 041d3555 <unknown>

eorqv   v23.16b, p3, z13.b  // 00000100-00011101-00101101-10110111
// CHECK-INST: eorqv   v23.16b, p3, z13.b
// CHECK-ENCODING: [0xb7,0x2d,0x1d,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 041d2db7 <unknown>

eorqv   v31.16b, p7, z31.b  // 00000100-00011101-00111111-11111111
// CHECK-INST: eorqv   v31.16b, p7, z31.b
// CHECK-ENCODING: [0xff,0x3f,0x1d,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 041d3fff <unknown>
