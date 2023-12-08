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

andqv   v0.8h, p0, z0.h  // 00000100-01011110-00100000-00000000
// CHECK-INST: andqv   v0.8h, p0, z0.h
// CHECK-ENCODING: [0x00,0x20,0x5e,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 045e2000 <unknown>

andqv   v21.8h, p5, z10.h  // 00000100-01011110-00110101-01010101
// CHECK-INST: andqv   v21.8h, p5, z10.h
// CHECK-ENCODING: [0x55,0x35,0x5e,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 045e3555 <unknown>

andqv   v23.8h, p3, z13.h  // 00000100-01011110-00101101-10110111
// CHECK-INST: andqv   v23.8h, p3, z13.h
// CHECK-ENCODING: [0xb7,0x2d,0x5e,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 045e2db7 <unknown>

andqv   v31.8h, p7, z31.h  // 00000100-01011110-00111111-11111111
// CHECK-INST: andqv   v31.8h, p7, z31.h
// CHECK-ENCODING: [0xff,0x3f,0x5e,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 045e3fff <unknown>

andqv   v0.4s, p0, z0.s  // 00000100-10011110-00100000-00000000
// CHECK-INST: andqv   v0.4s, p0, z0.s
// CHECK-ENCODING: [0x00,0x20,0x9e,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 049e2000 <unknown>

andqv   v21.4s, p5, z10.s  // 00000100-10011110-00110101-01010101
// CHECK-INST: andqv   v21.4s, p5, z10.s
// CHECK-ENCODING: [0x55,0x35,0x9e,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 049e3555 <unknown>

andqv   v23.4s, p3, z13.s  // 00000100-10011110-00101101-10110111
// CHECK-INST: andqv   v23.4s, p3, z13.s
// CHECK-ENCODING: [0xb7,0x2d,0x9e,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 049e2db7 <unknown>

andqv   v31.4s, p7, z31.s  // 00000100-10011110-00111111-11111111
// CHECK-INST: andqv   v31.4s, p7, z31.s
// CHECK-ENCODING: [0xff,0x3f,0x9e,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 049e3fff <unknown>

andqv   v0.2d, p0, z0.d  // 00000100-11011110-00100000-00000000
// CHECK-INST: andqv   v0.2d, p0, z0.d
// CHECK-ENCODING: [0x00,0x20,0xde,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 04de2000 <unknown>

andqv   v21.2d, p5, z10.d  // 00000100-11011110-00110101-01010101
// CHECK-INST: andqv   v21.2d, p5, z10.d
// CHECK-ENCODING: [0x55,0x35,0xde,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 04de3555 <unknown>

andqv   v23.2d, p3, z13.d  // 00000100-11011110-00101101-10110111
// CHECK-INST: andqv   v23.2d, p3, z13.d
// CHECK-ENCODING: [0xb7,0x2d,0xde,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 04de2db7 <unknown>

andqv   v31.2d, p7, z31.d  // 00000100-11011110-00111111-11111111
// CHECK-INST: andqv   v31.2d, p7, z31.d
// CHECK-ENCODING: [0xff,0x3f,0xde,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 04de3fff <unknown>

andqv   v0.16b, p0, z0.b  // 00000100-00011110-00100000-00000000
// CHECK-INST: andqv   v0.16b, p0, z0.b
// CHECK-ENCODING: [0x00,0x20,0x1e,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 041e2000 <unknown>

andqv   v21.16b, p5, z10.b  // 00000100-00011110-00110101-01010101
// CHECK-INST: andqv   v21.16b, p5, z10.b
// CHECK-ENCODING: [0x55,0x35,0x1e,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 041e3555 <unknown>

andqv   v23.16b, p3, z13.b  // 00000100-00011110-00101101-10110111
// CHECK-INST: andqv   v23.16b, p3, z13.b
// CHECK-ENCODING: [0xb7,0x2d,0x1e,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 041e2db7 <unknown>

andqv   v31.16b, p7, z31.b  // 00000100-00011110-00111111-11111111
// CHECK-INST: andqv   v31.16b, p7, z31.b
// CHECK-ENCODING: [0xff,0x3f,0x1e,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 041e3fff <unknown>
