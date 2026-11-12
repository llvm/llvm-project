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

umaxqv  v0.8h, p0, z0.h  // 00000100-01001101-00100000-00000000
// CHECK-INST: umaxqv  v0.8h, p0, z0.h
// CHECK-ENCODING: [0x00,0x20,0x4d,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 044d2000 <unknown>

umaxqv  v21.8h, p5, z10.h  // 00000100-01001101-00110101-01010101
// CHECK-INST: umaxqv  v21.8h, p5, z10.h
// CHECK-ENCODING: [0x55,0x35,0x4d,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 044d3555 <unknown>

umaxqv  v23.8h, p3, z13.h  // 00000100-01001101-00101101-10110111
// CHECK-INST: umaxqv  v23.8h, p3, z13.h
// CHECK-ENCODING: [0xb7,0x2d,0x4d,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 044d2db7 <unknown>

umaxqv  v31.8h, p7, z31.h  // 00000100-01001101-00111111-11111111
// CHECK-INST: umaxqv  v31.8h, p7, z31.h
// CHECK-ENCODING: [0xff,0x3f,0x4d,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 044d3fff <unknown>

umaxqv  v0.4s, p0, z0.s  // 00000100-10001101-00100000-00000000
// CHECK-INST: umaxqv  v0.4s, p0, z0.s
// CHECK-ENCODING: [0x00,0x20,0x8d,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 048d2000 <unknown>

umaxqv  v21.4s, p5, z10.s  // 00000100-10001101-00110101-01010101
// CHECK-INST: umaxqv  v21.4s, p5, z10.s
// CHECK-ENCODING: [0x55,0x35,0x8d,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 048d3555 <unknown>

umaxqv  v23.4s, p3, z13.s  // 00000100-10001101-00101101-10110111
// CHECK-INST: umaxqv  v23.4s, p3, z13.s
// CHECK-ENCODING: [0xb7,0x2d,0x8d,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 048d2db7 <unknown>

umaxqv  v31.4s, p7, z31.s  // 00000100-10001101-00111111-11111111
// CHECK-INST: umaxqv  v31.4s, p7, z31.s
// CHECK-ENCODING: [0xff,0x3f,0x8d,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 048d3fff <unknown>

umaxqv  v0.2d, p0, z0.d  // 00000100-11001101-00100000-00000000
// CHECK-INST: umaxqv  v0.2d, p0, z0.d
// CHECK-ENCODING: [0x00,0x20,0xcd,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 04cd2000 <unknown>

umaxqv  v21.2d, p5, z10.d  // 00000100-11001101-00110101-01010101
// CHECK-INST: umaxqv  v21.2d, p5, z10.d
// CHECK-ENCODING: [0x55,0x35,0xcd,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 04cd3555 <unknown>

umaxqv  v23.2d, p3, z13.d  // 00000100-11001101-00101101-10110111
// CHECK-INST: umaxqv  v23.2d, p3, z13.d
// CHECK-ENCODING: [0xb7,0x2d,0xcd,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 04cd2db7 <unknown>

umaxqv  v31.2d, p7, z31.d  // 00000100-11001101-00111111-11111111
// CHECK-INST: umaxqv  v31.2d, p7, z31.d
// CHECK-ENCODING: [0xff,0x3f,0xcd,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 04cd3fff <unknown>

umaxqv  v0.16b, p0, z0.b  // 00000100-00001101-00100000-00000000
// CHECK-INST: umaxqv  v0.16b, p0, z0.b
// CHECK-ENCODING: [0x00,0x20,0x0d,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 040d2000 <unknown>

umaxqv  v21.16b, p5, z10.b  // 00000100-00001101-00110101-01010101
// CHECK-INST: umaxqv  v21.16b, p5, z10.b
// CHECK-ENCODING: [0x55,0x35,0x0d,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 040d3555 <unknown>

umaxqv  v23.16b, p3, z13.b  // 00000100-00001101-00101101-10110111
// CHECK-INST: umaxqv  v23.16b, p3, z13.b
// CHECK-ENCODING: [0xb7,0x2d,0x0d,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 040d2db7 <unknown>

umaxqv  v31.16b, p7, z31.b  // 00000100-00001101-00111111-11111111
// CHECK-INST: umaxqv  v31.16b, p7, z31.b
// CHECK-ENCODING: [0xff,0x3f,0x0d,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 040d3fff <unknown>
