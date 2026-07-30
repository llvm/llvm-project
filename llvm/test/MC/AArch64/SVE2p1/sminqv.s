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

sminqv  v0.8h, p0, z0.h  // 00000100-01001110-00100000-00000000
// CHECK-INST: sminqv  v0.8h, p0, z0.h
// CHECK-ENCODING: [0x00,0x20,0x4e,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 044e2000 <unknown>

sminqv  v21.8h, p5, z10.h  // 00000100-01001110-00110101-01010101
// CHECK-INST: sminqv  v21.8h, p5, z10.h
// CHECK-ENCODING: [0x55,0x35,0x4e,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 044e3555 <unknown>

sminqv  v23.8h, p3, z13.h  // 00000100-01001110-00101101-10110111
// CHECK-INST: sminqv  v23.8h, p3, z13.h
// CHECK-ENCODING: [0xb7,0x2d,0x4e,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 044e2db7 <unknown>

sminqv  v31.8h, p7, z31.h  // 00000100-01001110-00111111-11111111
// CHECK-INST: sminqv  v31.8h, p7, z31.h
// CHECK-ENCODING: [0xff,0x3f,0x4e,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 044e3fff <unknown>

sminqv  v0.4s, p0, z0.s  // 00000100-10001110-00100000-00000000
// CHECK-INST: sminqv  v0.4s, p0, z0.s
// CHECK-ENCODING: [0x00,0x20,0x8e,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 048e2000 <unknown>

sminqv  v21.4s, p5, z10.s  // 00000100-10001110-00110101-01010101
// CHECK-INST: sminqv  v21.4s, p5, z10.s
// CHECK-ENCODING: [0x55,0x35,0x8e,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 048e3555 <unknown>

sminqv  v23.4s, p3, z13.s  // 00000100-10001110-00101101-10110111
// CHECK-INST: sminqv  v23.4s, p3, z13.s
// CHECK-ENCODING: [0xb7,0x2d,0x8e,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 048e2db7 <unknown>

sminqv  v31.4s, p7, z31.s  // 00000100-10001110-00111111-11111111
// CHECK-INST: sminqv  v31.4s, p7, z31.s
// CHECK-ENCODING: [0xff,0x3f,0x8e,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 048e3fff <unknown>

sminqv  v0.2d, p0, z0.d  // 00000100-11001110-00100000-00000000
// CHECK-INST: sminqv  v0.2d, p0, z0.d
// CHECK-ENCODING: [0x00,0x20,0xce,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 04ce2000 <unknown>

sminqv  v21.2d, p5, z10.d  // 00000100-11001110-00110101-01010101
// CHECK-INST: sminqv  v21.2d, p5, z10.d
// CHECK-ENCODING: [0x55,0x35,0xce,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 04ce3555 <unknown>

sminqv  v23.2d, p3, z13.d  // 00000100-11001110-00101101-10110111
// CHECK-INST: sminqv  v23.2d, p3, z13.d
// CHECK-ENCODING: [0xb7,0x2d,0xce,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 04ce2db7 <unknown>

sminqv  v31.2d, p7, z31.d  // 00000100-11001110-00111111-11111111
// CHECK-INST: sminqv  v31.2d, p7, z31.d
// CHECK-ENCODING: [0xff,0x3f,0xce,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 04ce3fff <unknown>

sminqv  v0.16b, p0, z0.b  // 00000100-00001110-00100000-00000000
// CHECK-INST: sminqv  v0.16b, p0, z0.b
// CHECK-ENCODING: [0x00,0x20,0x0e,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 040e2000 <unknown>

sminqv  v21.16b, p5, z10.b  // 00000100-00001110-00110101-01010101
// CHECK-INST: sminqv  v21.16b, p5, z10.b
// CHECK-ENCODING: [0x55,0x35,0x0e,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 040e3555 <unknown>

sminqv  v23.16b, p3, z13.b  // 00000100-00001110-00101101-10110111
// CHECK-INST: sminqv  v23.16b, p3, z13.b
// CHECK-ENCODING: [0xb7,0x2d,0x0e,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 040e2db7 <unknown>

sminqv  v31.16b, p7, z31.b  // 00000100-00001110-00111111-11111111
// CHECK-INST: sminqv  v31.16b, p7, z31.b
// CHECK-ENCODING: [0xff,0x3f,0x0e,0x04]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 040e3fff <unknown>
