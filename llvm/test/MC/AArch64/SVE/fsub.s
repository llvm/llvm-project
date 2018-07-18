// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

fsub    z0.h, p0/m, z0.h, #0.500000000000000
// CHECK-INST: fsub	z0.h, p0/m, z0.h, #0.5
// CHECK-ENCODING: [0x00,0x80,0x59,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 59 65 <unknown>

fsub    z0.h, p0/m, z0.h, #0.5
// CHECK-INST: fsub	z0.h, p0/m, z0.h, #0.5
// CHECK-ENCODING: [0x00,0x80,0x59,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 59 65 <unknown>

fsub    z0.s, p0/m, z0.s, #0.5
// CHECK-INST: fsub	z0.s, p0/m, z0.s, #0.5
// CHECK-ENCODING: [0x00,0x80,0x99,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 99 65 <unknown>

fsub    z0.d, p0/m, z0.d, #0.5
// CHECK-INST: fsub	z0.d, p0/m, z0.d, #0.5
// CHECK-ENCODING: [0x00,0x80,0xd9,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 d9 65 <unknown>

fsub    z31.h, p7/m, z31.h, #1.000000000000000
// CHECK-INST: fsub	z31.h, p7/m, z31.h, #1.0
// CHECK-ENCODING: [0x3f,0x9c,0x59,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 3f 9c 59 65 <unknown>

fsub    z31.h, p7/m, z31.h, #1.0
// CHECK-INST: fsub	z31.h, p7/m, z31.h, #1.0
// CHECK-ENCODING: [0x3f,0x9c,0x59,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 3f 9c 59 65 <unknown>

fsub    z31.s, p7/m, z31.s, #1.0
// CHECK-INST: fsub	z31.s, p7/m, z31.s, #1.0
// CHECK-ENCODING: [0x3f,0x9c,0x99,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 3f 9c 99 65 <unknown>

fsub    z31.d, p7/m, z31.d, #1.0
// CHECK-INST: fsub	z31.d, p7/m, z31.d, #1.0
// CHECK-ENCODING: [0x3f,0x9c,0xd9,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 3f 9c d9 65 <unknown>

fsub    z0.h, p7/m, z0.h, z31.h
// CHECK-INST: fsub	z0.h, p7/m, z0.h, z31.h
// CHECK-ENCODING: [0xe0,0x9f,0x41,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 9f 41 65 <unknown>

fsub    z0.s, p7/m, z0.s, z31.s
// CHECK-INST: fsub	z0.s, p7/m, z0.s, z31.s
// CHECK-ENCODING: [0xe0,0x9f,0x81,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 9f 81 65 <unknown>

fsub    z0.d, p7/m, z0.d, z31.d
// CHECK-INST: fsub	z0.d, p7/m, z0.d, z31.d
// CHECK-ENCODING: [0xe0,0x9f,0xc1,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 9f c1 65 <unknown>
