// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

sqdmulh z0.b, z1.b, z2.b
// CHECK-INST: sqdmulh	z0.b, z1.b, z2.b
// CHECK-ENCODING: [0x20,0x70,0x22,0x04]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 70 22 04 <unknown>

sqdmulh z0.h, z1.h, z2.h
// CHECK-INST: sqdmulh	z0.h, z1.h, z2.h
// CHECK-ENCODING: [0x20,0x70,0x62,0x04]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 70 62 04 <unknown>

sqdmulh z29.s, z30.s, z31.s
// CHECK-INST: sqdmulh z29.s, z30.s, z31.s
// CHECK-ENCODING: [0xdd,0x73,0xbf,0x04]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: dd 73 bf 04 <unknown>

sqdmulh z31.d, z31.d, z31.d
// CHECK-INST: sqdmulh z31.d, z31.d, z31.d
// CHECK-ENCODING: [0xff,0x73,0xff,0x04]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: ff 73 ff 04 <unknown>

sqdmulh z0.h, z1.h, z7.h[7]
// CHECK-INST: sqdmulh	z0.h, z1.h, z7.h[7]
// CHECK-ENCODING: [0x20,0xf0,0x7f,0x44]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 f0 7f 44 <unknown>

sqdmulh z0.s, z1.s, z7.s[3]
// CHECK-INST: sqdmulh	z0.s, z1.s, z7.s[3]
// CHECK-ENCODING: [0x20,0xf0,0xbf,0x44]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 f0 bf 44 <unknown>

sqdmulh z0.d, z1.d, z15.d[1]
// CHECK-INST: sqdmulh	z0.d, z1.d, z15.d[1]
// CHECK-ENCODING: [0x20,0xf0,0xff,0x44]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 f0 ff 44 <unknown>
