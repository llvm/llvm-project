// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d -mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

sqrdmulh z0.b, z1.b, z2.b
// CHECK-INST: sqrdmulh	z0.b, z1.b, z2.b
// CHECK-ENCODING: [0x20,0x74,0x22,0x04]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 74 22 04 <unknown>

sqrdmulh z0.h, z1.h, z2.h
// CHECK-INST: sqrdmulh	z0.h, z1.h, z2.h
// CHECK-ENCODING: [0x20,0x74,0x62,0x04]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 74 62 04 <unknown>

sqrdmulh z29.s, z30.s, z31.s
// CHECK-INST: sqrdmulh z29.s, z30.s, z31.s
// CHECK-ENCODING: [0xdd,0x77,0xbf,0x04]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: dd 77 bf 04 <unknown>

sqrdmulh z31.d, z31.d, z31.d
// CHECK-INST: sqrdmulh z31.d, z31.d, z31.d
// CHECK-ENCODING: [0xff,0x77,0xff,0x04]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: ff 77 ff 04 <unknown>

sqrdmulh z0.h, z1.h, z7.h[7]
// CHECK-INST: sqrdmulh	z0.h, z1.h, z7.h[7]
// CHECK-ENCODING: [0x20,0xf4,0x7f,0x44]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 f4 7f 44 <unknown>

sqrdmulh z0.s, z1.s, z7.s[3]
// CHECK-INST: sqrdmulh	z0.s, z1.s, z7.s[3]
// CHECK-ENCODING: [0x20,0xf4,0xbf,0x44]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 f4 bf 44 <unknown>

sqrdmulh z0.d, z1.d, z15.d[1]
// CHECK-INST: sqrdmulh	z0.d, z1.d, z15.d[1]
// CHECK-ENCODING: [0x20,0xf4,0xff,0x44]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 f4 ff 44 <unknown>
