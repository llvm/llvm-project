// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

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
