// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

orv b0, p7, z31.b
// CHECK-INST: orv	b0, p7, z31.b
// CHECK-ENCODING: [0xe0,0x3f,0x18,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 3f 18 04 <unknown>

orv h0, p7, z31.h
// CHECK-INST: orv	h0, p7, z31.h
// CHECK-ENCODING: [0xe0,0x3f,0x58,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 3f 58 04 <unknown>

orv s0, p7, z31.s
// CHECK-INST: orv	s0, p7, z31.s
// CHECK-ENCODING: [0xe0,0x3f,0x98,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 3f 98 04 <unknown>

orv d0, p7, z31.d
// CHECK-INST: orv	d0, p7, z31.d
// CHECK-ENCODING: [0xe0,0x3f,0xd8,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 3f d8 04 <unknown>
