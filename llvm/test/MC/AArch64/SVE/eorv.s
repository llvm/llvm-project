// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

eorv b0, p7, z31.b
// CHECK-INST: eorv	b0, p7, z31.b
// CHECK-ENCODING: [0xe0,0x3f,0x19,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 3f 19 04 <unknown>

eorv h0, p7, z31.h
// CHECK-INST: eorv	h0, p7, z31.h
// CHECK-ENCODING: [0xe0,0x3f,0x59,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 3f 59 04 <unknown>

eorv s0, p7, z31.s
// CHECK-INST: eorv	s0, p7, z31.s
// CHECK-ENCODING: [0xe0,0x3f,0x99,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 3f 99 04 <unknown>

eorv d0, p7, z31.d
// CHECK-INST: eorv	d0, p7, z31.d
// CHECK-ENCODING: [0xe0,0x3f,0xd9,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 3f d9 04 <unknown>
