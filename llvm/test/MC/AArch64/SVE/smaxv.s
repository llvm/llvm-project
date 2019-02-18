// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

smaxv b0, p7, z31.b
// CHECK-INST: smaxv	b0, p7, z31.b
// CHECK-ENCODING: [0xe0,0x3f,0x08,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 3f 08 04 <unknown>

smaxv h0, p7, z31.h
// CHECK-INST: smaxv	h0, p7, z31.h
// CHECK-ENCODING: [0xe0,0x3f,0x48,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 3f 48 04 <unknown>

smaxv s0, p7, z31.s
// CHECK-INST: smaxv	s0, p7, z31.s
// CHECK-ENCODING: [0xe0,0x3f,0x88,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 3f 88 04 <unknown>

smaxv d0, p7, z31.d
// CHECK-INST: smaxv	d0, p7, z31.d
// CHECK-ENCODING: [0xe0,0x3f,0xc8,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 3f c8 04 <unknown>
