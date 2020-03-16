// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

sminv b0, p7, z31.b
// CHECK-INST: sminv	b0, p7, z31.b
// CHECK-ENCODING: [0xe0,0x3f,0x0a,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 3f 0a 04 <unknown>

sminv h0, p7, z31.h
// CHECK-INST: sminv	h0, p7, z31.h
// CHECK-ENCODING: [0xe0,0x3f,0x4a,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 3f 4a 04 <unknown>

sminv s0, p7, z31.s
// CHECK-INST: sminv	s0, p7, z31.s
// CHECK-ENCODING: [0xe0,0x3f,0x8a,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 3f 8a 04 <unknown>

sminv d0, p7, z31.d
// CHECK-INST: sminv	d0, p7, z31.d
// CHECK-ENCODING: [0xe0,0x3f,0xca,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 3f ca 04 <unknown>
