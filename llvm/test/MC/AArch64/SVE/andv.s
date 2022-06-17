// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:   | llvm-objdump -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN

andv b0, p7, z31.b
// CHECK-INST: andv	b0, p7, z31.b
// CHECK-ENCODING: [0xe0,0x3f,0x1a,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 3f 1a 04 <unknown>

andv h0, p7, z31.h
// CHECK-INST: andv	h0, p7, z31.h
// CHECK-ENCODING: [0xe0,0x3f,0x5a,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 3f 5a 04 <unknown>

andv s0, p7, z31.s
// CHECK-INST: andv	s0, p7, z31.s
// CHECK-ENCODING: [0xe0,0x3f,0x9a,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 3f 9a 04 <unknown>

andv d0, p7, z31.d
// CHECK-INST: andv	d0, p7, z31.d
// CHECK-ENCODING: [0xe0,0x3f,0xda,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 3f da 04 <unknown>
