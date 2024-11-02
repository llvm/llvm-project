// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

uminv b0, p7, z31.b
// CHECK-INST: uminv	b0, p7, z31.b
// CHECK-ENCODING: [0xe0,0x3f,0x0b,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 3f 0b 04 <unknown>

uminv h0, p7, z31.h
// CHECK-INST: uminv	h0, p7, z31.h
// CHECK-ENCODING: [0xe0,0x3f,0x4b,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 3f 4b 04 <unknown>

uminv s0, p7, z31.s
// CHECK-INST: uminv	s0, p7, z31.s
// CHECK-ENCODING: [0xe0,0x3f,0x8b,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 3f 8b 04 <unknown>

uminv d0, p7, z31.d
// CHECK-INST: uminv	d0, p7, z31.d
// CHECK-ENCODING: [0xe0,0x3f,0xcb,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 3f cb 04 <unknown>
