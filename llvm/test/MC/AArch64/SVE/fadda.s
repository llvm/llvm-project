// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

fadda h0, p7, h0, z31.h
// CHECK-INST: fadda	h0, p7, h0, z31.h
// CHECK-ENCODING: [0xe0,0x3f,0x58,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 3f 58 65 <unknown>

fadda s0, p7, s0, z31.s
// CHECK-INST: fadda	s0, p7, s0, z31.s
// CHECK-ENCODING: [0xe0,0x3f,0x98,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 3f 98 65 <unknown>

fadda d0, p7, d0, z31.d
// CHECK-INST: fadda	d0, p7, d0, z31.d
// CHECK-ENCODING: [0xe0,0x3f,0xd8,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 3f d8 65 <unknown>
