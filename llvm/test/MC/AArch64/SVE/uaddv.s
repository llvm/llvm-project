// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

uaddv d0, p7, z31.b
// CHECK-INST: uaddv	d0, p7, z31.b
// CHECK-ENCODING: [0xe0,0x3f,0x01,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 3f 01 04 <unknown>

uaddv d0, p7, z31.h
// CHECK-INST: uaddv	d0, p7, z31.h
// CHECK-ENCODING: [0xe0,0x3f,0x41,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 3f 41 04 <unknown>

uaddv d0, p7, z31.s
// CHECK-INST: uaddv	d0, p7, z31.s
// CHECK-ENCODING: [0xe0,0x3f,0x81,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 3f 81 04 <unknown>

uaddv d0, p7, z31.d
// CHECK-INST: uaddv	d0, p7, z31.d
// CHECK-ENCODING: [0xe0,0x3f,0xc1,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 3f c1 04 <unknown>
