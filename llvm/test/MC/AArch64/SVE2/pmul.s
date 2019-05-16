// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d -mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

pmul z0.b, z1.b, z2.b
// CHECK-INST: pmul z0.b, z1.b, z2.b
// CHECK-ENCODING: [0x20,0x64,0x22,0x04]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 64 22 04 <unknown>

pmul z29.b, z30.b, z31.b
// CHECK-INST: pmul z29.b, z30.b, z31.b
// CHECK-ENCODING: [0xdd,0x67,0x3f,0x04]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: dd 67 3f 04 <unknown>
