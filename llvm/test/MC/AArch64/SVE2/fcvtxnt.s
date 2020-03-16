// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN


fcvtxnt z0.s, p0/m, z1.d
// CHECK-INST: fcvtxnt z0.s, p0/m, z1.d
// CHECK-ENCODING: [0x20,0xa0,0x0a,0x64]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 a0 0a 64 <unknown>

fcvtxnt z30.s, p7/m, z31.d
// CHECK-INST: fcvtxnt z30.s, p7/m, z31.d
// CHECK-ENCODING: [0xfe,0xbf,0x0a,0x64]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: fe bf 0a 64 <unknown>
