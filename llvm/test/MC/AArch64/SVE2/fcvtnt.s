// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN


fcvtnt z0.h, p0/m, z1.s
// CHECK-INST: fcvtnt z0.h, p0/m, z1.s
// CHECK-ENCODING: [0x20,0xa0,0x88,0x64]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 a0 88 64 <unknown>

fcvtnt z30.s, p7/m, z31.d
// CHECK-INST: fcvtnt z30.s, p7/m, z31.d
// CHECK-ENCODING: [0xfe,0xbf,0xca,0x64]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: fe bf ca 64 <unknown>
