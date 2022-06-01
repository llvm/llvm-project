// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN


histcnt z0.s, p0/z, z1.s, z2.s
// CHECK-INST: histcnt z0.s, p0/z, z1.s, z2.s
// CHECK-ENCODING: [0x20,0xc0,0xa2,0x45]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 c0 a2 45 <unknown>

histcnt z29.d, p7/z, z30.d, z31.d
// CHECK-INST: histcnt z29.d, p7/z, z30.d, z31.d
// CHECK-ENCODING: [0xdd,0xdf,0xff,0x45]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: dd df ff 45 <unknown>
