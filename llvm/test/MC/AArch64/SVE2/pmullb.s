// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN


pmullb z0.h, z1.b, z2.b
// CHECK-INST: pmullb z0.h, z1.b, z2.b
// CHECK-ENCODING: [0x20,0x68,0x42,0x45]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 68 42 45 <unknown>

pmullb z31.d, z31.s, z31.s
// CHECK-INST: pmullb z31.d, z31.s, z31.s
// CHECK-ENCODING: [0xff,0x6b,0xdf,0x45]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: ff 6b df 45 <unknown>
