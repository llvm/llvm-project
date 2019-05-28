// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2-sm4 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2-sm4 < %s \
// RUN:        | llvm-objdump -d -mattr=+sve2-sm4 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2-sm4 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN


sm4ekey z0.s, z1.s, z31.s
// CHECK-INST: sm4ekey z0.s, z1.s, z31.s
// CHECK-ENCODING: [0x20,0xf0,0x3f,0x45]
// CHECK-ERROR: instruction requires: sve2-sm4
// CHECK-UNKNOWN: 20 f0 3f 45 <unknown>
