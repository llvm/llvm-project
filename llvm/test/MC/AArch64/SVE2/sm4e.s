// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2-sm4 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2-sm4 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2-sm4 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2-sm4 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN


sm4e z0.s, z0.s, z31.s
// CHECK-INST: sm4e z0.s, z0.s, z31.s
// CHECK-ENCODING: [0xe0,0xe3,0x23,0x45]
// CHECK-ERROR: instruction requires: sve2-sm4
// CHECK-UNKNOWN: e0 e3 23 45 <unknown>
