// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2-sha3 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2-sha3 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2-sha3 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2-sha3 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN


rax1 z0.d, z1.d, z31.d
// CHECK-INST: rax1 z0.d, z1.d, z31.d
// CHECK-ENCODING: [0x20,0xf4,0x3f,0x45]
// CHECK-ERROR: instruction requires: sve2-sha3
// CHECK-UNKNOWN: 20 f4 3f 45 <unknown>
