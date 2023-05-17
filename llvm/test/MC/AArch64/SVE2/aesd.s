// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2-aes < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2-aes < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2-aes - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2-aes < %s \
// RUN:   | llvm-objdump -d --mattr=-sve2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN


aesd z0.b, z0.b, z31.b
// CHECK-INST: aesd z0.b, z0.b, z31.b
// CHECK-ENCODING: [0xe0,0xe7,0x22,0x45]
// CHECK-ERROR: instruction requires: sve2-aes
// CHECK-UNKNOWN: 4522e7e0 <unknown>
