// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

rdffrs  p0.b, p0/z
// CHECK-INST: rdffrs  p0.b, p0/z
// CHECK-ENCODING: [0x00,0xf0,0x58,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 f0 58 25 <unknown>

rdffrs  p15.b, p15/z
// CHECK-INST: rdffrs  p15.b, p15/z
// CHECK-ENCODING: [0xef,0xf1,0x58,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ef f1 58 25 <unknown>
