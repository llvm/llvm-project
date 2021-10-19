// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+streaming-sve < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

wrffr   p0.b
// CHECK-INST: wrffr   p0.b
// CHECK-ENCODING: [0x00,0x90,0x28,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 90 28 25 <unknown>

wrffr   p15.b
// CHECK-INST: wrffr   p15.b
// CHECK-ENCODING: [0xe0,0x91,0x28,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 91 28 25 <unknown>
