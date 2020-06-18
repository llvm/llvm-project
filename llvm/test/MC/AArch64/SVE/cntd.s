// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

cntd  x0
// CHECK-INST: cntd	x0
// CHECK-ENCODING: [0xe0,0xe3,0xe0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 e3 e0 04 <unknown>

cntd  x0, all
// CHECK-INST: cntd	x0
// CHECK-ENCODING: [0xe0,0xe3,0xe0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 e3 e0 04 <unknown>

cntd  x0, all, mul #1
// CHECK-INST: cntd	x0
// CHECK-ENCODING: [0xe0,0xe3,0xe0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 e3 e0 04 <unknown>

cntd  x0, all, mul #16
// CHECK-INST: cntd	x0, all, mul #16
// CHECK-ENCODING: [0xe0,0xe3,0xef,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 e3 ef 04 <unknown>

cntd  x0, pow2
// CHECK-INST: cntd	x0, pow2
// CHECK-ENCODING: [0x00,0xe0,0xe0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 e0 e0 04 <unknown>

cntd  x0, #28
// CHECK-INST: cntd	x0, #28
// CHECK-ENCODING: [0x80,0xe3,0xe0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 80 e3 e0 04 <unknown>
