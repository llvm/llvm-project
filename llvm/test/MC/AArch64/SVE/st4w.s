// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

st4w    { z0.s, z1.s, z2.s, z3.s }, p0, [x0]
// CHECK-INST: st4w    { z0.s, z1.s, z2.s, z3.s }, p0, [x0]
// CHECK-ENCODING: [0x00,0xe0,0x70,0xe5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 e0 70 e5 <unknown>

st4w    { z23.s, z24.s, z25.s, z26.s }, p3, [x13, #-32, mul vl]
// CHECK-INST: st4w    { z23.s, z24.s, z25.s, z26.s }, p3, [x13, #-32, mul vl]
// CHECK-ENCODING: [0xb7,0xed,0x78,0xe5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: b7 ed 78 e5 <unknown>

st4w    { z21.s, z22.s, z23.s, z24.s }, p5, [x10, #20, mul vl]
// CHECK-INST: st4w    { z21.s, z22.s, z23.s, z24.s }, p5, [x10, #20, mul vl]
// CHECK-ENCODING: [0x55,0xf5,0x75,0xe5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 f5 75 e5 <unknown>
