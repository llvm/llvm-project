// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

ld4d    { z0.d, z1.d, z2.d, z3.d }, p0/z, [x0, x0, lsl #3]
// CHECK-INST: ld4d    { z0.d, z1.d, z2.d, z3.d }, p0/z, [x0, x0, lsl #3]
// CHECK-ENCODING: [0x00,0xc0,0xe0,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 c0 e0 a5 <unknown>

ld4d    { z5.d, z6.d, z7.d, z8.d }, p3/z, [x17, x16, lsl #3]
// CHECK-INST: ld4d    { z5.d, z6.d, z7.d, z8.d }, p3/z, [x17, x16, lsl #3]
// CHECK-ENCODING: [0x25,0xce,0xf0,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 25 ce f0 a5 <unknown>

ld4d    { z0.d, z1.d, z2.d, z3.d }, p0/z, [x0]
// CHECK-INST: ld4d    { z0.d, z1.d, z2.d, z3.d }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xe0,0xe0,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 e0 e0 a5 <unknown>

ld4d    { z23.d, z24.d, z25.d, z26.d }, p3/z, [x13, #-32, mul vl]
// CHECK-INST: ld4d    { z23.d, z24.d, z25.d, z26.d }, p3/z, [x13, #-32, mul vl]
// CHECK-ENCODING: [0xb7,0xed,0xe8,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: b7 ed e8 a5 <unknown>

ld4d    { z21.d, z22.d, z23.d, z24.d }, p5/z, [x10, #20, mul vl]
// CHECK-INST: ld4d    { z21.d, z22.d, z23.d, z24.d }, p5/z, [x10, #20, mul vl]
// CHECK-ENCODING: [0x55,0xf5,0xe5,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 f5 e5 a5 <unknown>
