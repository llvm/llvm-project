// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

ld4w    { z0.s, z1.s, z2.s, z3.s }, p0/z, [x0, x0, lsl #2]
// CHECK-INST: ld4w    { z0.s, z1.s, z2.s, z3.s }, p0/z, [x0, x0, lsl #2]
// CHECK-ENCODING: [0x00,0xc0,0x60,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 c0 60 a5 <unknown>

ld4w    { z5.s, z6.s, z7.s, z8.s }, p3/z, [x17, x16, lsl #2]
// CHECK-INST: ld4w    { z5.s, z6.s, z7.s, z8.s }, p3/z, [x17, x16, lsl #2]
// CHECK-ENCODING: [0x25,0xce,0x70,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 25 ce 70 a5 <unknown>

ld4w    { z0.s, z1.s, z2.s, z3.s }, p0/z, [x0]
// CHECK-INST: ld4w    { z0.s, z1.s, z2.s, z3.s }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xe0,0x60,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 e0 60 a5 <unknown>

ld4w    { z23.s, z24.s, z25.s, z26.s }, p3/z, [x13, #-32, mul vl]
// CHECK-INST: ld4w    { z23.s, z24.s, z25.s, z26.s }, p3/z, [x13, #-32, mul vl]
// CHECK-ENCODING: [0xb7,0xed,0x68,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: b7 ed 68 a5 <unknown>

ld4w    { z21.s, z22.s, z23.s, z24.s }, p5/z, [x10, #20, mul vl]
// CHECK-INST: ld4w    { z21.s, z22.s, z23.s, z24.s }, p5/z, [x10, #20, mul vl]
// CHECK-ENCODING: [0x55,0xf5,0x65,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 f5 65 a5 <unknown>
